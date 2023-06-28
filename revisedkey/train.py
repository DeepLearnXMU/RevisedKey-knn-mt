import os
import sys
import torch
import torch.nn as nn
import numpy as np

import faiss
import time
import random
import pickle
from math import sqrt
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from revisedkey.model import Reviser
from revisedkey.dataset import RevisedkeyDataset
from revisedkey.utils import (
    DATASTORE_SIZE,
    TEMPERATURE,
    TOPK,
    setup_seed,
    memmap_to_tensor,
    State,
)

import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.revisedkey")


class Trainer:
    def __init__(self, args, model, vocab_size, train_dataset):
        self.args = args
        self.model = model
        self.vocab_size = vocab_size
        
        self.source_keys = train_dataset.source_keys
        self.target_keys = train_dataset.target_keys
        self.token_map = train_dataset.token_map

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))
 
        self.criterion = nn.MSELoss(reduction='none')
        self.global_step = 0

    def load_last_checkpoint(self):
        last_epoch = self.args.valid_interval_epoch
        interval = self.args.valid_interval_epoch
        while os.path.exists(f'{self.args.save_path}/checkpoint_{last_epoch + interval}.pt'):
            last_epoch += interval
        
        if os.path.exists(f'{self.args.save_path}/checkpoint_{last_epoch}.pt'):
            self.model.load_state_dict(torch.load(f'{self.args.save_path}/checkpoint_{last_epoch}.pt'))
            return last_epoch
        return 0


    def train(self, train_dataloader, valid_dataloader):
        train_state = State()
        best_valid_state = State()

        global_updates = 0
        for epoch in range(self.args.max_epoch):
            logger.info(f'begin epoch {epoch + 1}:')
            for step, batch in enumerate(train_dataloader):
                train_step_state = self.train_step(batch, epoch)
                train_state.add(train_step_state)
                global_updates += 1

                if global_updates % self.args.log_interval_update == 0:
                    train_state.mean()
                    logger.info(train_state.info(step=global_updates))

            logger.info(f'validate epoch {epoch + 1}:')
            if (epoch + 1) % self.args.valid_interval_epoch != 0 \
                or self.args.skip_build_datastore:

                valid_state = self.validate(train_dataloader, valid_dataloader, epoch=epoch, only_loss=True)
                logger.info(valid_state.info(epoch=epoch+1, valid=True))

            else:
                valid_state = self.validate(train_dataloader, valid_dataloader, epoch=epoch)
                logger.info(valid_state.info(epoch=epoch+1, valid=True))

    
    def compute_loss(self, batch, epoch, mode='train'):
        query_hidden = batch.query_hidden
        query_hidden_mse_expand = query_hidden.unsqueeze(1).expand_as(batch.key_source_hidden)

        target_key_hidden, increase_hidden = self.model.key_forward(
            source_hidden=batch.key_source_hidden,
            target_hidden=batch.key_target_hidden,
            token=batch.key_token,
            nearest_distance=batch.nearest_distance)

        mse_loss = self.criterion(query_hidden_mse_expand, target_key_hidden).sum(dim=-1)
        mse_loss = mse_loss - batch.deno_detail
        mse_loss.masked_fill_(mse_loss < 0, 0)
        mse_loss = mse_loss.mean()
        
        norm_loss = (increase_hidden ** 2).sum(dim=-1).mean()
        norm_loss = self.args.norm_weight * norm_loss

        loss = mse_loss + norm_loss
        state = State(loss, mse_loss, norm_loss, count=1)
        return state, loss

    
    def train_step(self, batch, epoch):
        self.model.train()
        state, loss = self.compute_loss(batch, epoch, mode='train')
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return state


    def validate(self, train_dataloader, valid_dataloader, epoch, only_loss=False):
        self.model.eval()

        with torch.no_grad():
            valid_state = State()
            for batch in valid_dataloader:
                state, loss = self.compute_loss(batch, epoch, mode='valid')
                valid_state.add(state)
        valid_state.mean()

        if not only_loss:
            torch.save(self.model.state_dict(), f'{self.args.save_path}/checkpoint_{epoch + 1}.pt')
            self.build_datastore(train_dataloader, epoch)

        return valid_state
    

    def save_datastore(self, train_dataloader, dstore_mmap):
        dstore_size = DATASTORE_SIZE[self.args.dataset]
        dstore_keys = np.memmap(dstore_mmap + '/keys.npy', dtype=np.float16, mode='w+', shape=(dstore_size, self.args.dimension))
        dstore_vals = np.memmap(dstore_mmap + '/vals.npy', dtype=np.int64, mode='w+', shape=(dstore_size, 1))

        # save datastore
        with torch.no_grad():
            source_keys = train_dataloader.dataset.source_keys
            target_keys = train_dataloader.dataset.target_keys
            token = train_dataloader.dataset.token_map
            nearest_distances = train_dataloader.dataset.ground_source_retrieve_dist[:, 1]
            for idx, (part_source_keys, part_target_keys, part_token, distance) in enumerate(zip(
                source_keys.split(self.args.dstore_max_tokens, 0),
                target_keys.split(self.args.dstore_max_tokens, 0),
                token.split(self.args.dstore_max_tokens, 0),
                nearest_distances.split(self.args.dstore_max_tokens, 0))):
                
                revised_keys, _ = self.model.key_forward(
                    source_hidden=part_source_keys.cuda(), 
                    target_hidden=part_target_keys.cuda(), 
                    token=part_token.cuda(),
                    nearest_distance=distance.cuda())
                
                start = idx * self.args.dstore_max_tokens
                dstore_keys[start: start + self.args.dstore_max_tokens] = revised_keys.cpu().numpy().astype(np.float16)
                dstore_vals[start: start + self.args.dstore_max_tokens] = part_token.unsqueeze(1).cpu().numpy().astype(np.int64)

        
    def train_datastore_on_gpu(self, dstore_mmap):
        dstore_size = DATASTORE_SIZE[self.args.dataset]
        dstore_keys = np.memmap(dstore_mmap + '/keys.npy', dtype=np.float16, mode='r', shape=(dstore_size, self.args.dimension))
        dstore_vals = np.memmap(dstore_mmap + '/vals.npy', dtype=np.int64, mode='r', shape=(dstore_size, 1))

        # initalize faiss index
        quantizer = faiss.IndexFlatL2(self.args.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.args.dimension,
                                self.args.ncentroids, self.args.code_size, 8)
        index.nprobe = self.args.probe
        res = faiss.StandardGpuResources()

        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)

        # train index
        random_sample = np.random.choice(np.arange(dstore_vals.shape[0]), size=[min(1000000, dstore_vals.shape[0])], replace=False)
        gpu_index.train(dstore_keys[random_sample].astype(np.float32))
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), dstore_mmap + "/knn_index.trained")

        # add keys
        index = faiss.read_index(dstore_mmap + "/knn_index.trained")
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)

        start = 0
        while start < dstore_size:
            end = min(dstore_size, start + self.args.num_keys_to_add_at_a_time)
            to_add_tensor = torch.from_numpy(dstore_keys[start:end].copy().astype(np.float32))
            gpu_index.add_with_ids(to_add_tensor, torch.arange(start, end))
            start += self.args.num_keys_to_add_at_a_time

            if (start % 1000000) == 0:
                faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), dstore_mmap + "/knn_index")
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), dstore_mmap + "/knn_index")


    def build_datastore(self, train_dataloader, epoch):
        logger.info('build datastore ...')
        start = time.time()
        dstore_mmap = f'{self.args.save_path}/checkpoint_{epoch + 1}_datastore'
        if not os.path.exists(dstore_mmap):
            os.mkdir(dstore_mmap)

        # build datastore
        self.save_datastore(train_dataloader, dstore_mmap)
        logger.info(f'save datastore, spend {time.time() - start:.4f} sec')
        start = time.time()

        self.train_datastore_on_gpu(dstore_mmap)

        if os.path.exists(dstore_mmap + '/keys.npy') and os.path.exists(dstore_mmap + '/knn_index'):
            os.remove(dstore_mmap + '/keys.npy')
        logger.info(f'train datastore, spend {time.time() - start:.4f} sec')
    

    def load_index(self, dstore_mmap):
        faiss_index = faiss.read_index(f'{dstore_mmap}/knn_index', faiss.IO_FLAG_ONDISK_SAME_DIR)
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index, co)
        self.faiss_index = faiss_index
        self.faiss_index.nprobe = self.args.probe


def main(args):
    setup_seed(args.seed)
    
    train_dataset = RevisedkeyDataset(args=args, mode='train', 
        source_dstore_mmap=args.source_dstore_mmap, 
        target_dstore_mmap=args.target_dstore_mmap)
    
    valid_dataset = RevisedkeyDataset(args=args, mode='valid', 
        source_dstore_mmap=args.source_dstore_mmap + '/valid', 
        target_dstore_mmap=args.target_dstore_mmap + '/valid',
        datastore_source_key=train_dataset.datastore_source_key,
        datastore_target_key=train_dataset.datastore_target_key,
        datastore_token_map=train_dataset.datastore_token_map)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.max_tokens,
        shuffle=True,
        collate_fn=train_dataset.collate)

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.max_tokens,
        shuffle=False,
        collate_fn=valid_dataset.collate,)

    model = Reviser(args).cuda()
    logger.info(model)

    trainer = Trainer(args, model, valid_dataset.vocab_size, train_dataset)
    trainer.train(train_dataloader, valid_dataloader)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    p = parser.add_argument_group('Data')
    p.add_argument('--dataset', '-dataset', type=str, required=True)
    p.add_argument('--dimension', '-dimension', type=int, default=1024)
    p.add_argument('--max-train-num', '-max-train-num', type=int, default=1000000)
    p.add_argument('--filte-data', '-filter-data', default=False, action='store_true')
    p.add_argument('--select-topk', '-select-topk', type=int, default=8)
    p.add_argument('--remove-top', '-remove-top', default=False, action='store_true')
    p.add_argument('--sample-num', '-sample-num', type=int, default=8)
    p.add_argument('--source-dstore-mmap', '-source-dstore-mmap', type=str, required=True)
    p.add_argument('--target-dstore-mmap', '-target-dstore-mmap', type=str, required=True)
    p.add_argument('--filte-unreliable-ratio', '-filte-unreliable-ratio', type=float, default=0.5)
    p.add_argument('--vocab-path', '-vocab-path', required=True, type=str)
    p.add_argument('--recompute-deno', '-recompute-deno', default=False, action='store_true')

    p = parser.add_argument_group('Model')
    p.add_argument('--dropout', '-dropout', type=float, default=0.2)
    p.add_argument('--ffn-size', '-ffn-size', type=int, default=8192)
    p.add_argument('--source-model', '-source-model', type=str, required=True)
    p.add_argument('--target-model', '-target-model', type=str, required=True)
    p.add_argument('--epsilon', '-epsilon', default=1.0, type=float)
    p.add_argument('--lambda', '-lambda', default=0.5, type=float)

    p = parser.add_argument_group('Train')
    p.add_argument('--seed', '-seed', type=int, default=42)
    p.add_argument('--use-cuda', '-use-cuda', type=bool, default=True)
    p.add_argument('--learning-rate', '-learning-rate', type=float, default=5e-4)
    p.add_argument('--max-tokens', '-max-tokens', type=int, default=10000)
    p.add_argument('--max-epoch', '-max-epoch', type=int, default=100)
    p.add_argument('--log-interval-update', '-log-interval-update', type=int, default=10)
    p.add_argument('--valid-interval-epoch', '-valid-interval-epoch', type=int, default=1)
    p.add_argument('--skip-build-datastore', '-skip-build-datastore', default=False, action='store_true')
    p.add_argument('--norm-weight', '-norm-weight', default=0.5, type=float)

    p = parser.add_argument_group('Datastore')
    p.add_argument('--save-path', '-save-path', type=str)
    p.add_argument('--ncentroids', '-ncentroids', type=int, default=4096)
    p.add_argument('--code-size', '-code-size', type=int, default=64)
    p.add_argument('--probe', '-probe', type=int, default=32)
    p.add_argument('--dstore-max-tokens', '-dstore-max-tokens', type=int, default=10000)
    p.add_argument('--num_keys_to_add_at_a_time', '-num_keys_to_add_at_a_time', type=int, default=50000)

    args = parser.parse_args()
    logger.info(args)
    main(args)