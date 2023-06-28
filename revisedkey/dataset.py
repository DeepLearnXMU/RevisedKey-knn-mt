import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Any
from dataclasses import dataclass
from collections import Counter, defaultdict
from fairseq.data.dictionary import Dictionary

from torch.utils.data import Dataset, DataLoader
from revisedkey.utils import (
    load_pretrain_embeddings,
    compute_target_retrieve_probs,
    memmap_to_tensor,
    memmap_to_tensor_and_clip,
    DATASTORE_SIZE,
    TEMPERATURE,
    EFFECT_SIZE,
    TOPK
)

import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.dataset")


def load_datastore(dstore_path, dstore_size, dimension, mode, double=False):
    if os.path.exists(dstore_path + '/retrieve_vals.pt') and mode != 'valid':
        keys = torch.load(dstore_path + '/keys.pt')
        vals = torch.load(dstore_path + '/vals.pt')
        retrieve_dist = torch.load(dstore_path + '/retrieve_dist.pt')
        retrieve_vals = torch.load(dstore_path + '/retrieve_vals.pt')
    
    else:
        if mode == 'valid':
            keys = torch.from_numpy(np.load(dstore_path + '/keys.npy').astype(np.float32))
            vals = torch.from_numpy(np.load(dstore_path + '/vals.npy')).squeeze(1)
            retrieve_dist = torch.from_numpy(np.load(dstore_path + '/retrieve_dist.npy', allow_pickle=True))[:, :32]
            retrieve_vals = torch.from_numpy(np.load(dstore_path + '/retrieve_vals.npy', allow_pickle=True))[:, :32]

        else:
            keys_from_mem = np.memmap(dstore_path + '/keys.npy', dtype=np.float16, mode='r', shape=(dstore_size, dimension))
            keys = memmap_to_tensor(keys_from_mem)
            del keys_from_mem

            vals_from_mem = np.memmap(dstore_path + '/vals.npy', dtype=np.int64, mode='r', shape=(dstore_size, 1))
            vals = memmap_to_tensor(vals_from_mem).squeeze(1)
            del vals_from_mem

            if double:
                dstore_size = dstore_size // 2

            if os.path.exists(dstore_path + f'/retrieve_results_start0_dist_size{dstore_size}_k32_int64.npy'):
                retrieve_dist_from_mem = np.memmap(dstore_path + f'/retrieve_results_start0_dist_size{dstore_size}_k32_int64.npy', dtype=np.float32, mode='r', shape=(dstore_size, 32))
            elif os.path.exists(dstore_path + f'/retrieve_results_start0_dist_size{dstore_size}_k64_int64.npy'):
                retrieve_dist_from_mem = np.memmap(dstore_path + f'/retrieve_results_start0_dist_size{dstore_size}_k64_int64.npy', dtype=np.float32, mode='r', shape=(dstore_size, 64))
            else:
                retrieve_dist_from_mem = np.memmap(dstore_path + f'/retrieve_results_start0_dist_size{dstore_size}_k32_int32.npy', dtype=np.float32, mode='r', shape=(dstore_size, 32))

            retrieve_dist = memmap_to_tensor_and_clip(retrieve_dist_from_mem)
            del retrieve_dist_from_mem

            if os.path.exists(dstore_path + f'/retrieve_results_start0_dist_size{dstore_size}_k32_int64.npy'):
                retrieve_vals_from_mem = np.memmap(dstore_path + f'/retrieve_results_start0_vals_size{dstore_size}_k32_int64.npy', dtype=np.int64, mode='r', shape=(dstore_size, 32))
            elif os.path.exists(dstore_path + f'/retrieve_results_start0_dist_size{dstore_size}_k64_int64.npy'):
                retrieve_vals_from_mem = np.memmap(dstore_path + f'/retrieve_results_start0_vals_size{dstore_size}_k64_int64.npy', dtype=np.int64, mode='r', shape=(dstore_size, 64))
            else:
                retrieve_vals_from_mem = np.memmap(dstore_path + f'/retrieve_results_start0_vals_size{dstore_size}_k32_int32.npy', dtype=np.int32, mode='r', shape=(dstore_size, 32))
            retrieve_vals = memmap_to_tensor_and_clip(retrieve_vals_from_mem)
            del retrieve_vals_from_mem

        torch.save(keys, dstore_path + '/keys.pt')
        torch.save(vals, dstore_path + '/vals.pt')
        torch.save(retrieve_dist, dstore_path + '/retrieve_dist.pt')
        torch.save(retrieve_vals, dstore_path + '/retrieve_vals.pt')
    return keys, vals, retrieve_dist, retrieve_vals


@dataclass
class Batch:
    index: Any = None
    sample_index: Any = None
    deno_detail: Any = None
    query_hidden: Any = None
    key_token: Any = None
    key_source_hidden: Any = None
    key_target_hidden: Any = None
    nearest_distance: Any = None
    

class RevisedkeyDataset(Dataset):
    def __init__(self, args, mode, source_dstore_mmap, target_dstore_mmap, 
                 datastore_source_key=None, datastore_target_key=None, datastore_token_map=None):
        self.args = args
        self.mode = mode
        self.source_dstore_mmap = source_dstore_mmap
        self.target_dstore_mmap = target_dstore_mmap

        dstore_size = DATASTORE_SIZE[args.dataset]

        logger.info(f'{self.mode}: load target-domain datastore data from {target_dstore_mmap}')
        target_keys, _, target_retrieve_dist, target_retrieve_vals = load_datastore(
            target_dstore_mmap, dstore_size, args.dimension, mode)
        
        self.target_keys = target_keys
        self.target_retrieve_dist = target_retrieve_dist
        self.target_retrieve_vals = target_retrieve_vals

        logger.info(f'{self.mode}: load source-domain datastore data from {source_dstore_mmap}')
        source_keys, token_map, source_retrieve_dist, source_retrieve_vals = load_datastore(
            source_dstore_mmap, dstore_size, args.dimension, mode)

        self.token_map = token_map
        self.source_keys = source_keys
        self.source_retrieve_dist = source_retrieve_dist
        self.source_retrieve_vals = source_retrieve_vals

        if self.mode == 'train':
            self.datastore_source_key = source_keys
            self.datastore_target_key = target_keys
            self.datastore_token_map = token_map

            if self.args.remove_top:
                logger.info(f'{self.mode}: remove top example for each query')
        else:
            assert datastore_source_key != None
            assert datastore_target_key != None
            assert datastore_token_map != None

            self.datastore_source_key = datastore_source_key
            self.datastore_target_key = datastore_target_key
            self.datastore_token_map = datastore_token_map

        self.topk = TOPK[self.args.dataset]
        self.temperature = TEMPERATURE[self.args.dataset]
        self.vocab = Dictionary.load(self.args.vocab_path)
        self.embed = load_pretrain_embeddings(self.args.target_model)

        if self.args.sample_num > self.topk:
            logger.info(f'{self.mode}: reset sample_num = {self.topk}')
            self.args.sample_num = self.topk

        self.vocab_size = len(self.vocab)
        self.compute_distance()
        self.construct_data()


    def compute_norm_hub(self):
        if self.mode != 'train':
            hubs = torch.ones_like(self.deno).float()
            return hubs

        logger.info(f'{self.mode}: compute norm-hub')
        start, end = 0, self.args.select_topk
        if self.mode == 'train' and self.args.remove_top:
            start += 1
            end += 1

        retrieve_topk = self.target_retrieve_vals[:, start:end].contiguous()
        occurence_cnt = Counter(retrieve_topk.view(-1).tolist())

        token_occurence_cnt = defaultdict(lambda: 0)
        for index, value in occurence_cnt.items():
            token_occurence_cnt[self.token_map[index].item()] += value

        norm_occurence_cnt = {}
        for index in occurence_cnt:
            token = self.token_map[index].item()
            norm_occurence_cnt[index] = occurence_cnt[index] / token_occurence_cnt[token]

        index = torch.LongTensor(list(norm_occurence_cnt.keys()))
        value = torch.FloatTensor(list(norm_occurence_cnt.values()))
        norm_hubs = torch.zeros([self.source_keys.size(0)]).float().scatter_(0, index, value)

        return norm_hubs

    
    def construct_data(self):
        logger.info(f'{self.mode}: construct data')

        if self.mode == 'train':
            end_index = EFFECT_SIZE[self.args.dataset]
            logger.info(f'{self.mode}: datastore shape = {self.target_keys.size(0)}')
            logger.info(f'{self.mode}: datastore end_index = {end_index}')
        else:
            end_index = self.target_keys.size(0)

        total_indices = torch.arange(end_index)
        self.end_index = end_index
        self.target_retrieve_vals = self.target_retrieve_vals

        # compute retrieve weight distance on target domain as deno
        if os.path.exists(self.target_dstore_mmap + '/deno.pt') and not self.args.recompute_deno:
            logger.info(f'{self.mode}: load deno & deno_detail')
            self.deno = torch.load(self.target_dstore_mmap + f'/deno.pt')
            self.deno_detail = torch.load(self.target_dstore_mmap + f'/deno_detail.pt')
        else:
            logger.info(f'{self.mode}: compute retrieve deno')
            deno, deno_detail = self.compute_retrieve_deno()
            torch.save(deno, self.target_dstore_mmap + f'/deno.pt')
            torch.save(deno_detail, self.target_dstore_mmap + f'/deno_detail.pt')
            self.deno = deno
            self.deno_detail = deno_detail

        self.norm_hubs = self.compute_norm_hub()

        if self.mode == 'train' and self.args.filte_data:
            start, end = 0, self.args.sample_num
            if self.args.remove_top:
                end += 1
                start += 1
            index = total_indices

            # data filte: remove unreliable examples
            filte_threshold = np.percentile(self.norm_hubs[index].numpy(), self.args.filte_unreliable_ratio * 100)
            filte_index = torch.where(self.norm_hubs[index] < filte_threshold)[0]
            keep_index = torch.where(self.norm_hubs[index] >= filte_threshold)[0]
            logger.info(f'{self.mode}: filter ratio={self.args.filte_unreliable_ratio}')
            logger.info(f'{self.mode}: filter threshold={filte_threshold:.6f}')

            self.examples = index[keep_index]
            self.filte_examples = index[filte_index]
            
            cover_rate = len(set(self.target_retrieve_vals[self.examples, start:end].view(-1).tolist())) / end_index
            logger.info(f'{self.mode}: select {len(self.examples) / end_index * 100:.2f}% data')
            logger.info(f'{self.mode}: cover {cover_rate * 100:.2f}% data')
            self.total_indices = total_indices

        else:
            if self.mode == 'train' and len(total_indices) > self.args.max_train_num:
                examples = total_indices.tolist()
                random.shuffle(examples)
                self.examples = torch.LongTensor(sorted(examples[:self.args.max_train_num]))
            else:
                self.examples = total_indices
            self.total_indices = total_indices
        
        logger.info(f'{self.mode}: size = {len(self.examples)}')


    def __getitem__(self, index):
        example_index = self.examples[index]
        query_hidden = self.source_keys[example_index]

        sample_index = torch.LongTensor(range(self.args.sample_num))
        if self.args.remove_top:
            sample_index += 1

        deno_detail = self.deno_detail[example_index, sample_index]
        key_indices = self.target_retrieve_vals[example_index, sample_index]
        key_source_hidden = self.datastore_source_key[key_indices]
        key_target_hidden = self.datastore_target_key[key_indices]
        key_token = self.datastore_token_map[key_indices]
        nearest_distance = self.ground_source_retrieve_dist[example_index, 1]

        instance = {
            'index': example_index,
            'sample_index': sample_index,
            'deno_detail': deno_detail,
            'query_hidden': query_hidden,
            'key_token': key_token,
            'key_source_hidden': key_source_hidden,
            'key_target_hidden': key_target_hidden,
            'nearest_distance': nearest_distance,
        }
        return instance

    def collate(self, batch):
        index = torch.LongTensor([instance['index'] for instance in batch])
        sample_index = torch.stack([instance['sample_index'] for instance in batch], dim=0)
        deno_detail = torch.stack([instance['deno_detail'] for instance in batch], dim=0)
        query_hidden = torch.stack([instance['query_hidden'] for instance in batch], dim=0)
        
        key_token = torch.stack([instance['key_token'] for instance in batch], dim=0)
        key_source_hidden = torch.stack([instance['key_source_hidden'] for instance in batch], dim=0)
        key_target_hidden = torch.stack([instance['key_target_hidden'] for instance in batch], dim=0)
        nearest_distance = torch.stack([instance['nearest_distance'] for instance in batch], dim=0)

        if self.args.use_cuda:
            query_hidden = query_hidden.to('cuda')
            deno_detail = deno_detail.to('cuda')
            key_token = key_token.to('cuda')
            key_source_hidden = key_source_hidden.to('cuda')
            key_target_hidden = key_target_hidden.to('cuda')
            nearest_distance = nearest_distance.to('cuda')
            
        batch = Batch()
        batch.index = index
        batch.sample_index = sample_index
        batch.deno_detail = deno_detail
        batch.query_hidden = query_hidden

        batch.key_token = key_token
        batch.key_source_hidden = key_source_hidden
        batch.key_target_hidden = key_target_hidden
        batch.nearest_distance = nearest_distance

        return batch


    def __len__(self):
        return len(self.examples)


    def compute_distance(self):
        if os.path.exists(self.source_dstore_mmap + '/ground_retrieve_dist.pt') \
           and os.path.exists(self.target_dstore_mmap + '/ground_retrieve_dist.pt'):
            ground_source_retrieve_dist = torch.load(self.source_dstore_mmap + '/ground_retrieve_dist.pt')
            ground_target_retrieve_dist = torch.load(self.target_dstore_mmap + '/ground_retrieve_dist.pt')
            
        else:
            criterion = nn.MSELoss(reduction='none')
            source_retrieve_dist = []
            for query, key_index in zip(
                self.source_keys.split(self.args.max_tokens, 0),
                self.source_retrieve_vals.split(self.args.max_tokens, 0)):

                keys = self.datastore_source_key[key_index]
                query = query.unsqueeze(1).expand_as(keys)
                distance = criterion(query, keys).sum(dim=-1)
                source_retrieve_dist.append(distance)
            
            target_retrieve_dist = []
            for query, key_index in zip(
                self.target_keys.split(self.args.max_tokens, 0),
                self.target_retrieve_vals.split(self.args.max_tokens, 0)):

                keys = self.datastore_target_key[key_index]
                query = query.unsqueeze(1).expand_as(keys)
                distance = criterion(query, keys).sum(dim=-1)
                target_retrieve_dist.append(distance)
            
            ground_source_retrieve_dist = torch.cat(source_retrieve_dist, dim=0)
            ground_target_retrieve_dist = torch.cat(target_retrieve_dist, dim=0)

            torch.save(ground_source_retrieve_dist, self.source_dstore_mmap + '/ground_retrieve_dist.pt')
            torch.save(ground_target_retrieve_dist, self.target_dstore_mmap + '/ground_retrieve_dist.pt')

        self.ground_source_retrieve_dist = ground_source_retrieve_dist
        self.ground_target_retrieve_dist = ground_target_retrieve_dist


    def compute_retrieve_deno(self):
        # remove top1 for train
        start, end = 0, self.args.select_topk
        if self.mode == 'train' and self.args.remove_top:
            start += 1
            end += 1

        criterion = nn.MSELoss(reduction='none')
        def compute_deno(retrieve_query, retrieve_indices, datastore_keys):
            deno = []
            deno_detail = []
            for query, key_index in zip(
                retrieve_query.split(self.args.max_tokens, 0),
                retrieve_indices.split(self.args.max_tokens, 0)):

                keys = datastore_keys[key_index]
                query = query.unsqueeze(1).expand_as(keys)
                distance = criterion(query, keys).sum(dim=-1)
                    
                # add 1 to avoid Nan
                weight = torch.softmax(- distance / self.temperature, dim=-1)
                weight_distance = (weight * distance).sum(dim=-1).unsqueeze(1) + 1
                deno_detail.append(distance)
                deno.append(weight_distance)
            
            deno_detail = torch.cat(deno_detail, dim=0)
            deno = torch.cat(deno, dim=0)
            return deno, deno_detail
            
        target_deno, target_deno_detail = compute_deno(
            self.target_keys, self.target_retrieve_vals[:, :end], self.datastore_target_key)

        if self.args.remove_top and self.mode == 'train':
            target_deno_detail = target_deno_detail[:, 0].unsqueeze(1).expand_as(target_deno_detail)
        else:
            target_deno_detail_ = target_deno_detail[:, 1].unsqueeze(1).expand_as(target_deno_detail)
            target_deno_detail[:, 1:] = target_deno_detail_[:, 1:]

        return target_deno, target_deno_detail
