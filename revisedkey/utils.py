import os
import torch
import numpy as np
import random

from typing import Any
from dataclasses import dataclass

DATASTORE_SIZE = {
    'koran': 524400,
    'it': 3613350,
    'law': 19070000,
    'medical': 6903320
}

TEMPERATURE = {
    'koran': 100,
    'it': 10,
    'law': 10,
    'medical': 10
}

TOPK = {
    'koran': 8,
    'it': 8,
    'law': 4,
    'medical': 4
}

EFFECT_SIZE = {
    'koran': 524374,
    'it': 3602862,
    'law': 19061382,
    'medical': 6903141
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def memmap_to_tensor(mem_data):
    array_data = np.zeros(mem_data.shape, dtype=mem_data.dtype)
    array_data = mem_data[:]
    if mem_data.dtype == np.float16:
        array_data = array_data.astype('float32')
    if mem_data.dtype == np.int32:
        array_data = array_data.astype('int64')
    data = torch.from_numpy(array_data.copy())
    return data


def memmap_to_tensor_and_clip(mem_data):
    array_data = np.zeros(mem_data.shape, dtype=mem_data.dtype)
    array_data = mem_data[:]
    if mem_data.dtype == np.float16:
        array_data = array_data.astype('float32')
    if mem_data.dtype == np.int32:
        array_data = array_data.astype('int64')
    data = torch.from_numpy(array_data.copy())
    data = data[:, :32]
    return data


def compute_retrieve_probs(distance, token, vocab_size, temp, topk=None):
    scaled_distance = - distance / temp
    sparse_distribution = torch.softmax(scaled_distance, dim=-1)
    zeros = torch.zeros(size=(sparse_distribution.size(0), vocab_size), dtype=sparse_distribution.dtype)
    distribution = torch.scatter_add(zeros, -1, token, sparse_distribution)

    if topk is None:
        return distribution
    else:
        topk_distribution = distribution.topk(topk, dim=-1)
        return topk_distribution.values.squeeze(1), topk_distribution.indices.squeeze(1)


def compute_target_retrieve_probs(distance, token, target_token, temp):
    scaled_distance = - distance / temp
    sparse_distribution = torch.softmax(scaled_distance, dim=-1)
    target_mask = (target_token.unsqueeze(1) == token)
    probs = (target_mask * sparse_distribution).sum(dim=-1)
    return probs


def difference(a, b):
    assert a.size(0) == b.size(0)
    batch_size = a.size(0)
    a_expand = a.unsqueeze(-1).expand(batch_size, a.size(1), b.size(1))
    b_expand = b.unsqueeze(1).expand(batch_size, a.size(1), b.size(1))
    mask = (a_expand == b_expand).sum(dim=-1) == 0
    return mask


def load_pretrain_embeddings(model_path):
    if os.path.exists(model_path + '.emb'):
        embeddings = torch.load(model_path + '.emb')
    else:
        ckpt = torch.load(model_path)
        embeddings = ckpt['model']['encoder.embed_tokens.weight']
        torch.save(embeddings, model_path + '.emb')
    return embeddings

@dataclass
class State:
    loss: Any = 0
    mse_loss: Any = 0
    norm_loss: Any = 0
    count: Any = 0        
    
    def merge(self, state):
        if self.loss == 0:
            loss_func = lambda x, y: y
            accu_func = loss_func
        elif state.loss == 0:
            loss_func = lambda x, y: x
            accu_func = loss_func
        else:
            loss_func = lambda x, y: min(x, y)
            accu_func = lambda x, y: max(x, y)

        self.loss = loss_func(self.loss, state.loss)
        self.mse_loss = loss_func(self.mse_loss, state.mse_loss)
        self.norm_loss = loss_func(self.norm_loss, state.norm_loss)
        self.count = 1

    def add(self, state):
        self.loss += state.loss
        self.mse_loss += state.mse_loss
        self.norm_loss += state.norm_loss
        self.count += state.count
    
    def clear(self):
        self.loss = 0
        self.mse_loss = 0
        self.norm_loss = 0
        self.count = 0
    
    def mean(self):
        if self.count != 0:
            self.loss = self.loss / self.count
            self.mse_loss = self.mse_loss / self.count
            self.norm_loss = self.norm_loss / self.count
        self.count = 1
    
    def info(self, epoch=None, step=None, valid=False):
        assert epoch is not None or step is not None
        mode = 'valid' if valid else 'train'
        prefix_str = f'[{mode}][epoch={epoch}]' if epoch is not None else f'[{mode}][step={step}]'
        info_str = f'{prefix_str}: loss={self.loss:.4f}  mse_loss={self.mse_loss:.4f}'
        if self.norm_loss != 0: info_str += f' norm_loss={self.norm_loss:.4f}'
        return info_str