import argparse
import numpy as np
import faiss
import ctypes
import time
import pickle
import logging
import os
import sys

from collections import defaultdict

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dstore-mmap', type=str,
    help='memmap where keys and vals are stored')
parser.add_argument('--dstore-size', type=int,
    help='the dstore vectors')
parser.add_argument('--actual-dstore-size', type=int,
    default=None, help='the dstore vectors')
parser.add_argument('--index', type=str,
    help='the faiss index file')
parser.add_argument('--dstore-fp16', default=False,
    action='store_true')
parser.add_argument('--nprobe', type=int, default=32)
parser.add_argument('--dimension', type=int, default=1024)
parser.add_argument('--k', type=int, default=1024,
    help='the number of nearest neighbors')
parser.add_argument('--save', type=str,
    help='the number of nearest neighbors')

# for the purpose of parallel computation
parser.add_argument('--start-point', type=int, default=0,
    help='the starting point to traverse the datastore')
parser.add_argument('--num', type=int, default=1e11,
    help='number of points to traverse')

args = parser.parse_args()

if args.actual_dstore_size is None:
    args.actual_dstore_size = args.dstore_size

logger.info(args)
logger.info(f'shape ({args.dstore_size}, {args.dimension})')

if args.dstore_fp16:
    keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
else:
    keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))

save_size = min(args.num, args.actual_dstore_size - args.start_point)
retrieve_vals = np.memmap(args.save + f'_vals_size{save_size}_k{args.k}_int64.npy', dtype=np.int64, mode='w+', shape=(save_size, args.k))
retrieve_dist = np.memmap(args.save + f'_dist_size{save_size}_k{args.k}_int64.npy', dtype=np.float32, mode='w+', shape=(save_size, args.k))

index = faiss.read_index(args.index, faiss.IO_FLAG_ONDISK_SAME_DIR)

# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

bsz = 3072
batches = []
cnt = 0
offset = 0

for id_, i in enumerate(range(args.start_point, min(args.start_point + args.num, args.actual_dstore_size))):
    if i % 10000 == 0:
        logger.info(f'processing {i}th entries')
    batches.append(keys[i])
    cnt += 1

    if cnt % bsz == 0:
        dists, knns = index.search(np.array(batches).astype(np.float32), args.k)
        assert knns.shape[0] == bsz

        retrieve_vals[offset:offset + knns.shape[0]] = knns
        retrieve_dist[offset:offset + knns.shape[0]] = dists

        cnt = 0
        batches = []

        offset += knns.shape[0]

if len(batches) > 0:
    dists, knns = index.search(np.array(batches).astype(np.float32), args.k)
    retrieve_vals[offset:offset + knns.shape[0]] = knns
    retrieve_dist[offset:offset + knns.shape[0]] = dists