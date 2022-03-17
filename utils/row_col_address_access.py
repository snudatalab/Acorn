from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import pickle
import math
import pandas as pd
import numpy as np
import os


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def count_address_access(cmds, ranks, bankgroups, banks, addresses, step=1024*16):
    # create address data type and initialize to 0
    address = {}
    for r in range(2):
        address[str(r)] = {}
        for bg in range(4):
            address[str(r)][str(bg)] = {}
            for b in range(4):
                address[str(r)][str(bg)][str(b)] = {}
                # track active row
                address[str(r)][str(bg)][str(b)]['active_row'] = -1

    x = [0 for i in range(1024)]
    # count appearances
    for c, r, bg, b, a in zip(cmds, ranks, bankgroups, banks, addresses):
        if c == '1':
            active_row = a//step
            # save compressed format active row
            address[r][bg][b]['active_row'] = active_row
        elif c == '3' or c == '5' and address[r][bg][b]['active_row'] > -1:
            col = a//8
            row = address[r][bg][b]['active_row']
            # increment count for active row and col
            x[(row*128)+col] += 1

    return x

def thread_process(data):
    rows_per_file = 10*10000
    x = []
    for cmds, ranks, bankgroups, banks, addresses in zip(batch(data[0], rows_per_file), batch(data[1], rows_per_file), batch(data[2], rows_per_file), batch(data[3], rows_per_file), batch(data[4], rows_per_file)):
        x.append(count_address_access(cmds, ranks, bankgroups, banks, addresses))

    return x

def get_address_access():
    store_folder = "./final_data/row_col_address_access_counts/"
    workload_path = './raw_data/'

    rows_per_file = 10*10000
    n_cores = mp.cpu_count()
    # All the workload names P01.lc, P02.lc etc
    data_files = [f'P{p}' if p > 9 else f'P0{p}' for p in range(1, 35)]
    # create folder if not exist
    os.makedirs(store_folder, exist_ok=True)
    
    for file_name in data_files:
        print('Starting', file_name)
        X_total = []
        cnt = 0
        for chunk in pd.read_csv(workload_path + file_name, chunksize=5*n_cores*rows_per_file, header=None, dtype = str):
            cnt += 1
            print(f'Processing {cnt}th chunk...')
            chunk = chunk.dropna()
            cmds        = chunk[0]
            ranks       = chunk[1]
            bankgroups  = chunk[2]
            banks       = chunk[3]
            addresses   = chunk[4].astype(int)
      
            file_len = len(ranks)

            total_chunks = file_len//rows_per_file
            # trim data
            cmds = cmds[:total_chunks*rows_per_file]
            ranks = ranks[:total_chunks*rows_per_file]
            bankgroups = bankgroups[:total_chunks*rows_per_file]
            banks = banks[:total_chunks*rows_per_file]
            addresses = addresses[:total_chunks*rows_per_file]

            # get number of chunks each worker should do
            chunks_per_worker = math.ceil(total_chunks/n_cores)
            # split data
            cmds = list(batch(cmds, chunks_per_worker*rows_per_file))
            ranks = list(batch(ranks, chunks_per_worker*rows_per_file))
            bankgroups = list(batch(bankgroups, chunks_per_worker*rows_per_file))
            banks = list(batch(banks, chunks_per_worker*rows_per_file))
            addresses = list(batch(addresses, chunks_per_worker*rows_per_file))

            # combine data
            chunks = list(zip(cmds, ranks, bankgroups, banks, addresses))
            cmds = []
            ranks = []
            bankgroups = []
            banks = []
            addresses = []
            # create pool
            pool = Pool(n_cores)
            chunck_results = pool.map(thread_process, chunks)
            pool.close()
            pool.join()
            
            X = [row for chunck in chunck_results for row in chunck]
            for x in X:
                X_total.append(x)
        # Saving the objects:
        with open(f'{store_folder}{file_name}.npy', 'wb') as f: 
            pickle.dump(np.array(X_total), f)

if __name__ == "__main__":
    get_address_access()
