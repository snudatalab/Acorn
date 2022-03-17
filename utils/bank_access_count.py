from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import pickle
import math
import pandas as pd
import os



def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def count_address_access(ranks, bankgroups, banks):
    # create address data type and initialize to 0
    address = {}
    for r in range(2):
        address[str(r)] = {}
        for bg in range(4):
            address[str(r)][str(bg)] = {}
            for b in range(4):
                address[str(r)][str(bg)][str(b)] = 0
    # count appearances
    for r, bg, b in zip(ranks, bankgroups, banks):
        address[r][bg][b] += 1
    # turn into vector of len 32
    x = [address[r][bg][b] for r in address for bg in address[r] for b in address[r][bg]]
    return x

def thread_process(data):
    # subsequence length
    rows_per_file = 10*10000
    x = []
    for ranks, bankgroups, banks in zip(batch(data[0], rows_per_file), batch(data[1], rows_per_file), batch(data[2], rows_per_file)):
        x.append(count_address_access(ranks, bankgroups, banks))
    return x

def count_bank_access():
    workload_path = "./raw_data/"
    # pickle file store path
    bank_access_count_path = './final_data/bank_access_counts/'
    # create folder if not exist
    os.makedirs(bank_access_count_path, exist_ok=True)
    
    # subsequence length
    rows_per_file = 10*10000

    # get available cpu thread count
    n_cores = mp.cpu_count()
    # All the workload names P01.lc, P02.lc etc
    data_files = [f'P{p}' if p > 9 else f'P0{p}' for p in range(1, 35)]

    for file_name in data_files:
        print('Starting', file_name)
        X_total = []
        for chunk in pd.read_csv(workload_path + file_name, chunksize=5*n_cores*rows_per_file, header=None, dtype = str):
            ranks       = chunk[1]
            bankgroups  = chunk[2]
            banks       = chunk[3]

            file_len = len(ranks)

            total_chunks = file_len//rows_per_file
    
            # trim data
            ranks = ranks[:total_chunks*rows_per_file]
            bankgroups = bankgroups[:total_chunks*rows_per_file]
            banks = banks[:total_chunks*rows_per_file]
    
            # get number of chunks each worker should do
            chunks_per_worker = math.ceil(total_chunks/n_cores)
            # if last chunk
            if chunks_per_worker == 0: continue
            # # split data
            ranks = list(batch(ranks, chunks_per_worker*rows_per_file))
            bankgroups = list(batch(bankgroups, chunks_per_worker*rows_per_file))
            banks = list(batch(banks, chunks_per_worker*rows_per_file))
            chunks = list(zip(ranks, bankgroups, banks))

            # create pool
            pool = Pool(len(chunks))
            
            chunck_results = pool.map(thread_process, chunks)
            pool.close()
            pool.join()
            
            X = [row for chunck in chunck_results for row in chunck]
            for x in X:
                X_total.append(x)
        # Saving the objects:
        with open(f'{bank_access_count_path}{file_name}.pkl', 'wb') as f: 
            pickle.dump([X_total], f)

if __name__ == "__main__":  
    count_bank_access()
    