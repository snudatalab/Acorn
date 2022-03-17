from tqdm import tqdm
from multiprocessing import Pool
import math
import pandas as pd
import os

def batch(iterable1, n=1):
    l = len(iterable1)
    for ndx in range(0, l, n):
        yield iterable1[ndx:min(ndx + n, l)]

def workload_to_cmd(file_name):
    # output directory path
    cmds_path = './intermediate_data/cmds/'
    # path where original workloads have located
    workload_path = './raw_data/' 
    # clear old file content
    os.makedirs(cmds_path, exist_ok=True)
    cmds_out = open(cmds_path + file_name, 'w')
    cmds_out.close()

    for chunk in pd.read_csv(workload_path + file_name, chunksize=100000, header=None,  dtype = str):
        cmds         = "".join(chunk[0])
        with open(cmds_path + file_name, 'a') as f:
            f.write(cmds)


def get_cmd(n_cores = 4):
    # All the workload names P01.lc, P02.lc etc
    data_files = [f'P{p}' if p > 9 else f'P0{p}' for p in range(1, 35)]
    for files in tqdm(batch(data_files, n_cores), total=math.ceil(len(data_files)/n_cores)):
        pool = Pool(len(files))
        pool.map(workload_to_cmd, files) 
        pool.close()
        pool.join()

if __name__ == "__main__":
    # number of files to be processed in a time
    n_cores = 4
    get_cmd(n_cores = 4)
 
