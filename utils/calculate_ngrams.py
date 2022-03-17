from tqdm import tqdm
from multiprocessing import Pool
import collections
from nltk import ngrams
from sklearn.model_selection import train_test_split
import pickle
import math
import os


def batch(iterable1, n=1):
    l = len(iterable1)
    for ndx in range(0, l, n):
        yield iterable1[ndx:min(ndx + n, l)]



def get_train_ids(data_len, data_file):
    file_path = f'./final_data/data_split_ids/{data_file}.pkl'
    if not os.path.isfile(file_path):
        print("Generating new train/val/test ids")
        train_ids, test_ids = train_test_split(range(data_len), test_size=0.3, random_state=5)
        train_ids, val_ids = train_test_split(train_ids, test_size=1/7, random_state=22)

        # create folder if not exist
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        print(f"Storing train/val/test ids in {directory}")
        with open(file_path, 'wb') as f:
            pickle.dump((train_ids, val_ids, test_ids), f)
        return train_ids
    else:
        print(f"Reading train/val/test ids from {file_path}")        
        with open(file_path, 'rb') as f: 
            (train_ids, val_ids, test_ids) = pickle.load(f)
        return train_ids


def get_ngrams(n_cores = 30, n = 15):
    # All the workload names P01, P02 etc
    data_files = [f'P{p}' if p > 9 else f'P0{p}' for p in range(1, 32)]
    cmds_path = './intermediate_data/cmds/'
    ngram_path = f'./intermediate_data/ngrams/{n}-grams/'
    rows_per_file = 10*10000
    # create folder if not exist
    os.makedirs(ngram_path, exist_ok=True)
    

    def search_ngrams_total(data, n_cores=1):
        global search_ngrams
        # create helper finctions
        def search_ngrams_partial(data):
            my_ngrams = ngrams(list(data), n)
            my_ngrams_freq = collections.Counter(my_ngrams)
            return my_ngrams_freq

        def search_ngrams(data):
            total_freq = None
            for d in data:
                if total_freq == None:
                    total_freq = search_ngrams_partial(d)
                else:
                    total_freq += search_ngrams_partial(d)
            return total_freq
        chunks_per_worker = math.ceil(len(data)/n_cores)
        data = list(batch(data, chunks_per_worker))
        pool = Pool(n_cores)
        total_freqs = pool.map(search_ngrams, data) 
        pool.close()
        pool.join()

        total_freq = total_freqs[0]
        if n_cores > 1:
            for i in range(1, len(total_freqs)):
                total_freq += total_freqs[i]
        return total_freq
    
    # get ngrams
    for data_file in data_files:
        print('Working on', data_file)
        with open(cmds_path + data_file, 'r') as f:
            data = f.readline()
            file_len = len(data)

            total_chunks = file_len//rows_per_file
            # trim data
            data = data[:total_chunks*rows_per_file]
            data = list(batch(data, rows_per_file))

            train_ids = get_train_ids(len(data), data_file)
            
            data = [data[id] for id in train_ids]
            freqs = search_ngrams_total(data, n_cores=n_cores)
            
            with open(f'{ngram_path}{n}-gram.{data_file}.txt', 'w') as f:
                for item in freqs.most_common(200):
                    f.write(str(item) + '\n')
   

if __name__ == "__main__":
    for n in [7, 11, 15]:
        get_ngrams(n=n)
    
 