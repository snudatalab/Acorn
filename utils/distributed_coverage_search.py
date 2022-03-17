from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import pickle
import math
import os


def get_ngrams(data_files, ngram_path, topn=25, n=7, test_set_len=31):
    combined_sequences = set()
    for data_file in data_files[:test_set_len]:
        topn_from_file = []
        with open(f'{ngram_path}{n}-gram.{data_file}.txt') as f:
            for i in range(topn):
                topn_from_file.append(f.readline())
         

        # get cmd strings
        topn_from_file = list(map(lambda x: eval(x), topn_from_file))
        p01_topn_strs = set(map(lambda x: ''.join(x[0]), topn_from_file))
        combined_sequences = combined_sequences.union(p01_topn_strs)

    return {e:i for i, e in enumerate(combined_sequences)}

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_coverage():
    # All the workload names P01.lc, P02.lc etc
    data_files = [f'P{p}' if p > 9 else f'P0{p}' for p in range(1, 35)]
    topn = 25
    n_cores = mp.cpu_count()
    rows_per_file = 10*10000
    cmds_path = './intermediate_data/cmds/'
    for ngrams in [7, 11, 15]:
        sequence_path = f'./intermediate_data/sequences/{ngrams}-grams.top{topn}_osr.pkl'
        ngram_path = f'./intermediate_data/ngrams/{ngrams}-grams/'
        output_path = f"./final_data/{ngrams}-grams/"
        # create output directory in not exists
        os.makedirs(output_path, exist_ok=True)

        if not os.path.isfile(sequence_path):
            sequences = get_ngrams(data_files, ngram_path, n=ngrams)
            print(f"Generated {ngrams}-gram sequence with top-{topn} {ngrams}-grams")
            print(f"{ngrams}-gram length with top-{topn} is {len(sequences)}")
            # create dir if not exists
            directory = os.path.dirname(sequence_path)
            os.makedirs(directory, exist_ok=True)

            with open(sequence_path, 'wb') as f:
                pickle.dump(sequences, f)
        else:
            print(f'Reading generated {ngrams}-gram sequence with top-{topn} {ngrams}-grams')         
            with open(sequence_path, 'rb') as f: 
                sequences = pickle.load(f)

        # create worker process
        def get_covarage(data):
            ngram_n = ngrams
            counter = [0 for i in range(len(sequences))]
            start = 0
            for end in range(ngram_n, len(data)):
                s = data[start:end]
                if s in sequences:
                    counter[sequences[s]] += 1
                start += 1

            return counter
        global thread_process
        # create thread process
        def thread_process(data):
            x = []
            for subsequence in batch(data, rows_per_file):
                x.append(get_covarage(subsequence))
            return x

        print(f'Starting with calculating ngrams with {n_cores} processes')
        for file_name in data_files:
            print(f'Starting {ngrams}-gram for {file_name}')
            with open(cmds_path + file_name, 'r') as f:
                data = f.readline()
                file_len = len(data)

                total_chunks = file_len//rows_per_file
                # trim data
                data = data[:total_chunks*rows_per_file]
                # get number of chunks each worker should do
                chunks_per_worker = math.ceil(total_chunks/n_cores)
                # split data
                chunks = list(batch(data, chunks_per_worker*rows_per_file))
                # release data
                data = []
                # create pool
                pool = Pool(n_cores)
                
                chunck_results = pool.map(thread_process, chunks)
                pool.close()
                pool.join()
            
            X = [row for chunck in chunck_results for row in chunck]
            # Saving the objects:
            with open(f'{output_path}{file_name}.pkl', 'wb') as f: 
                pickle.dump([X], f)


if __name__ == "__main__":
    get_coverage()