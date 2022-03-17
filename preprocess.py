from utils.workload_to_sep import get_cmd
from utils.calculate_ngrams import get_ngrams
from utils.distributed_coverage_search import get_coverage
from utils.bank_access_count import count_bank_access
from utils.row_col_address_access import get_address_access

if __name__ == "__main__":
    print("Splitting original workloads...")
    get_cmd(n_cores = 4)
    
    print("Calculating ngrams...")
    for n in [7, 11, 15]:
        get_ngrams(n=n)

    print("Creating ngram vectors...")
    get_coverage()

    print("Creating bank access vectors...")
    count_bank_access()

    print("Creating address access vectors...")
    get_address_access()