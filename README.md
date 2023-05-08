# Accurate Open-set Recognition for Memory Workload
These are codes and datasets for "Accurate Open-set Recognition for Memory Workload", submitted to PAKDD, 2022.

## Dataset
We used 2 real-world workload sequence datasets in the experiment.
The following table describes datasets used in our experiment.  

| dataset       | # of known | # of unknown | # of train | # of test of known | # of test of unknown |
|---------------|-----------:|-------------:|-----------:|-------------------:|---------------------:|
| SEC-seq       |         40 |            4 |    586,885 |            293,444 |               93,491 |
| Memtest86-seq |         31 |            3 |    433,334 |            216,696 |               77,018 |

### Sample data for Memtest86-seq data

Due to the size limit (about 2TB) and the anonymous policy, we cannot upload the entire raw data for the second data.
We upload the sample raw data and their feature vectors of the second dataset (Memtest86-seq).
 * `raw_data`: [\[Download\]](https://drive.google.com/file/d/11MUi-WCCtSZot-HvoCeBKHEbKfyegeCs/view?usp=sharing)
 * `intermediate_data`:[\[Download\]](https://drive.google.com/file/d/1IzJsVc9_3yGMMYu2_uDcnyZNWWHEIJb-/view?usp=sharing)
 * `final_data`:[\[Download\]](https://drive.google.com/file/d/1FyNuJ-AARvMdAszMpnYwPYBzZ8xb_Qnd/view?usp=sharing)

### Full feature vectors for Memtest86-seq data

We also upload the full feature vectors of the second dataset (Memtest86-seq).
If you train and test our model, we recommend using the full version of our feature vectors.
* `final_data_original`:[\[Download\]](https://drive.google.com/file/d/1SOg3uk1zGFaJTRQjf6m9ewThEbIwgUmV/view?usp=sharing)

### Raw data description
The following is the first 10 rows in P01 workload:
```
5,1,1,0,648
5,1,0,0,648
1,0,2,1,35778
1,1,2,1,35778
5,0,2,1,640
1,0,1,3,32991
5,1,2,1,640
1,1,1,3,32991
5,0,1,3,640
5,1,1,3,640
```
Where each column indicates: 
1. **_CMD_** - Command ID: Values: \[1,3,5,6,7\]
2. **_Rank_** - Rank number: Range: \[0,1\]
3. **_Bank Group_** - Bank group number. Range: \[0,3\]
4. **_Bank_** - Bank number. Range: \[0,3\]
5. **_Address_** - Row address number when _CMD_ is ACT and col address number when _CMD_ is RDA or WRA. Range: \[-1, 131071\]

Decoding for the **_CMD_** is as following: 
* 1 - ACT
* 3 - RDA
* 5 - WRA
* 6 - PRE
* 7 - PREA

More about CMD field can be found in [Documentation](https://www.jedec.org/sites/default/files/docs/JESD79-4.pdf).


## Model
We provide  MLP and SVD-based detectors trained for the second dataset in `models` directory.
* `model_mlp.pt`: a trained 2-layer MLP model
* `detectors.npy`: SVD-based detectors for all class. Due to the size limit, we provide the download link: [\[Download\]](https://drive.google.com/file/d/1P065Kf64_1OdqC5W-6QAMMOYvL-vdq3l/view?usp=sharing).
* `detectors_threshold.npy`: thresholds for all class


## Code Information
Codes in this directory are implemented by Python 3.7.
This repository contains the code for Acorn, an ACcurate Open-set recognition method for woRkload sequeNce. 

* The code of Acorn is in this directory.
    * `main.py`: the code related to training a classification model, constructing unknown class detectors, and measuring accuracies for our metrics.
    * `svd_detector.py`: the code related to constructing unknown class detectors and evaluating test samples with the detectors.
    * `model.py`: the code related to a known classification model (MLP).
    * `preprocess.py`: the code that preprocesses raw data and creates feature vectors.
    * `utils/workload_to_sep.py`: the code that takes the cmd field from the raw workloads for the convenience of usage in further steps.
    * `utils/calculate_ngrams.py`: the code that calculates n-grams for the training samples.
    * `utils/distributed_coverage_search.py`: the code that creates n-gram set and calculates n-gram vectors.
    * `utils/bank_access_count.py`: the code that counts access to each bank and creates bank access vectors.
    * `utils/row_col_address_access.py`: the code that counts address access and creates address access vectors.
    * `utils/dataloader.py`: the code related to loading workload sequence data, and extracting feature vectors.

## Usage

The required Python packages are described in ./requirments.txt.
If pip3 is installed on your system, you can type the following command to
install the required packages:
```bash
    pip install -r requirements.txt
```

### How to extract feature vectors for sample data

Type the following command to extract feature vectors for the cmd field and the address-related fields:
```bash
    python preprocess.py
```
The script will create the following two folders:
```
current directory
├── final_data
└── intermediate_data
```
and reprocessed files are stored in:
```
current directory
└── final_data
    ├── 7-grams
    ├── 11-grams
    ├── 15-grams
    ├── data_split_ids
    ├── bank_access_counts
    └── row_col_address_access_counts
```


### How to train and test unknown workload detection for sample data

Type the following command for training and testing new workload detection:  
```bash
    python main.py --data './final_data_original' --batch_size 128 --learning_rate 0.0001 --alpha 2 --only_test false
```
If you already have models and want to test them, type the following command:
```bash
    python main.py --data './final_data_original' --batch_size 128 --learning_rate 0.0001 --alpha 2 --only_test true
``` 

### Citation
Please cite this paper when you use our code.
```
@article{jang2022accurate,
  title={Accurate Open-set Recognition for Memory Workload},
  author={Jang, Jun-Gi and Shim, Sooyeon and Egay, Vladimir and Lee, Jeeyong and Park, Jongmin and Chae, Suhyun and Kang, U},
  journal={arXiv preprint arXiv:2212.08817},
  year={2022}
}
```  
