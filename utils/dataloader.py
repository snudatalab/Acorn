from os import listdir
from os.path import isfile, join
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def concatenate_ngram_vecs(ids_path, path_7gram, path_11gram, path_15gram, num_known_classes):
    X1_train, X1_valid, X1_test, X_out1, Y_train, Y_valid, Y_test = read_ngram_vec(ids_path, path_7gram, num_known_classes)

    X2_train, X2_valid, X2_test, X_out2, _, _, _ = read_ngram_vec(ids_path, path_11gram, num_known_classes)
    X3_train, X3_valid, X3_test, X_out3, _, _, _ = read_ngram_vec(ids_path, path_15gram, num_known_classes)

    X_train = np.concatenate([X1_train, X2_train, X3_train], axis=1)
    X_valid = np.concatenate([X1_valid, X2_valid, X3_valid], axis=1)
    X_test = np.concatenate([X1_test, X2_test, X3_test], axis=1)
    X_out = np.concatenate([X_out1, X_out2, X_out3], axis=1)    

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)    
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    
    X_out = scaler.transform(X_out)    

    return X_train, X_valid, X_test, X_out, Y_train, Y_valid, Y_test

def concatenate_address_vecs(ids_path, path_bank, path_address, num_known_classes):
    X_train_bank, X_valid_bank, X_test_bank, X_out_bank = read_bank_vec(ids_path, path_bank, num_known_classes)
    X_train_address, X_valid_address, X_test_address, X_out_address = read_address_vec(ids_path, path_address, num_known_classes)
    
    X_train = np.concatenate([X_train_bank, X_train_address], axis=1)
    X_valid = np.concatenate([X_valid_bank, X_valid_address], axis=1)
    X_test = np.concatenate([X_test_bank, X_test_address], axis=1)
    X_out = np.concatenate([X_out_bank, X_out_address], axis=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)    
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    
    X_out = scaler.transform(X_out)    
    
    return X_train, X_valid, X_test, X_out

def read_ngram_vec(ids_path, mypath, num_known_classes):

    idx_path = ids_path
    split_train_ids_dic, split_valid_ids_dic, split_test_ids_dic = train_test_idx(idx_path)

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)

    X1_train = []
    X1_valid = []
    X1_test = []
    X_out = []

    Y_train = []
    Y_valid = []
    Y_test = []

    for i, file_name in tqdm(enumerate(onlyfiles)):
        path = mypath + file_name
        with open(path, 'rb') as f:
            X_tmp = np.array(pickle.load(f)).squeeze()
            if i < num_known_classes:
                X1_train.append(X_tmp[split_train_ids_dic[i],:])
                X1_valid.append(X_tmp[split_valid_ids_dic[i],:])
                X1_test.append(X_tmp[split_test_ids_dic[i],:])        
                length1 = len(split_train_ids_dic[i])
                length2 =len(split_valid_ids_dic[i])
                length3 = len(split_test_ids_dic[i])
                Y_tmp1 = np.ones((length1,)) * i
                Y_tmp2 = np.ones((length2,)) * i
                Y_tmp3 = np.ones((length3,)) * i        
                Y_train.append(Y_tmp1)
                Y_valid.append(Y_tmp2)
                Y_test.append(Y_tmp3)
            else:
                X_out.append(X_tmp)

    # Standardization
    X1_train = np.concatenate(X1_train, axis=0)
    scaler = StandardScaler()
    scaler.fit(X1_train)
    X1_train = scaler.transform(X1_train)

    Y_train = np.concatenate(Y_train, axis=0)
    Y_valid = np.concatenate(Y_valid, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    X1_valid = np.concatenate(X1_valid, axis=0)
    X1_test = np.concatenate(X1_test, axis=0)
    X_out = np.concatenate(X_out, axis=0)    

    X1_valid = scaler.transform(X1_valid)
    X1_test = scaler.transform(X1_test)
    
    X_out = scaler.transform(X_out)

    return X1_train, X1_valid, X1_test, X_out, Y_train, Y_valid, Y_test


def train_test_idx(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    split_train_ids_dic = {}
    split_valid_ids_dic = {}
    split_test_ids_dic = {}
    for i in range(len(onlyfiles)):
        path = mypath + onlyfiles[i]
        with open(path, 'rb') as f:
            (train_ids, val_ids, test_ids) = pickle.load(f)
            split_train_ids_dic[i] = train_ids
            split_valid_ids_dic[i] = val_ids
            split_test_ids_dic[i] = test_ids 

    return split_train_ids_dic, split_valid_ids_dic, split_test_ids_dic

def read_bank_vec(ids_path, mypath, num_known_classes):
    idx_path = ids_path
    split_train_ids_dic, split_valid_ids_dic, split_test_ids_dic = train_test_idx(idx_path)

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)

    X0_train = []
    X0_valid = []
    X0_test = []
    X_out = []
    sum_l = 0
    test_X = []
    for i, file_name in tqdm(enumerate(onlyfiles)):
        path = mypath + file_name     
        with open(path, 'rb') as f:
            X_tmp = np.array(pickle.load(f)).squeeze()            
            if i<num_known_classes:
                X0_train.append(X_tmp[split_train_ids_dic[i],:])
                X0_valid.append(X_tmp[split_valid_ids_dic[i],:])
                X0_test.append(X_tmp[split_test_ids_dic[i],:])
            else:
                X_out.append(X_tmp)


    X0_train = np.concatenate(X0_train, axis=0)
    scaler = StandardScaler()
    scaler.fit(X0_train)
    X0_train = scaler.transform(X0_train)

    X0_valid = np.concatenate(X0_valid, axis=0)
    X0_test = np.concatenate(X0_test, axis=0)
    X_out = np.concatenate(X_out, axis=0)

    X0_valid = scaler.transform(X0_valid)
    X0_test = scaler.transform(X0_test)
    X_out = scaler.transform(X_out)    

    return X0_train, X0_valid, X0_test, X_out

def read_address_vec(ids_path, mypath, num_known_classes):
    idx_path = ids_path
    split_train_ids_dic, split_valid_ids_dic, split_test_ids_dic = train_test_idx(idx_path)

    X_train_address = []
    X_valid_address = []
    X_test_address = []
    X_out = []

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = sorted(onlyfiles)

    
    for i, name in tqdm(enumerate(onlyfiles)):
        filename = name
        loadname = mypath+filename
        address = np.load(loadname,allow_pickle=True)
        if i< num_known_classes:
            X_train_address.append(address[split_train_ids_dic[i],:])
            X_valid_address.append(address[split_valid_ids_dic[i],:])
            X_test_address.append(address[split_test_ids_dic[i],:])
        else:
            X_out.append(address)

            
    X_train_address = np.concatenate(X_train_address, axis=0)
    scaler = StandardScaler()
    scaler.fit(X_train_address)
    X_train_address = scaler.transform(X_train_address)

    X_valid_address = np.concatenate(X_valid_address, axis=0)
    X_test_address = np.concatenate(X_test_address, axis=0)    
    X_out = np.concatenate(X_out, axis=0)        

    X_valid_address = scaler.transform(X_valid_address)
    X_test_address = scaler.transform(X_test_address)
    X_out = scaler.transform(X_out)    


    return X_train_address, X_valid_address, X_test_address, X_out

