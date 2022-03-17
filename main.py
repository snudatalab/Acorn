import math
import os
from time import time

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.special import softmax
from torch.utils.data import Dataset, DataLoader

from model import FNN
from utils.dataloader import concatenate_ngram_vecs, concatenate_address_vecs
from svd_detector import detector_construction, unknown_evaluation, evaluate_performance

from tqdm import tqdm


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WorkloadDataset(Dataset):
    def __init__(self, numpy_data, numpy_dataY):
        self.data = np.concatenate((numpy_data,  numpy_dataY[:,np.newaxis]), axis=1)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

@click.command()
@click.option('--data', type=click.Path(), default='./data')
@click.option('--batch_size', type=int, default=128)
@click.option('--learning_rate', type=float, default=0.0001)
@click.option('--alpha', type=float, default=0)
@click.option('--only_test', type=bool, default=False)
@click.option('--num_known_classes', type=int, default=31)
def main(data, batch_size, learning_rate, alpha, only_test, num_known_classes):

    num_classes = num_known_classes
    ids_path = os.path.join(data, 'data_split_ids/')
    path_7gram = os.path.join(data, '7-grams/')
    path_11gram = os.path.join(data, '11-grams/')
    path_15gram = os.path.join(data, '15-grams/')

    mypath_bank = os.path.join(data, 'bank_access_counts/')
    mypath_address = os.path.join(data, 'row_col_address_access_counts/')


    X_train_ngram, X_valid_ngram, X_test_ngram, X_out_ngram, Y_train, Y_valid, Y_test = concatenate_ngram_vecs(ids_path, path_7gram, path_11gram, path_15gram, num_classes)
    X_train_address, X_valid_address, X_test_address, X_out_address = concatenate_address_vecs(ids_path, mypath_bank, mypath_address, num_classes)

    X_train = np.concatenate((X_train_ngram, X_train_address), axis=1)
    X_valid = np.concatenate((X_valid_ngram, X_valid_address), axis=1)
    X_test = np.concatenate((X_test_ngram, X_test_address), axis=1)

    X_out = np.concatenate((X_out_ngram, X_out_address), axis=1)

    Y_out = np.ones((X_out.shape[0],))*(num_classes)
    
    trn_loader = DataLoader(WorkloadDataset(X_train, Y_train), num_workers=8, batch_size=batch_size, shuffle=True)
    vld_loader = DataLoader(WorkloadDataset(X_valid, Y_valid), num_workers=8, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WorkloadDataset(X_test, Y_test), num_workers=8, batch_size=batch_size, shuffle=False)
    out_loader = DataLoader(WorkloadDataset(X_out, Y_out), num_workers=8, batch_size=batch_size, shuffle=False)
    
    

    input_size = X_train.shape[1]
    net = FNN(input_size, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    if only_test:
        net = torch.load('./models/model_mlp.pt')
        v_col = np.load('./models/detectors.npy', allow_pickle=True)
        thres = np.load('./models/detectors_threshold.npy')
        print('Loading models is done')

    else:
        print('Training a classification model...')
        best_vld_loss = math.inf
        for epoch in range(5):
            t1 = time()
            trn_loss, vld_loss, test_loss = 0., 0., 0.
            trn_hit, vld_hit, test_hit = 0, 0, 0
            trn_total, vld_total, test_total = 0, 0, 0
            loss_cmd = 0
            loss_address = 0

            net.train()
            for i, batch in enumerate(trn_loader, 0):
                x, y = batch[:, :-1], batch[:, -1]
                x = x.type(torch.FloatTensor).to(DEVICE)
                y = y.type(torch.LongTensor).to(DEVICE)
                optimizer.zero_grad()
                pred = net(x)

                loss = criterion(pred, y)

                loss.backward()
                optimizer.step()
                trn_loss += loss.item()

                pred_cls = torch.argmax(pred, dim=1)
                trn_hit += sum(pred_cls == y)
                trn_total += y.shape[0]


            trn_loss /= (i+1)
            trn_acc = float(trn_hit)/trn_total

            net.eval()
            for i, batch in enumerate(vld_loader, 0):
                x, y = batch[:, :-1], batch[:, -1]
                x = x.type(torch.FloatTensor).to(DEVICE)
                y = y.type(torch.LongTensor).to(DEVICE)
                pred = net(x)
                loss = criterion(pred, y)
                vld_loss += loss.item()
                pred_cls = torch.argmax(pred, dim=1)
                vld_hit += sum(pred_cls == y)
                vld_total += y.shape[0]

            vld_loss /= (i+1)
            vld_acc = float(vld_hit)/vld_total

            for i, batch in enumerate(test_loader, 0):
                x, y = batch[:, :-1], batch[:, -1]
                x = x.type(torch.FloatTensor).to(DEVICE)
                y = y.type(torch.LongTensor).to(DEVICE)
                pred = net(x)
                loss = criterion(pred, y)
                test_loss += loss.item()
                pred_cls = torch.argmax(pred, dim=1)
                test_hit += sum(pred_cls == y)
                test_total += y.shape[0]

            test_loss /= (i+1)
            test_acc = float(test_hit)/test_total

            if vld_loss < best_vld_loss:
                best_vld_loss = vld_loss
                best_result = f'[BEST] Epoch: {epoch+1:03d}, TrnLoss: {trn_loss:.4f}, VldLoss: {vld_loss:.4f}, TestLoss: {test_loss:.4f}, ' \
                              f'TrnAcc: {trn_acc:.4f}, VldAcc: {vld_acc:.4f}, TestAcc: {test_acc:.4f}'
            print(f'[{time()-t1:.2f}sec] Epoch: {epoch+1:03d}, TrnLoss: {trn_loss:.4f}, VldLoss: {vld_loss:.4f}, TestLoss: {test_loss:.4f}, '
                  f'TrnAcc: {trn_acc:.4f}, VldAcc: {vld_acc:.4f}, TestAcc: {test_acc:.4f}')
        print(best_result)
        torch.save(net, './models/model_mlp.pt')


        print('Finish to train a classification model')

        print('Start unknown class detector construction')

        v_col, thres = detector_construction(X_train, Y_train, num_classes)
        save_v = np.empty((len(v_col),), dtype=object)
        for i in range(len(v_col)):
                save_v[i] = v_col[i]
        np.save('./models/detectors.npy', save_v)
        np.save('./models/detectors_threshold.npy', thres)

        print('Finish to construct unknown class detectors')

    print('Test new workload detection')


    print('predict labels for known and unknown classes')
    test_pred_res = []
    test_true_label = []
    for i, batch in tqdm(enumerate(test_loader, 0)):
        x, y = batch[:, :-1], batch[:, -1]
        x = x.type(torch.FloatTensor).to(DEVICE)
        y = y.type(torch.LongTensor).to(DEVICE)
        pred = net(x)
        test_pred_res.append(pred.to(torch.device("cpu")).detach().numpy())
        test_true_label.append(y.to(torch.device("cpu")).detach().numpy())

    y_prob = np.vstack(test_pred_res)
    test_true_label = np.concatenate(test_true_label)

    y_prob3 = softmax(y_prob, axis=1)

    out_pred_res = []
    out_true_label = []
    for i, batch in tqdm(enumerate(out_loader, 0)):
        x, y = batch[:, :-1], batch[:, -1]
        x = x.type(torch.FloatTensor).to(DEVICE)
        y = y.type(torch.LongTensor).to(DEVICE)
        pred = net(x)
        out_pred_res.append(pred.to(torch.device("cpu")).detach().numpy())
        out_true_label.append(y.to(torch.device("cpu")).detach().numpy())

    y_prob2 = np.vstack(out_pred_res)
    out_true_label = np.concatenate(out_true_label)
    y_prob4 = softmax(y_prob2, axis=1)

    print('Unknown class detection')

    test_labels = []
    for i in tqdm(range(X_test.shape[0])):
        min_ind = np.argmax(y_prob3[i,:])
        final_label = unknown_evaluation(X_test[i,:], min_ind, v_col, thres, alpha)
        test_labels.append(final_label)

    out_labels = []
    for i in tqdm(range(X_out.shape[0])):
        min_ind = np.argmax(y_prob4[i,:])
        final_label = unknown_evaluation(X_out[i,:], min_ind, v_col, thres, alpha)
        out_labels.append(final_label)


    print('Measure accuracies for known class accuracy, precision, and recall for unknown')
    evaluate_performance(test_true_label, test_labels, out_labels, X_test.shape[0], X_out.shape[0])
    

if __name__ == '__main__':
    main()



