import numpy as np
from tqdm import tqdm
from scipy import linalg

def evaluate_performance(test_true_label, test_pred_labels, out_pred_labels, num_test, num_out):
    cor = 0
    for i in range(num_test):
        if test_pred_labels[i] == test_true_label[i]:
            cor +=1

    out_cor = 0
    for i in range(num_out):
        if out_pred_labels[i] == -1:
            out_cor +=1

    prec_cor = 0
    for i in range(num_test):
        if test_pred_labels[i] == -1:
            prec_cor +=1

    known_class_accuracy = cor/num_test
    recall_unknown = out_cor/num_out
    precision_unknown = out_cor/(out_cor + prec_cor)
    print(f'Known Class Accuracy: {known_class_accuracy:.4f}, Recall for Unknown: {recall_unknown:.4f}, Precision for Unknown: {precision_unknown:.4f}')


def unknown_evaluation(feature_vector, label, v_collections, threshold, alpha):

    recon = np.matmul(np.matmul(feature_vector, v_collections[label]), v_collections[label].T)
    recon_error = np.linalg.norm(feature_vector - recon)

    thres = threshold[label,0] + threshold[label, 1] * alpha

    final_label = label
    if (recon_error < thres):
        final_label = label
    else:
        final_label = -1

    return final_label

def detector_construction(feature_vectors, labels, num_classes):
    v_collections = []

    print('Construct detectors for each pair of class and feature type')
    
    for i in tqdm(range(num_classes)):
        find_ind = np.where(labels == i)[0]
        u,s,v = linalg.svd(feature_vectors[find_ind,:], full_matrices=False)
        energy = 0
        total_energy = np.sum(s*s)
        stop_crit = 1
        for j in range(len(s)):
            energy = energy + s[j]*s[j]
            if energy/total_energy > 0.999:
                stop_crit = j+1
                break
        v_collections.append(v[:stop_crit+1,:].T)

    print('Detector consturction is done')

    print('Find thresholds...')

    train_recon_errors = np.zeros((feature_vectors.shape[0],))
    for i in tqdm(range(feature_vectors.shape[0])):
        class_id = int(labels[i])
        recon = np.matmul(np.matmul(feature_vectors[i,:], v_collections[class_id]), v_collections[class_id].T)
        recon_error = np.linalg.norm(feature_vectors[i,:] - recon)
        train_recon_errors[i] = recon_error

    meanNstd = np.zeros((num_classes,2))
    for i in range(num_classes):
        find_ind = np.where(labels == i)[0]
        meanNstd[i,0] = np.mean(train_recon_errors[find_ind])
        meanNstd[i,1] = np.std(train_recon_errors[find_ind])


    print('Finding thresholds is done')

    return v_collections, meanNstd


