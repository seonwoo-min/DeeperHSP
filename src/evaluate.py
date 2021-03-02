# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


def compute_metrics(labels, outputs, average=True):
    """ compute set-wise evaluation metrics """
    num_samples, num_classes = outputs.shape
    predictions = np.argmax(outputs, 1)

    ACC, F1, PR, RE, SP, MCC, AUROC, AUPR = [], [], [], [], [], [], [], []
    for i in range(num_classes):
        Y, P, Z = np.zeros(num_samples), np.zeros(num_samples), np.zeros(num_samples)
        for j in range(num_samples):
            if labels[j] == i:      Y[j] = 1
            if predictions[j] == i: Z[j] = 1
            P[j] = outputs[j, i]

        A = confusion_matrix(Y, Z)
        if   len(set(Y)) > 1: tp, fp, fn, tn = A[1, 1], A[0, 1], A[1, 0], A[0, 0]
        else:                 tp, fp, fn, tn = 0, 0, 0, 0
        ACC.append(float(tp + tn) / float(tp + fp + fn + tn) if tp + fp + fn + tn > 0 else 0)
        F1.append(float(2 * tp) / float(2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0)
        PR.append(float(tp) / float(tp + fp) if tp + fp > 0 else 0)
        RE.append(float(tp) / float(tp + fn) if tp + fn > 0 else 0)
        SP.append(float(tn) / float(fp + tn) if fp + tn > 0 else 0)
        denominator = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        MCC.append((float(tp * tn) - float(fp * fn)) / denominator if denominator > 0 else 0)
        AUROC.append(roc_auc_score(Y, P) if len(set(Y)) > 1 else 0)
        AUPR.append(average_precision_score(Y, P) if len(set(Y)) > 1 else 0)

    if average:
        acc = np.sum(labels == predictions) / num_samples
        return (acc, *[np.average(m) for m in [F1, PR, RE, SP, MCC, AUROC, AUPR]])
    else:
        return ACC, F1, PR, RE, SP, MCC, AUROC, AUPR