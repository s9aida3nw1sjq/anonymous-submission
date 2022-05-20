from glob import glob
import os
import random

import causaldag as cd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import tensorflow as tf
import torch


def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    torch.manual_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def get_cpdag(B):
    assert is_dag(B)
    return cd.DAG.from_amat(B).cpdag().to_amat()[0]


def compute_shd_cpdag(B_bin_true, B_bin_est):
    assert is_dag(B_bin_true)
    assert is_dag(B_bin_est)
    cpdag_true = get_cpdag(B_bin_true)
    cpdag_est = get_cpdag(B_bin_est)
    return cd.PDAG.from_amat(cpdag_true).shd(cd.PDAG.from_amat(cpdag_est))


def count_precision_recall_f1(tp, fp, fn):
    # Precision
    if tp + fp == 0:
        precision = None
    else:
        precision = float(tp) / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = None
    else:
        recall = float(tp) / (tp + fn)

    # F1 score
    if precision is None or recall is None:
        f1 = None
    elif precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def count_skeleton_accuracy(B_bin_true, B_bin_est):
    dag_est = cd.DAG.from_amat(B_bin_est)
    dag_true = cd.DAG.from_amat(B_bin_true)
    cm_skeleton = dag_est.confusion_matrix_skeleton(dag_true)
    tp_skeleton = cm_skeleton['num_true_positives']
    fp_skeleton = cm_skeleton['num_false_positives']
    fn_skeleton = cm_skeleton['num_false_negatives']
    precision_skeleton, recall_skeleton, f1_skeleton \
        = count_precision_recall_f1(tp_skeleton, fp_skeleton, fn_skeleton)

    return {'f1_skeleton': f1_skeleton, 'precision_skeleton': precision_skeleton, 'recall_skeleton': recall_skeleton}


def count_arrows_accuracy(B_bin_true, B_bin_est):
    dag_est = cd.DAG.from_amat(B_bin_est)
    dag_true = cd.DAG.from_amat(B_bin_true)
    cm_cpdag = dag_est.confusion_matrix(dag_true)
    tp_arrows = len(cm_cpdag['true_positive_arcs'])
    fp_arrows = len(cm_cpdag['false_positive_arcs'])
    fn_arrows = len(cm_cpdag['false_negative_arcs'])
    precision_arrows, recall_arows, f1_arrows \
        = count_precision_recall_f1(tp_arrows, fp_arrows, fn_arrows)
    return {'f1_arrows': f1_arrows, 'precision_arrows': precision_arrows, 'recall_arows': recall_arows}


def count_dag_accuracy(B_bin_true, B_bin_est):
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B_bin_est)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    if pred_size == 0:
        fdr = None
    else:
        fdr = float(len(reverse) + len(false_pos)) / pred_size
    if len(cond) == 0:
        tpr = None
    else:
        tpr = float(len(true_pos)) / len(cond)
    if cond_neg_size == 0:
        fpr = None
    else:
        fpr = float(len(reverse) + len(false_pos)) / cond_neg_size
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    precision, recall, f1 = count_precision_recall_f1(tp=len(true_pos),
                                                      fp=len(reverse) + len(false_pos),
                                                      fn=len(false_neg))
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 
            'precision': precision, 'recall': recall, 'f1': f1}


def count_accuracy(B_bin_true, B_bin_est):
    """Compute various accuracy metrics for B_bin_est."""
    results = {}
    try:
        # Calculate performance metrics for DAG
        results_dag = count_dag_accuracy(B_bin_true, B_bin_est)
        results.update(results_dag)
    except:    # To be safe
        pass

    try:
        # Calculate SHD-CPDAG
        shd_cpdag = compute_shd_cpdag(B_bin_true, B_bin_est)
        results['shd_cpdag'] = shd_cpdag
    except:    # To be safe
        pass

    try:
        # Calculate performance metrics for skeleton
        results_skeleton = count_skeleton_accuracy(B_bin_true, B_bin_est)
        results.update(results_skeleton)
    except:    # To be safe
        pass

    try:
        # Calculate performance metrics for arrows
        results_arrows = count_arrows_accuracy(B_bin_true, B_bin_est)
        results.update(results_arrows)
    except:    # To be safe
        pass
    return results