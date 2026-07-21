import os.path as osp
import os
import numpy as np
import random

import torch as tr
import torch.nn as nn
import torch.utils.data
import torch.utils.data as Data
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from scipy.linalg import fractional_matrix_power

from utils.alg_utils import EA



def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def cal_acc_comb(loader, model, flag=True, fc=None, args=None):
    start_test = True
    model.eval()
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            inputs = inputs
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred)

    return acc * 100, all_output



def data_alignment(X, num_subjects, args):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
   
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X

def data_loader(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # cross-subject loader
    dset_loaders = {}
    train_bs = args.batch_size

    Xt_copy = Xt
    if args.align:
        # offline EA
        Xs = data_alignment(Xs, args.N - 1, args)
        Xt = data_alignment(Xt, 1, args)

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(
        tr.float32), tr.from_numpy(Yt.reshape(-1, )).to(tr.long)
    Xt = Xt.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt = Xt.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xs, Ys, Xt, Yt = Xs.cuda(), Ys.cuda(), Xt.cuda(), Yt.cuda()

    data_src = Data.TensorDataset(Xs, Ys)
    data_tar = Data.TensorDataset(Xt, Yt)

    # for TL train
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True, drop_last=True)

    # for TL test
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    if args.method == 'EEGNet':
        # IEA baseline EEGNet results.
        # For other TL TTA approaches, IEA is done on-the-fly at test time

        # Online(Incremental) EA
        # Much proper way to do EA for target subject considering online BCIs
        # For offline EA, refer to tl/utils/alg_utils.py
        Xt_aligned = []
        R = 0
        num_samples = 0
        for ind in range(len(Xt_copy)):
            curr = Xt_copy[ind]
            cov = np.cov(curr)
            # Note that the following line is an update of the mean covariance matrix (R), instead of a full recalculation. It is much faster computation in this way.
            # Note also that the covariance matrix calculation should take in all visible samples(trials) for this domain(subject)
            R = (R * num_samples + cov) / (num_samples + 1)
            num_samples += 1
            sqrtRefEA = fractional_matrix_power(R, -0.5)
            # transform the original trial. All latter algorithms only use the transformed data as input
            curr_aligned = np.dot(sqrtRefEA, curr)
            Xt_aligned.append(curr_aligned)
        Xt_aligned = np.array(Xt_aligned)
        # EA done

        Xt_aligned = tr.from_numpy(Xt_aligned).to(tr.float32)
        Xt_aligned = Xt_aligned.unsqueeze_(3)
        if 'EEGNet' in args.backbone:
            Xt_aligned = Xt_aligned.permute(0, 3, 1, 2)
        if args.data_env != 'local':
            Xt_aligned = Xt_aligned.cuda()
        data_tar_online = Data.TensorDataset(Xt_aligned, Yt)
        dset_loaders["Target-Online-Prealigned"] = Data.DataLoader(data_tar_online, batch_size=32, shuffle=False, drop_last=False)

    Xt_copy = tr.from_numpy(Xt_copy).to(tr.float32)
    Xt_copy = Xt_copy.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_copy = Xt_copy.permute(0, 3, 1, 2)
    if args.data_env != 'local':
        Xt_copy = Xt_copy.cuda()
    data_tar_online = Data.TensorDataset(Xt_copy, Yt)

    # for online TL test
    dset_loaders["Target-Online"] = Data.DataLoader(data_tar_online, batch_size=1, shuffle=False, drop_last=False)

    # for online imbalanced dataset
    # only implemented for binary (class_num=2) for now
    class_0_ids = torch.where(Yt == 0)[0][:args.trial_num // 2]
    class_1_ids = torch.where(Yt == 1)[0][:args.trial_num // 4]
    all_ids = torch.cat([class_0_ids, class_1_ids])
    if args.data_env != 'local':
        data_tar_imb = Data.TensorDataset(Xt_copy[all_ids].cuda(), Yt[all_ids].cuda())
    else:
        data_tar_imb = Data.TensorDataset(Xt_copy[all_ids], Yt[all_ids])
    dset_loaders["Target-Online-Imbalanced"] = Data.DataLoader(data_tar_imb, batch_size=1, shuffle=True,
                                                               drop_last=False)
    dset_loaders["target-Imbalanced"] = Data.DataLoader(data_tar_imb, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["Target-Imbalanced"] = Data.DataLoader(data_tar_imb, batch_size=train_bs * 3, shuffle=True,
                                                        drop_last=False)

    return dset_loaders