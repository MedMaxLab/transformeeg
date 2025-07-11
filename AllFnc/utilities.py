import argparse
import glob
import itertools
import os
import pickle
import pandas as pd
import numpy as np
from math import floor
import torch
from collections.abc import Iterable, Callable
from numpy.typing import ArrayLike
from scipy import linalg

def makeGrid(pars_dict):
        keys = pars_dict.keys()
        combinations = itertools.product(*pars_dict.values())
        ds = [dict(zip(keys, cc)) for cc in combinations]
        return ds

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.casefold() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.casefold() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v):
    if (v is None) or isinstance(v, list):
        return v
    elif v=="None":
        return None
    else:
        v = v.split('\'')
        v = [i for n, i in enumerate(v) if n % 2 != 0]
        return v

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0)" % (x,))
    return x

def positive_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x

def positive_int_nozero(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an integer" % (x,))

    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x

def positive_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an integer" % (x,))

    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x

def findSeq(inputList):
    """
    Taken from https://stackoverflow.com/questions/38621615/find-the-set-of-elements-
    in-a-list-without-sorting-in-python
    """
    dictionary = {}
    newList = []
    for elem in inputList:
        if elem not in dictionary:
            dictionary[elem] = True
            newList += [elem]
    return newList

def get_aug_idx(augmentation_to_idx):
    i = augmentation_to_idx
    if i == 'flip_horizontal':
        return 0
    elif i == 'flip_vertical':
        return 1
    elif i == 'add_band_noise':
        return 2
    elif i == 'add_eeg_artifact':
        return 3
    elif i == 'add_noise_snr':
        return 4
    elif i == 'channel_dropout':
        return 5
    elif i == 'masking':
        return 6
    elif i == 'warp_signal':
        return 7
    elif i == 'random_FT_phase':
        return 8
    elif i == 'phase_swap':
        return 9

def get_aug_name(idx_to_augmentation):
    i = idx_to_augmentation
    if i == 0:
        return "None"
    elif i == 1:
        return 'flip_horizontal'
    elif i == 2:
        return 'flip_vertical'
    elif i == 3:
        return 'add_band_noise'
    elif i == 4:
        return 'add_eeg_artifact'
    elif i == 5:
        return 'add_noise_snr'
    elif i == 6:
        return 'channel_dropout'
    elif i == 7:
        return 'masking'
    elif i == 8:
        return 'warp_signal'
    elif i == 9:
        return 'random_FT_phase'
    elif i == 10:
        return 'phase_swap'

def column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df.reindex(columns = i)
    return df


def gather_results(save = False):
    
    metrics_list = [ 
        'accuracy_unbalanced', 'accuracy_weighted',
        'precision_micro',     'precision_macro',   'precision_weighted',
        'recall_micro',        'recall_macro',      'recall_weighted',
        'f1score_micro',       'f1score_macro',     'f1score_weighted',
        'rocauc_micro',        'rocauc_macro',      'rocauc_weighted',
        'cohen_kappa'
    ]
    
    piece_list = [
        'task',         'pipeline', 'sampling_rate',
        'model',      'outer_fold',    'inner_fold',
        'channels',       'window',       'overlap',
        'csp_filters', 'first_aug',    'second_aug',
        'learning_rate'
    ]

    path_list = ["baseline", "baseline_with_aug"]
    eval_list = ["th_standard", "th_corrected", "subject"]
    for eval_type in eval_list:
        result_tables = []
        for search_path in path_list:  
            file_list = glob.glob(f'**/Results/{search_path}/*.pickle') 
            results_list = [None]*len(file_list)
            for i, path in enumerate(file_list):
        
                # Get File name
                file_name = path.split(os.sep)[-1]
                file_name = file_name[:-7]
                
                # Get all name pieces
                pieces = file_name.split('_')
                pieces = pieces[:13]
                
                # convert to numerical some values
                for k in [2, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                    if k == 8:
                        pieces[k] = int(pieces[k])*100
                    if k == 10 or k == 11:
                        pieces[k]= get_aug_name(int(pieces[k]))
                    elif k == 12:
                        pieces[k] = int(pieces[k])/1e6
                    else:
                        pieces[k] = int(pieces[k])
                pieces.append(eval_type)
        
                # open results
                with open(path, "rb") as f:
                    mdl_res = pickle.load(f)
        
                # append results
                for metric in metrics_list:
                    try:
                        pieces.append(mdl_res[eval_type][metric])
                    except Exception:
                        print(mdl_res.keys())
        
                # final list
                results_list[i] = pieces
        
            # convert to DataFrame and swap two columns for convenience
            results_table = pd.DataFrame(
                results_list,
                columns= piece_list + ["evaluation_type"] + metrics_list
            )
            results_table = column_switch(results_table, 'model', 'sampling_rate')
            results_table.sort_values(
                ['model', 'inner_fold', 'outer_fold'],
                ascending=[True, True, True],
                inplace=True
            )
            result_tables.append(results_table)
            # store if required
            if save:
                csv_filename = search_path
                results_table.to_csv(
                    f'csv_summary_files/ResultsTable_{search_path}_{eval_type}.csv',
                    index=False
                )
        
    return result_tables


def GetLrDict():
    model_list = [
        'eegnet', 'shallownet', 'xeegnet', 'deepconvnet',
        'eegconformer', 'atcnet', 'psdnet3','resnet', 'transformeeg'
    ]
    lr_dict = {i: {'parkinson': 1e-04} for i in model_list}
    return lr_dict

def GetLearningRateString(model, task):
    return str(int(0.00025*1e6)).zfill(6)


class CSPScaler:
    def __init__(
        self,
        Nfilters: int = 2,
        use_torch: bool = True,
        device: str or torch.device = 'cpu'
    ):
        self.Wcsp = None
        self.Nfilters = Nfilters
        self._use_torch = use_torch
        self._device = device

    def _cov(self, data, array = False):
        if array:
            numerator = np.matmul(data, np.transpose(data, (0,2,1)))
            denominator = np.atleast_3d(np.expand_dims(np.trace(numerator, axis1=-2, axis2=-1),1))
            cov_list = numerator / denominator
        else:
            cov_list = []
            for trial in data:
                cov = trial @ trial.T / np.trace(trial @ trial.T)
                cov_list.append(cov)
                
        cov_mean = np.mean(cov_list, axis=0)
        return cov_mean
    
    def _zero_mean(self, data, array = False):
        if array:
            data_out = data - np.mean(data, axis=2, keepdims=True)
        else:
            data_out = []
            for trial in data:
                trial_norm = trial - np.mean(trial, axis=1, keepdims=True)
                data_out.append(trial_norm)
        return data_out

    def _compute_cov(self, data1, data2, array = False):
        data1_0m = self._zero_mean(data1, array)
        data2_0m = self._zero_mean(data2, array)
        cov1 = self._cov(data1_0m, array)
        cov2 = self._cov(data2_0m, array)
        return cov1, cov2
        
    def _compute_csp(self, cov1: ArrayLike, cov2: ArrayLike):
        # Compute the composite covariance matrix
        C = cov1 + cov2
        
        # Perform eigenvalue decomposition of the composite covariance matrix
        Lc, Uc = np.linalg.eigh(C)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx_sort = np.argsort(Lc)[::-1]
        Lc = Lc[idx_sort]
        Uc = Uc[:, idx_sort]
        
        # Whitening transformation matrix
        P = np.diag(1.0 / np.sqrt(Lc)) @ Uc.T
        
        # Whiten the covariance matrices of both classes
        S1 = P @ cov1 @ P.T
        S2 = P @ cov2 @ P.T
        
        # Solve the generalized eigenvalue problem in whitened space
        Ls, B = linalg.eigh(S1, S2)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx_sort = np.argsort(Ls)[::-1]
        Ls = Ls[idx_sort]
        B = B[:, idx_sort]
        
        # Compute the CSP projection matrix
        W_CSP = P.T @ B
        self.Wcsp = W_CSP
    
    def fit(
        self,
        data1: ArrayLike = None,
        data2: ArrayLike = None,
        path_list_1: Iterable = None,
        path_list_2: Iterable = None,
        load_func: Callable = None,
        load_func_args: list or dict = None,
    ):
        
        # Import the data
        if (data1 is None) or (data2 is None):
            if isinstance(load_func_args, list):
                data1 = [load_func(path, *load_func_args) for path in path_list_1]
                data2 = [load_func(path, *load_func_args) for path in path_list_2]
            elif isinstance(load_func_args, dict):
                data1 = [load_func(path, **load_func_args) for path in path_list_1]
                data2 = [load_func(path, **load_func_args) for path in path_list_2]
            else:
                data1 = [load_func(path) for path in path_list_1]
                data2 = [load_func(path) for path in path_list_2]
            
    
        # Compute the covariance matrices
        if isinstance(data1, list):
            cov1, cov2 = self._compute_cov(data1, data2)
        else:
            cov1, cov2 = self._compute_cov(data1, data2, array = True)
        self._compute_csp(cov1, cov2)
        if self._use_torch:
            self.Wcsp = torch.from_numpy(self.Wcsp).to(device=self._device)

    def set_filter_number(self, N: int= 2):
        if N>0:
            self.Nfilters = N
        else:
            raise ValueError("number of filters must be a positive integer")

    def get_csp_matrix(self):
        if self._use_torch:
            return torch.clone(self.Wcsp)
        else:
            return np.copy(self.Wcsp)

    def __call__(self, X):
        return self.transform(X)
    
    def transform(self, X):
        if self._use_torch:
            X = torch.matmul(self.Wcsp, X)
            X = torch.cat((X[..., :self.Nfilters, :], X[..., -self.Nfilters:, :]), dim=-2)
        else:
            X = np.matmul(self.Wcsp, X)
            X = np.concatenate((X[..., :self.Nfilters, :], X[..., -self.Nfilters:, :]), axis=-2)
        return X
