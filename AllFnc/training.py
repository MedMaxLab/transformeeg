from __future__ import annotations
import matplotlib.pyplot as plt
import mne
import autoreject
import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy.stats import zscore
import selfeeg.augmentation as aug
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Union
from .utilities import GetLrDict, findSeq
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    roc_curve,
)

__all__ = [
    'loadEEG',
    'lossBinary',
    'lossMulti',
    'get_performances',
    'GetLearningRate',
]

def loadEEG(path: str, 
    return_label: bool=True, 
    downsample: bool=False,
    use_only_original: bool= False,
    apply_zscore: bool = True,
    detect_noise: bool = False,
    zscore_no_noise: bool = False,
    onehot_label: bool = False,
    winlen: float = -1.0,
    nclass: int  = -1,
    seed: int = 83136297,
):
    '''
    ``loadEEG`` loads the entire EEG signal stored in path.
    It is supposed to load pickle files with names 
    
        {dataset_ID}_{subject_ID}_{session_ID}_{object_ID}.pickle
    
    where each file contains a dictionary with keys:
        
        - 'data'  : for the signal.
        - 'label' : for the label. 

    Parameters
    ----------
    path: str
        The full path to the pickle file.
    return_label: bool, optional
        Whether to return the label or not. The function GetEEGPartitionNumber
        doesn't want a label. That's why we added the option to omit it.
        Default = True
    downsample: bool, optional
        Whether to downsample the EEG data to 125 Hz or not. Note that all files
        are supposed to have 250 Hz, since they come from the BIDSAlign preprocessing
        pipeline presented in the paper.
        Default = False
    use_only_original: bool, optional
        Whether to use only the original EEG channels or not. BIDSAlign apply a 
        template alignment, which included a spherical interpolation of channels not
        included in the library's 10_10 61 channels template.
        Default = False
    apply_zscore: bool, optional
        Whether to apply the z-score on each channel or not. 
        Default = True
    detect_noise: bool, optional
        Whether to detect noisy partitions of the EEG to be considered as an extra
        noisy class. Noisy class will have the highest possible integer class. For
        example, in binary tasks, 0,1 will remain the original classes, while 2 will
        be the new added noise class.
        Since the addition of a new class will result in a multi-class calssification
        problem, y will be a new one_hot encoded tensor.
        Noise is detected with autoreject.get_rejection_threshold function.
        If this is true, you must also pass the window length, and number of classes.
        This function also assumes a sampling rate of either 250 or 125
        (except for the MI task, 160) according to the downsampling argument.
        It also assumes that windows have 0 percent overlap.
        Default= False
    onehot_label: bool, optional
        Whether to apply one-hot encoding on tasks with at least 3 classes.
        This is ignored if detect_noise is set to True.
        Default = False
    winlen: float, optional
        The window length. It will be used to epoch the EEG and detect noisy pieces.
        Default = -1.0
    nclass: int, optional
        The number of classes. It will be used to determine which integer the noisy
        class should be assigned.
        Default = -1
    seed: int, optional
        The seed used by the autoreject.get_rejection_threshold function.
        Default = 83136297

    Returns
    -------
    x: Arraylike
        The arraylike object with the entire eeg signal to be partitioned by the 
        Pytorch's Dataset class (or whatever function is assigned for such task)
    y: float or ArrayLike
        A float value with the EEG label. if detect_noise is True, this will be a
        LongTensor with the one-hot encoded label including the noisy class.

    Note
    ----
    detect_noise was used during initial test. We decided to leave it there to not
    mess up things
    
    '''

    if detect_noise:
        if nclass == -1:
            raise ValueError('pass the number of classes "nclass" if detect_noise=True')
        if winlen <= 0:
            raise ValueError('pass the window length "winlen" if detect_noise=True')
    
    # NOTE: files were converted in pickle with the 
    # MatlabToPickle Jupyter Notebook. 
    with open(path, 'rb') as eegfile:
        EEG = pickle.load(eegfile)

    # extract and adapt data to training setting
    x = EEG['data']

    # get the dataset ID to coordinate some operations
    data_id = int(path.split(os.sep)[-1].split('_')[0])
    pipe_id = path.split(os.sep)[-2]
    
    # if 125 Hz take one sample every 2
    if "iir" in pipe_id:
        ff = 125
    else:
        ff = 250
        if downsample:
            ff = 125 
            x = x[:,::2]
    
    # if use original, interpolated channels are removed.
    # Check the dataset_info.json in each Summary folder file 
    # to know which channel was interpolated during the preprocessing
    # if detect_noise is true, it will be used to store the l
    if use_only_original or detect_noise:
        if data_id == 10:
            chan2dele = [
                1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 15, 
                16, 17, 18, 19, 21, 23, 24, 26, 27, 29, 30, 
                32, 33, 34, 36, 38, 40, 41, 42, 43, 44, 46, 
                48, 50, 51, 52, 53, 54, 56, 58, 59
            ]
        #elif data_id == 2:
        #    chan2dele = [34,44]
        #elif data_id == 19:
        #    chan2dele = [28, 30]
        else:
            chan2dele = [
                1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 
                23, 27, 29, 30, 32, 34, 36, 38, 40, 42, 44,
                46, 48, 50, 52, 54, 56, 58
            ]
        if use_only_original:
            x = np.delete(x, chan2dele, 0)

    # GetEEGPartitionNumber doesn't want a label, so we need to add a function
    # to omit the label
    if return_label:
        if detect_noise:
            
            # Load Template and remove intrpolated channels if asked
            ch_map = np.load('template.npy')
            if use_only_original:
                ch_int = list(set([i for i in range(61)]).difference(set(chan2dele)))
                ch_map = ch_map[ch_int].tolist()
            else:
                ch_map = ch_map.tolist()

            # change channel name to make them MNE friendly
            for n, i in enumerate(ch_map):
                ch_map[n] = ch_map[n].replace(' ', '').capitalize()
                ch_to_capital = (
                    'Af7', 'Ft7', 'Fc3', 'Tp7', 'Cp3', 'Po7', 'Af8', 'Cp5',
                    'Ft8', 'Fc4', 'Tp8', 'Cp4', 'Po8', 'Af3', 'Fc5', 'Fc1',
                    'Cp1', 'Po3', 'Af4', 'Fc6', 'Fc2', 'Cp6', 'Cp2', 'Po4'
                )
                ch_to_caital_first_2 = ('Poz', 'Cpz', 'Afz', 'Fcz')
                if ch_map[n] in ch_to_capital:
                    ch_map[n] =  ch_map[n].upper()
                if ch_map[n] in ch_to_caital_first_2:
                    ch_map[n] =  ch_map[n][0].upper() + ch_map[n][1].upper() + ch_map[n][2].lower()

            # create mne raw array with template
            xmne = mne.io.RawArray(
                x/1e6,
                mne.create_info(ch_map, ff, ['eeg']*x.shape[0]),
                verbose= False
            )

            # epoch raw data
            epochs = mne.make_fixed_length_epochs(
                xmne,
                duration = winlen,
                reject_by_annotation = False,
                preload = True,
                verbose = False
            )

            # set montage
            epochs.set_montage('standard_1005')

            # create label tensor. This is necessary since Y must have
            # the same number of samples after rejection
            if data_id==10:
                y = EEG['label']
                y = torch.LongTensor([y]*len(epochs))
            else:
                if data_id==2:
                    y = 0.0 # only healthy
                else:
                    y = 1.0 if EEG['label']>0 else 0.0
                y = torch.tensor([y]*len(epochs))

            # apply autoreject
            reject = autoreject.get_rejection_threshold(
                epochs, cv=10, verbose=False, random_state=seed)
            epochs.drop_bad(reject=reject, verbose=False)
            bad_idx = [idx for idx, chlog in enumerate(epochs.drop_log) if chlog != ()]

            # discard labels related to bad windows
            y[bad_idx] = nclass
            if onehot_label:
                y = F.one_hot(y, num_classes = nclass+1)
        else:
            if data_id==10:
                y = EEG['label'] # Alzheimer's dataset is a 3 class
            else:
                if data_id==2:
                    y = 0 # only healthy
                else:
                    y = 1 if EEG['label']>0 else 0 # healthy = 0, PD otherwise
            y = float(y)
            # one hot is needed only if multiclass classification is performed
            if onehot_label and data_id == 10:
                y = F.one_hot(y, num_classes = 3)
        # apply z-score on the channels.
        if apply_zscore:
            x = zscore(x,1)
        return x, y
    else:
        # apply z-score on the channels.
        if apply_zscore:
            x = zscore(x,1)
        return x

def phase_swap(X, Y=None):
    """
    Compute the augmented version of the input with the phase swap algorithm
    """

    # Compute fft, module and phase
    Xfft = torch.fft.fft(X)
    amplitude = Xfft.abs()
    phase = Xfft.angle()
    N = X.shape[0]

    # Augmented X
    X_aug = torch.fft.fft(X).to(device=X.device)
    
    if Y is None:
        # Random shuffle indeces
        idx_shuffle = torch.randperm(N).to(device=X.device)
        idx_shuffle_1 = idx_shuffle[:(N//2)]
        idx_shuffle_2 = idx_shuffle[(N//2):(N//2)*2]

        # Apply phase swap
        X_aug[idx_shuffle_1] = amplitude[idx_shuffle_1]*torch.exp(1j*phase[idx_shuffle_2])
        X_aug[idx_shuffle_2] = amplitude[idx_shuffle_2]*torch.exp(1j*phase[idx_shuffle_1])
    
    else:
        for i in range(2):
            class_idx = torch.arange(N).to(device=X.device)
            class_idx = class_idx[Y==i]
            Ni = len(class_idx)
            idx_shuffle = class_idx[torch.randperm(Ni)]
            idx_shuffle_1 = idx_shuffle[:(Ni//2)]
            idx_shuffle_2 = idx_shuffle[(Ni//2):(Ni//2)*2]
    
            # Apply phase swap
            X_aug[idx_shuffle_1] = amplitude[idx_shuffle_1]*torch.exp(1j*phase[idx_shuffle_2])
            X_aug[idx_shuffle_2] = amplitude[idx_shuffle_2]*torch.exp(1j*phase[idx_shuffle_1]) 

    # Reconstruct the signal
    X_aug = (torch.fft.ifft(X_aug)).real.to(device=X.device)

    return X_aug


def set_augmenter(aug_list, fs=125, winlen=4.0):
    augmenter = None
    if aug_list==None:
        return augmenter

    aug_list = findSeq(aug_list)
    aug_call = []
    for i in aug_list:
        if i == 'flip_horizontal':
            aug_call.append(aug.flip_horizontal)
        elif i == 'flip_vertical':
            aug_call.append(aug.flip_vertical)
        elif i == 'add_band_noise':
            aug_band = aug.DynamicSingleAug(
                aug.add_band_noise,
                discrete_arg = {'samplerate': [fs],
                                'bandwidth': [
                                    'delta', 'theta', 'alpha',
                                    'beta', 'gamma', 'gamma_low'
                                ],
                               },
                range_arg    = {'std': [0.8, 0.9]},
                range_type   = {'std': False}
            )
            aug_call.append(aug_band)
        elif i == 'add_eeg_artifact':
            aug_drift = aug.DynamicSingleAug(
                aug.add_eeg_artifact,
                discrete_arg = {
                    'artifact': ['drift'],
                    'Fs': [fs],
                    'drift_slope': [
                        -0.5/(fs*winlen),
                        -0.4/(fs*winlen),
                        -0.3/(fs*winlen),
                        0.3/(fs*winlen),
                        0.4/(fs*winlen),
                        0.5/(fs*winlen),
                    ]
                }
            )
            aug_call.append(aug_drift)
        elif i == 'add_noise_snr':
            aug_snr = aug.DynamicSingleAug(
                aug.add_noise_SNR,
                range_arg    = {'target_snr': [8, 10]},
                range_type   = {'target_snr': False}
            )
            aug_call.append(aug_snr)
        elif i == 'channel_dropout':
            aug_chan = aug.DynamicSingleAug(
                aug.channel_dropout,
                range_arg    = {'Nchan': [4,16]},
                range_type   = {'Nchan': True}
            )
            aug_call.append(aug_chan)
        elif i == 'masking':
            aug_mask = aug.DynamicSingleAug(
                aug.masking,
                discrete_arg = {'mask_number': [3,4,5]},
                range_arg    = {'masked_ratio': [0.20, 0.35]},
                range_type   = {'masked_ratio': False}
            )
            aug_call.append(aug_mask)
        elif i == 'warp_signal':
            aug_warp = aug.DynamicSingleAug(
                aug.warp_signal,
                discrete_arg = {'segments': [6, 7, 8]},
                range_arg    = {'stretch_strength': [1.25, 1.50]},
                range_type   = {'stretch_strength': False}
            )
            aug_call.append(aug_warp)
        elif i == 'random_FT_phase':
            aug_ft = aug.StaticSingleAug(
                aug.random_FT_phase,
                {'value': 0.9, 'batch_equal': True}
            )
            aug_call.append(aug_ft)
        elif i == 'phase_swap':
            aug_call.append(phase_swap)

    if len(aug_list)<2:
        augmenter1 = aug_call[0]
    else:
        augmenter1 = aug.SequentialAug(*aug_call)
    
    augmenter = aug.RandomAug(aug.identity, augmenter1, p=[0.25, 0.75])
    return augmenter

class TempPsdAugmenter:
    def __init__(self, temp, Taug, Faug=None):
        self.temp = temp
        self.Taug = Taug
        if Faug is None:
            self.Fnoise = aug.DynamicSingleAug(
                aug.masking,
                discrete_arg = {'mask_number': [2,3,4], 'batch_equal': [False]},
                range_arg    = {'masked_ratio': [0.15, 0.25]},
                range_type   = {'masked_ratio': False}
            )
            self.Faug = aug.RandomAug(aug.identity, self.Fnoise, p=[0.15, 0.85])
        else:
            self.Faug = Faug

    def __call__(self, x):
        x[..., :self.temp] = self.Taug(x[..., :self.temp])
        x[..., self.temp:] = self.Faug(x[..., self.temp:])
        return x

def GetLearningRate(model, task):
    lr_dict = GetLrDict()
    if "alzheimer" in task.casefold():
        task = "alzheimer"
    lr = lr_dict.get(model).get(task)
    return lr


def lossBinary(yhat, ytrue):
    '''
    Just an alias to the binary_cross_entropy_with_logits function.
    Remember that yhat must be a tensor with the model output in the logit form,
    so no sigmoid operator should be applied on the model's output. 
    Remember that ytrue must be a float tensor with the same size as yhat and 
    with 0 or 1 based on the binary class.
    '''
    yhat = yhat.flatten()
    return F.binary_cross_entropy_with_logits(yhat, ytrue)


def lossMulti(yhat, ytrue):
    '''
    Just an alias to the binary_cross_entropy_with_logits function.
    Remember that yhat must be a tensor with the model output in the logit form,
    so no sigmoid operator should be applied on the model's output. 
    Remember that ytrue must be a float tensor with the same size as yhat and 
    with 0 or 1 based on the true class (e.g., [[0.,1.,0.], [1.,0.,0.], [0.,0.,1.]])
    Alternatively, it must be a long tensor with the class index (e.g., [1,0,2])
    '''
    return F.cross_entropy(yhat, ytrue)


class WinRatio:

    def __init__(self):
        self.yhat       = np.array([], dtype="float32")
        self.ytrue      = np.array([], dtype="float32")
        self.subj_id    = np.array([], dtype="float32")
        self.ytrue_subj = np.array([], dtype="float32")

    def add_data(self, dataloader, model, th=0.5, device="cpu"):

        # this part will add new subject ids
        try:
            curr_max = np.max(self.subj_id).item()
        except Exception:
            curr_max = 0
        cumulative = dataloader.dataset.EEGcumlen
        for n, win_num in enumerate(cumulative):
            if n==0:
                new_subj = np.repeat(curr_max+1, win_num)
            else:
                new_subj = np.repeat(curr_max+1, cumulative[n]-cumulative[n-1])
            self.subj_id = np.concatenate((self.subj_id, new_subj))
            curr_max += 1

        model.eval()
        model.to(device=device)
        for X, y in dataloader:
            X = X.to(device=device)
            y = y.to(device=device)
            yhat  = model(X).flatten().detach().cpu().numpy() > th
            ytrue = y.flatten().detach().cpu().numpy()
            self.yhat  = np.concatenate((self.yhat, yhat))
            self.ytrue = np.concatenate((self.ytrue, ytrue))

    def _get_info(self):
        print('yhat', self.yhat.shape)
        print('ytrue', self.ytrue.shape)
        print('subj_id', self.subj_id.shape)

    def clear_data(self):
        self.yhat       = np.array([], dtype = "float32")
        self.ytrue      = np.array([], dtype = "float32")
        self.subj_id    = np.array([], dtype = "float32")
        self.ytrue_subj = np.array([], dtype = "float32")

    def compute_ratio(self):
        ratio_tot = []
        for subj_id in np.unique(self.subj_id):
            
            # Extract yhat and ytrue for each subject
            idxs = np.where(self.subj_id == subj_id)[0]
            ytrue = self.ytrue[idxs]

            # calculate the ratio of positive samples, which is equivalent to
            # the probabilty of a subject being unhealthy based on the number of
            # EEG windows predicted to be so
            ratio_subj = ((self.yhat[idxs] == 1.0).sum())/len(ytrue)

            # concatenate
            ratio_tot.append(ratio_subj)

            # add subject label but check if it is unique
            try:
                ytrue = np.unique(ytrue)
                check_y = ytrue.item()
                self.ytrue_subj = np.concatenate((self.ytrue_subj, ytrue))
            except Exception:
                msg = "subject ratio stopped because a subject have multiple labels"
                raise ValueError(msg)

        # Append to the ratio tensor the average across patients
        ratio_tot = np.array(ratio_tot)
        self.ratio = thresh_roc_2(self.ytrue_subj, ratio_tot, window=False)

    def get_ratio(self):
        return self.ratio


def thresh_roc_2(ytrue, ypred, window=True):
    """
    Compute the best threshold that maximize the balanced accuracy
    
    Notes
    -----
    THIS IS COMPUTED ON THE VALIDATION SET AS DOING IT ON THE TEST SETIS EQUIVALENT
    TO DOING DATA LEAKAGE
    """
    # Compute roc curve
    best_th = 0.5
    best_ba = balanced_accuracy_score(ytrue, ypred>0.5)
    range_of_values = np.linspace(0.3,0.7,401) if window else np.linspace(0.001,1,1000)
    for i in range_of_values:
        curr_th = i
        curr_ba = balanced_accuracy_score(ytrue, ypred>curr_th)
        if curr_ba>best_ba:
            best_th = curr_th
            best_ba = curr_ba
    return best_th


def get_performances(
    loader2eval, 
    Model, 
    device         = 'cpu', 
    nb_classes     = 2,
    return_scores  = True,
    verbose        = False,
    plot_confusion = False,
    class_labels   = None,
    roc_correction = False,
    plot_roc       = False,
    th             = 0.5,
    subj_ratio     = None,
):
    '''
    ``get_performances`` calculates numerous metrics to evaluate a Pytorch's
    model. If specified, it also display a summary and plot two confusion matrices.

    Parameters
    ----------
    loader2eval: torch.utils.data.Dataloader
        A Pytorch's Dataloader with the samples to use for the evaluation. 
    Model: torch.nn.Module
        A Pytorch's model to evaluate.
    device: torch.device, optional
        The device to use during batch forward.
        Default = 'cpu'
    nb_classes: int, optional
        The number of classes. Some operations are different between the binary
        and multiclass case.
        Default = 2
    return_scores: dict, optional
        Whether to return all the calculated metrics, predictions, and confusion
        matrices inside a dictionary.
        Default = True
    verbose: bool, optional
        Whether to print all the calculated metrics or not. A scikit-learn's
        classification report is also displayed.
        Default = False
    plot_confusion: bool, optional
        Whether to plot a confusion matrix or not.
        Default = False
    class_labels: list, optional
        A list with the labels to use for the confusion matrix plot. If None,
        values between 0 and the number of classes - 1 will be used.
        Default = None

    Returns
    -------
    scores: dict, optional
        A dictionary with a set of metrics, predictions, and confusion
        matrices calculated inside this function. The full list of values is:
            
            - 'logits': model's activations (logit output) as a numpy array.
            - 'probabilities': model's predicted probabilities as a numpy array.
            - 'predictions': model's predicted classes as a numpy array.
            - 'labels': true labels as a numpy array,
            - 'confusion': confusion matrix with absolute values as a 
              Pandas DataFrame.
            - 'confusion_normalized': normalized confusion matrix with 
              absolute values as a Pandas DataFrame.
            - 'accuracy_unbalanced': unbalanced accuracy,
            - 'accuracy_weighted': weighted accuracy,
            - 'precision_micro': micro precision,
            - 'precision_macro': macro precision,
            - 'precision_weighted': weighted precision,
            - 'precision_matrix': matrix with single class precisions,
            - 'recall_micro': micro recall,
            - 'recall_macro': macro recall,
            - 'recall_weighted': weighted recall,
            - 'recall_matrix': matrix with single calss recalls,
            - 'f1score_micro': micro f1-score,
            - 'f1score_macro': macro f1-score,
            - 'f1score_weighted': weighted f1-score,
            - 'f1score_matrix': matrix with single class f1-scores,
            - 'rocauc_micro': micro ROC AUC,
            - 'rocauc_macro': macro ROC AUC,
            - 'rocauc_weighted': weighted ROC AUC,
            - 'cohen_kappa': Cohen's Kappa score  


    '''
    # calculate logits, probabilities, and classes
    Model.to(device=device)
    Model.eval()

    # correct classification threshold based on ROC curve.
    # This is calculated on the validation set
    if nb_classes <=2 and roc_correction:
        y_true = []
        y_prob = []
        for X,Y in loader2eval:
            with torch.no_grad():
                x_out = Model(X)
                # need to check because extend do not work with 0-dim tensor
                if x_out.numel() == 1:
                    y_prob.append(torch.squeeze(torch.sigmoid(x_out)).cpu().numpy())
                else:
                    y_prob.extend(torch.squeeze(torch.sigmoid(x_out)).cpu().numpy())
                y_true.extend(Y.cpu().numpy())       
        th = thresh_roc_2(np.array(y_true), np.array(y_prob), window=True)

    
    ytrue = torch.zeros(len(loader2eval.dataset))
    ypred = torch.zeros_like(ytrue)
    if nb_classes<=2:
        logit = torch.zeros(len(loader2eval.dataset))
    else:
        logit = torch.zeros(len(loader2eval.dataset), nb_classes)
    proba = torch.zeros_like(logit)
    cnt=0
    for i, (X, Y) in enumerate(loader2eval):
        if isinstance(X, torch.Tensor):
            if X.device.type != device.type:
                X = X.to(device=device)
            Xshape = X.shape[0]
        else:
            for i in range(len(X)):
                if X[i].device.type != device.type:
                    X[i] = X[i].to(device=device)
            Xshape = X[0].shape[0]

        if isinstance(Y, torch.Tensor):
            ytrue[cnt:cnt+Xshape]= Y
        else:
            ytrue[cnt:cnt+Xshape]= Y[0]
        with torch.no_grad():
            yhat = Model(X)
            if isinstance(yhat, torch.Tensor):
                yhat = yhat.to(device='cpu')
            else:
                yhat = yhat[0].to(device='cpu')
                
            if nb_classes == 2:
                logit[cnt:cnt+Xshape] = torch.squeeze(yhat)
                yhat = torch.sigmoid(yhat)
                yhat = torch.squeeze(yhat)
                proba[cnt:cnt+Xshape] = yhat
                ypred[cnt:cnt+Xshape] = yhat > th 
            else:
                logit[cnt:cnt+Xshape] = yhat
                yhat = torch.softmax(yhat, 1)
                proba[cnt:cnt+Xshape] = yhat
                yhat = torch.argmax(yhat, 1)
                ypred[cnt:cnt+Xshape] = torch.squeeze(yhat) 
        cnt += Xshape

    # convert to numpy for score computation
    proba = proba.numpy()
    logit = logit.numpy()
    ytrue = ytrue.numpy()
    ypred = ypred.numpy()

    # If the win ratio is not None compute the subject-wise prediction
    if subj_ratio is not None:
        data_collector = WinRatio()
        data_collector.add_data(loader2eval, Model, th=th, device=device)
        subj_id = data_collector.subj_id
        single_id = np.unique(subj_id)
        Nsubj = len(single_id)
        ytrue_new = np.zeros(Nsubj)
        ypred_new = np.zeros(Nsubj)

        # Cycle over subjects
        for i in range(Nsubj):
            # the np.unique perform a double check on the uniqueness of the label
            ytrue_new[i] = np.unique(data_collector.ytrue[subj_id==single_id[i]]).item()
            ypred_subj   = data_collector.yhat[subj_id==single_id[i]]
            ypred_new[i] = (np.sum(ypred_subj)/ypred_subj.size) > subj_ratio

        # Overwrite the ytrue and ypred with the new ones
        ytrue = ytrue_new
        ypred = ypred_new

    if verbose:
        print("Evaluating the performance on a set of ", ytrue.shape, " samples/subjects")
    # confusion matrices
    labels1 = [i for i in range(nb_classes)]
    if (class_labels is not None) and (len(class_labels)==nb_classes):
        index1  = class_labels
    else:
        index1 = [str(i) for i in range(nb_classes)]
    ConfMat = confusion_matrix(ytrue, ypred, labels=labels1).T
    ConfMat_df = pd.DataFrame(ConfMat, index = index1, columns = index1)
    Acc_mat = confusion_matrix(ytrue, ypred, labels=labels1, normalize='true').T
    Acc_mat_df = pd.DataFrame(Acc_mat, index = index1, columns = index1)

    # accuracy, precision, recall, f1, roc_auc, cohen's kappa
    acc_unbal = accuracy_score(ytrue, ypred)
    acc_weigh = balanced_accuracy_score(ytrue, ypred)
    
    f1_mat = f1_score(ytrue, ypred, average = None, zero_division = 0.0)
    f1_micro = f1_score(ytrue, ypred, average = 'micro', zero_division = 0.0)
    f1_macro = f1_score(ytrue, ypred, average = 'macro', zero_division = 0.0)
    f1_weigh = f1_score(ytrue, ypred, average = 'weighted', zero_division = 0.0)
    
    prec_mat = precision_score(ytrue, ypred, average = None, zero_division=0.0)
    prec_micro = precision_score(ytrue, ypred, average = 'micro', zero_division = 0.0)
    prec_macro = precision_score(ytrue, ypred, average = 'macro', zero_division = 0.0)
    prec_weigh = precision_score(ytrue, ypred, average = 'weighted',zero_division = 0.0)
    
    recall_mat = recall_score(ytrue, ypred, average = None, zero_division=0.0)
    recall_micro = recall_score(ytrue, ypred, average = 'micro', zero_division = 0.0)
    recall_macro = recall_score(ytrue, ypred, average = 'macro', zero_division = 0.0)
    recall_weigh = recall_score(ytrue, ypred, average = 'weighted', zero_division = 0.0)
    
    cohen_kappa = cohen_kappa_score(ytrue, ypred)

    try:
        if nb_classes == 2:
            roc_micro = roc_auc_score(ytrue, proba, average = 'micro', multi_class = 'ovo')
        else:
            roc_micro = np.nan
        roc_macro = roc_auc_score(ytrue, proba, average = 'macro', multi_class = 'ovr')
        roc_weigh = roc_auc_score(ytrue, proba, average = 'weighted', multi_class = 'ovr')
    except Exception:
        roc_micro = np.nan
        roc_macro = np.nan
        roc_weigh = np.nan

    # print everything plus a classification report if asked
    if verbose:
        print('           |-----------------------------------------|')
        print('           |                SCORE SUMMARY            |')
        print('           |-----------------------------------------|')
        print('           |  Accuracy score:                 %.3f  |' %acc_unbal) 
        print('           |  Accuracy score weighted:        %.3f  |' %acc_weigh) 
        print('           |-----------------------------------------|')
        print('           |  Precision score micro:          %.3f  |' %prec_micro)
        print('           |  Precision score macro:          %.3f  |' %prec_macro)
        print('           |  Precision score weighted:       %.3f  |' %prec_weigh)
        print('           |-----------------------------------------|')
        print('           |  Recall score micro:             %.3f  |' %recall_micro)
        print('           |  Recall score macro:             %.3f  |' %recall_macro)
        print('           |  Recall score weighted:          %.3f  |' %recall_weigh)
        print('           |-----------------------------------------|')
        print('           |  F1-score micro:                 %.3f  |' %f1_micro)
        print('           |  F1-score macro:                 %.3f  |' %f1_macro)
        print('           |  F1-score weighted:              %.3f  |' %f1_weigh)
        print('           |-----------------------------------------|')
        #print('           |  ROC AUC micro:                  %.3f  |' %roc_micro)
        #print('           |  ROC AUC macro:                  %.3f  |' %roc_macro)
        #print('           |  ROC AUC weighted:               %.3f  |' %roc_weigh)
        #print('           |-----------------------------------------|')
        print('           |  Cohen\'s kappa score:            %.3f  |' %cohen_kappa)
        print('           |-----------------------------------------|')

        print(' ')
        print(classification_report(ytrue,ypred, zero_division=0))
        print(' ')

    # plot a confusion matrix if asked
    if plot_confusion:
        const_size = 30
        vmin = np.min(ConfMat)
        vmax = np.max(ConfMat)
        off_diag_mask = np.eye(*ConfMat.shape, dtype=bool)
        
        plt.figure(figsize=(14,6),layout="constrained")
        sns.set(font_scale=1.5)
        plt.subplot(1,2,1)
        sns.heatmap(ConfMat_df, vmin= 0, vmax=vmax, mask=~off_diag_mask, fmt="4d",
                    annot=True, cmap='Blues', linewidths=1, cbar_kws={'pad': 0.01},
                    annot_kws={"size": const_size / np.sqrt(len(ConfMat_df))})
        sns.heatmap(ConfMat_df, annot=True, mask=off_diag_mask, cmap='OrRd', 
                    vmin=vmin, vmax=vmax, linewidths=1, fmt="4d",
                    cbar_kws={'ticks':[], 'pad': 0.05},
                    annot_kws={"size": const_size / np.sqrt(len(ConfMat_df))})
        plt.xlabel('true labels', fontsize=20)
        plt.ylabel('predicted labels', fontsize=20)
        plt.title('Confusion Matrix', fontsize=25)
        
        sns.set(font_scale=1.5)
        plt.subplot(1,2,2)
        sns.heatmap(Acc_mat_df, vmin= -0.01, vmax=1.01, mask=~off_diag_mask, 
                    fmt=".3f", cbar_kws={'pad': 0.01},
                    annot=True, cmap='Blues', linewidths=1)
        sns.heatmap(Acc_mat_df, annot=True, mask=off_diag_mask, 
                    cmap='OrRd', fmt=".3f",
                    cbar_kws={'ticks':[], 'pad': 0.05},
                    vmin=-0.01, vmax=1.01, linewidths=1)
        plt.xlabel('true labels', fontsize=20)
        plt.ylabel('predicted labels', fontsize=20)
        plt.title('Normalized Confusion Matrix', fontsize=25)
        plt.show()

    if return_scores:
        scores = {
            'logits': logit,
            'probabilities': proba,
            'predictions': ypred,
            'labels': ytrue,
            'confusion': ConfMat_df,
            'confusion_normalized': Acc_mat_df,
            'accuracy_unbalanced': acc_unbal,
            'accuracy_weighted': acc_weigh,
            'precision_micro': prec_micro,
            'precision_macro': prec_macro,
            'precision_weighted': prec_weigh,
            'precision_matrix': prec_mat,
            'recall_micro': recall_micro,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weigh,
            'recall_matrix': recall_mat,
            'f1score_micro': f1_micro,
            'f1score_macro': f1_macro,
            'f1score_weighted': f1_weigh,
            'f1score_matrix': f1_mat,
            'rocauc_micro': roc_micro,
            'rocauc_macro': roc_macro,
            'rocauc_weighted': roc_weigh,
            'cohen_kappa': cohen_kappa    
        }
        if roc_correction:
            scores['best_th'] = th
        return scores
    
    elif roc_correction:
        scores = {'best_th' : th}
        return scores
        
    else:
        return