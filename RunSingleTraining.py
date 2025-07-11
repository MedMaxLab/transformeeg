# This code runs a single training of a 10-outer 10-inner
# Nested-Leave-N-Subjects-Out cross validation using only
# two of the 4 selected datasets

# set save subfolder
flag_dir = "two_dataset_baseline/"

# ===========================
#  Section 1: package import
# ===========================
# This section includes all the packages to import. 
# To run this notebook, you must install in your environment. 
# They are: numpy, pandas, matplotlib, scipy, scikit-learn, pytorch, selfeeg

import argparse
import glob
import os
import random
import pickle
import copy
import warnings
warnings.filterwarnings(
    "ignore", message = "Using padding='same'", category = UserWarning
)

# IMPORT STANDARD PACKAGES
from mne.time_frequency import psd_array_multitaper
import numpy as np
import pandas as pd
from scipy import signal

# IMPORT TORCH
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# IMPORT SELFEEG 
import selfeeg
import selfeeg.models as zoo
import selfeeg.dataloading as dl
import selfeeg.augmentation as aug
from selfeeg.ssl import fine_tune as train_model

# IMPORT REPOSITORY FUNCTIONS
import AllFnc
from AllFnc import split
from AllFnc.models import (
    XEEGNet,
    TransformEEG,
    EEGResNet18,
    Conformer,
    PSDNetFinal,
)
from AllFnc.training import (
    loadEEG,
    lossBinary,
    lossMulti,
    get_performances,
    GetLearningRate,
    set_augmenter,
    TempPsdAugmenter,
    phase_swap,
    WinRatio,
)
from AllFnc.utilities import (
    restricted_float,
    positive_float,
    positive_int_nozero,
    positive_int,
    str2bool,
    str2list,
    CSPScaler,
    get_aug_idx
)

import warnings
warnings.filterwarnings(
    "ignore",
    message= "numpy.core.numeric is deprecated",
    category=DeprecationWarning
)

def _reset_seed_number(seed):
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    # ===========================
    #  Section 2: set parameters
    # ===========================
    # In this section all tunable parameters are instantiated. The entire training 
    # pipeline is configured here, from the task definition to the model evaluation.
    # Other code cells compute their operations using the given configuration. 
    
    help_d = """
    RunSingleTraining runs a single training of a 10-outer 10-inner
    Nested-Leave-N-Subjects-Out cross validation using only
    two of the 4 selected datasets. Many parameters 
    can be set, which will be then used to create a custom 
    file name. The only one required is the root dataset path.
    Others have a default in case you want to check a single demo run.
    This code is usually called by RunCV.py
    """
    parser = argparse.ArgumentParser(description=help_d)
    parser.add_argument(
        "-D",
        "--datapath",
        dest      = "dataPath",
        metavar   = "datasets path",
        type      = str,
        nargs     = 1,
        required  = True,
        help      = """
        The dataset path.
        dataPath must point to a directory which contains four subdirecotries, one with 
        all the pickle files containing EEGs preprocessed with a specific pipeline.
        Subdirectoties are expected to have the following names, which are the same as
        the preprocessing pipelinea to evaluate: 1) raw; 2) filt; 3) ica; 4) icasr
        The name of the subfolder is passed with the pipeline args
        """,
    )
    parser.add_argument(
        "-p",
        "--pipeline",
        dest      = "pipelineToEval",
        metavar   = "preprocessing pipeline",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'ica',
        choices   =['ica'],
        help      = """
        The preprocessing pipeline to consider.
        It can only be "ica" for this experiment.
        """,
    )
    parser.add_argument(
        "-t",
        "--task",
        dest      = "taskToEval",
        metavar   = "task",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'eyes',
        choices   =['parkinson'],
        help      = """
        The task to evaluate.
        It can be only "parkinson" for this experiment.
        """,
    )
    parser.add_argument(
        "-m",
        "--model",
        dest      = "modelToEval",
        metavar   = "model",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'shallownet',
        choices   =['eegnet',  'deepconvnet',   'shallownet',
                    'resnet',       'atcnet', 'transformeeg',
                    'xeegnet',     'psdnet3', 'eegconformer',
                   ],
        help      = """
        The model to evaluate. It can be one of the following:
        1) eegnet; 2) shallownet; 3) deepconvnet; 4) resnet; 
        5) atcnet; 6) psdnet3; 7) transformeeg; 8) eegconformer; 9) xeegnet
        """,
    )
    parser.add_argument(
        "-o",
        "--outer",
        dest      = "outerFold",
        metavar   = "outer fold",
        type      = int,
        nargs     = '?',
        required  = False,
        default   = 1,
        choices   = range(1,11),
        help      = 'The outer fold to evaluate. It can be a number between 1 and 10'
    )
    parser.add_argument(
        "-i",
        "--inner",
        dest      = "innerFold",
        metavar   = "inner fold",
        type      = int,
        nargs     = '?',
        required  = False,
        default   = 1,
        choices   = range(1,11),
        help      = 'The inner fold to evaluate. It can be a number between 1 and 10'
    )
    parser.add_argument(
        "-d",
        "--downsample",
        dest      = "downsample",
        metavar   = "downsample",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = True,
        help      = """
        A boolean that set if downsampling at 125 Hz should be applied or not.
        The presented analysis uses 250 Hz, which is 5.55 times the maximum investigated 
        frequency (45 Hz). Models usually perform better with 125 Hz. 
        """
    )
    parser.add_argument(
        "-z",
        "--zscore",
        dest      = "z_score", 
        metavar   = "zscore",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = True,
        help      = """
        A boolean that set if the z-score should be applied or not. 
        """
    )
    parser.add_argument(
        "-r",
        "--reminterp",
        dest      = "rem_interp",
        metavar   = "remove interpolated",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = False,
        help      = """
        A boolean that set if the interpolated channels should be 
        removed or not.
        Data were preprocessed with BIDSAlign, a library that aligns all EEGs
        to a common 61 channel template based on the 10_10 International System
        with spherical interpolation.
        """
    )
    parser.add_argument(
        "-b",
        "--batch",
        dest      = "batchsize",
        metavar   = "batch size",
        type      = positive_int_nozero,
        nargs     = '?',
        required  = False,
        default   = 64,
        help      = """
        Define the Batch size. It is suggested to use 64 or 128.
        The experimental analysis was performed on batch 64.
        """
    )
    parser.add_argument(
        "-O",
        "--overlap",
        dest      = "overlap",
        metavar   = "windows overlap",
        type      = restricted_float,
        nargs     = '?',
        required  = False,
        default   = 0.25,
        help      = """
        The overlap between time windows. Higher values means more samples 
        but higher correlation between them. 0.25 is a good trade-off.
        Must be a value in [0,1)
        """
    )
    parser.add_argument(
        "-l",
        "--learningrate",
        dest      = "lr",
        metavar   = "learning rate",
        type      = positive_float,
        nargs     = '?',
        required  = False,
        default   = 2.5e-5,
        help      = "The learning rate. Must be a positive value"
    )
    parser.add_argument(
        "-a",
        "--adamdecay",
        dest      = "adam",
        metavar   = "adam weight decay",
        type      = positive_float,
        nargs     = '?',
        required  = False,
        default   = 0.0,
        help      = "The weight decay to use in Adam Optimizer"
    )
    parser.add_argument(
        "-w",
        "--window",
        dest      = "window",
        metavar   = "window",
        type      = positive_float,
        nargs     = '?',
        required  = False,
        default   = 16.0,
        help      = """
        The window (input) size, in seconds. Each EEG will be partitioned in
        windows of length equals to the one specified by this input.
        """
    )
    parser.add_argument(
        "-c",
        "--csp",
        dest      = "csp",
        metavar   = "common spatial pattern scaler",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = False,
        help      = """
        A boolean that set if EEG data should be transformed
        with the common spatial pattern.
        """
    )
    parser.add_argument(
        "-f",
        "--cspfilt",
        dest      = "filters",
        metavar   = "csp filters",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 10,
        help      = """
        The number of filters to preserve after CSP is fitted.
        Remember that the channel dimensions will be reduced to filters*2.
        """
    )
    parser.add_argument(
        "-A",
        "--aug",
        dest      = "augmentation",
        metavar   = "Augmentation list",
        type      = str2list,
        nargs     = '?',
        required  = False,
        default   = None,
        help      = """
        A list identifying the combination of data augmentations.
        Augmentations will be applied following a 85/15 rule, in the order given
        in the list. Possible augmentation that can be used come from selfEEG
        augmentation module:
            add_band_noise, add_eeg_artifacts, add_noise_SNR, channel_dropout,
            bandpass_filter, flip_horizontal, flip_vertical, masking, warp_signal,
            phase_swap
        """,
    )
    parser.add_argument(
        "-W",
        "--workers",
        dest      = "workers",
        metavar   = "dataloader workers",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 0,
        help      = """
        The number of workers to set for the dataloader. Datasets are preloaded
        for faster computation, so 0 is more than enough.
        """
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest      = "verbose",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = False,
        help      = """
        Set the verbosity level of the whole script. If True, information about
        the choosen split, and the training progression will be displayed
        """
    )
    parser.add_argument(
        "-g",
        "--gpu",
        dest      = "gpu",
        metavar   = "gpu device",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'cpu',
        help      = "A string specifing the torch device to use. Default is cpu",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest      = "seed",
        metavar   = "seed",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 42,
        help      = "The random seed to use"
    )
    args = vars(parser.parse_args())
    
    if args['verbose']:
        print('running training with the following parameters:')
        print(' ')
        for key in args:
            if key == 'dataPath':
                print( f"{key:15} ==> {args[key][0]:<15}") 
            elif key == "augmentation":
                print(f"{key:15} ==> ", args[key])
            else:
                print( f"{key:15} ==> {args[key]:<15}") 
    
    dataPath       = args['dataPath'][0]
    pipelineToEval = args['pipelineToEval']
    taskToEval     = args['taskToEval'].casefold() 
    modelToEval    = args['modelToEval'].casefold() 
    outerFold      = args['outerFold'] - 1
    innerFold      = args['innerFold'] - 1
    downsample     = args['downsample']
    z_score        = args['z_score']
    rem_interp     = args['rem_interp']
    batchsize      = args['batchsize']
    overlap        = args['overlap']
    workers        = args['workers']
    window         = args['window']
    verbose        = args['verbose']
    lr             = args['lr']
    weight_decay   = args['adam']
    csp            = args['csp']
    Nfilters       = args['filters']
    augment_list   = args['augmentation']
    device         = args['gpu'].casefold() 
    seed           = args['seed']

    # Force as much determinism as possible for transformeeg
    torch.use_deterministic_algorithms(
        True, warn_only=False if modelToEval=='transformeeg' else True
    )
    if modelToEval=='transformeeg':
        torch.backends.cudnn.deterministic = True

    # Define the device to use
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # fold to eval is the correct index to get the desired train/val/test partition
    foldToEval = outerFold*10 + innerFold
    
    # ==================================
    #  Section 3: create partition list
    # ==================================
    #ds003490 - ID 5 - 3Stim
    ctl_id_5 = [i for i in range(28,51)] + [3,5]
    pds_id_5 = [i for i in range(6,28)] + [1,2,4]
    part_c = split.create_nested_kfold_subject_split(ctl_id_5, 10, 10, seed)
    part_p = split.create_nested_kfold_subject_split(pds_id_5, 10, 10, seed)
    partition_list_1 = split.merge_partition_lists(part_c, part_p, 10, 10)

    #ds002778 - ID 8 - UCSD
    ctl_id_8 = [1, 2, 4, 7,  8, 10, 17, 19, 20, 23, 24, 27, 28, 29, 30, 31]
    pds_id_8 = [3, 5, 6, 9, 11, 12, 13, 14, 15, 16, 18, 21, 22, 25, 26]
    if pipelineToEval == 'ica':
        ctl_id_8 = [i for i in range(1, 17)]
        pds_id_8 = [i for i in range(17,32)]
    part_c = split.create_nested_kfold_subject_split(ctl_id_8, 10, 10, seed)
    part_p = split.create_nested_kfold_subject_split(pds_id_8, 10, 10, seed)
    partition_list_2 = split.merge_partition_lists(part_c, part_p, 10, 10)
           
    # ======================================
    # Section 4: set the training parameters
    # =====================================
    if dataPath[-1] != os.sep:
        dataPath += os.sep
    if pipelineToEval[-1] != os.sep:
        eegpath = dataPath + pipelineToEval + os.sep
    else:
        eegpath = dataPath + pipelineToEval
    
    # Define the number of Channels to use. 
    if rem_interp:
        Chan = 32
    else:
        Chan = 61
    
    freq = 125 if downsample else 250
    
    # Define the number of classes to predict.
    nb_classes = 2
    
    # For selfEEG's models instantiation
    Samples = int(freq*window)
    
    # Set the Dataset ID for glob.glob operation in SelfEEG's GetEEGPartitionNumber().
    # It is a single number for every dataset
    datasetID_1 = '5'  # EEG 3-Stim
    datasetID_2 = '8'  # UC SD

    # Set the class label in case of plot of functions
    class_labels = ['CTL', 'PD']
    
    
    # =====================================================
    #  Section 5: Define pytorch's Datasets and dataloaders
    # =====================================================
    
    # GetEEGPartitionNumber doesn't need the labels
    loadEEG_args = {
        'return_label': False, 
        'downsample': downsample, 
        'use_only_original': rem_interp,
        'apply_zscore': z_score
    }
    
    glob_input = [
        datasetID_1 + '_*.pickle',   # only off medication
        datasetID_2 + '_*.pickle',   # only off medication
    ] 
    
    # calculate dataset length.
    # Basically it automatically retrieves all the partitions 
    # that can be extracted from each EEG signal
    EEGlen = dl.get_eeg_partition_number(
        eegpath, freq, window, overlap, 
        file_format             = glob_input,
        load_function           = loadEEG,
        optional_load_fun_args  = loadEEG_args,
        includePartial          = False if overlap == 0 else True,
        verbose                 = verbose
    )
    
    # Now we also need to load the labels
    loadEEG_args['return_label'] = True
    
    # Set functions to retrieve dataset, subject, and session from each filename.
    # They will be used by GetEEGSplitTable to perform a subject based split
    dataset_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[0])
    subject_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[1]) 
    session_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[2]) 
    
    # Now call the GetEEGSplitTable. Since Parkinson task merges two datasets
    # we need to differentiate between this and other tasks
    # Remember: 5 = 3-Stim   &&   8 = UCSD
    train_id   = { 
        5:  partition_list_1[foldToEval][0], 
        8:  partition_list_2[foldToEval][0],
    }
    val_id     = {
        5:  partition_list_1[foldToEval][1], 
        8:  partition_list_2[foldToEval][1],
    }
    test_id    = {
        5:  partition_list_1[foldToEval][2], 
        8:  partition_list_2[foldToEval][2],
    }
    EEGsplit= dl.get_eeg_split_table(
        partition_table      = EEGlen,
        exclude_data_id      = None,
        val_data_id          = val_id,
        test_data_id         = test_id, 
        split_tolerance      = 0.001,
        dataset_id_extractor = dataset_id_ex,
        subject_id_extractor = subject_id_ex,
        perseverance         = 10000
    )
    
    if verbose:
        print(' ')
        print('Subjects used for test')
        print(test_id)
    
    # Define Datasets and preload all data
    trainset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'train', 
        supervised             = True, 
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args
    )
    trainset.preload_dataset()

    if csp:
        flag_dir = "csp/"
        _reset_seed_number(seed)
        CSP = CSPScaler(Nfilters = Nfilters, device = device)
        data1 = trainset.x_preload[trainset.y_preload==0].detach().clone().numpy()
        data2 = trainset.x_preload[trainset.y_preload==1].detach().clone().numpy()
        CSP.fit(data1 , data2)
        del data1, data2
        Chan = Nfilters*2
        CSPval = copy.deepcopy(CSP)
        CSPval._use_torch = False
        CSPval.Wcsp = CSPval.Wcsp.detach().cpu().numpy()
    
    valset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'validation',
        supervised             = True, 
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args,
        transform_function     = CSPval if csp else None,
    )
    valset.preload_dataset()
    
    testset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'test',
        supervised             = True,
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args,
        transform_function     = CSPval if csp else None,
    )
    testset.preload_dataset()
    
    if 'psdnet' in modelToEval:
        kwargs_multitaper = {
            'sfreq':         freq,
            'fmin':          0.5,
            'fmax':          45.25,
            'n_jobs':        16,
            'bandwidth':     2.5,
            'normalization': 'full',
        }
        Pxxtr, f = psd_array_multitaper( trainset.x_preload, **kwargs_multitaper)
        Pxxva, f = psd_array_multitaper(   valset.x_preload, **kwargs_multitaper)
        Pxxte, f = psd_array_multitaper(  testset.x_preload, **kwargs_multitaper)
        
        Pxxtr = torch.from_numpy(Pxxtr).to(dtype = torch.float32, device=device)
        Pxxva = torch.from_numpy(Pxxva).to(dtype = torch.float32, device=device)
        Pxxte = torch.from_numpy(Pxxte).to(dtype = torch.float32, device=device)
    
        trainset.x_preload = trainset.x_preload.to(device=device)
        trainset.y_preload = trainset.y_preload.to(device=device)
        valset.x_preload = valset.x_preload.to(device=device)
        valset.y_preload = valset.y_preload.to(device=device)
        testset.x_preload = testset.x_preload.to(device=device)
        testset.y_preload = testset.y_preload.to(device=device)

        timelen = trainset.x_preload.shape[-1]
        freqlen = Pxxtr.shape[-1]

        trainset.x_preload = torch.cat((trainset.x_preload, Pxxtr), -1 )
        valset.x_preload   = torch.cat((valset.x_preload,   Pxxva), -1 )
        testset.x_preload = torch.cat((testset.x_preload, Pxxte), -1 )
    else:
        
        trainset.x_preload = trainset.x_preload.to(device=device)
        trainset.y_preload = trainset.y_preload.to(device=device)
        valset.x_preload = valset.x_preload.to(device=device)
        valset.y_preload = valset.y_preload.to(device=device)
        testset.x_preload = testset.x_preload.to(device=device)
        testset.y_preload = testset.y_preload.to(device=device)
    
    # Finally, Define Dataloaders
    # (no need to use more workers in validation and test dataloaders)
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = batchsize,
        shuffle     = True,
        num_workers = workers
    )
    valloader = DataLoader(
        dataset     = valset,
        batch_size  = batchsize,
        shuffle     = False,
        num_workers = 0
    )
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = batchsize,
        shuffle     = False,
        num_workers = 0
    )
    
    # ===================================================
    #  Section 6: define the loss, model, and optimizer
    # ==================================================
    
    lossVal = None
    validation_loss_args = []
    lossFnc = lossBinary

    # Set data augmentation
    if augment_list is None:
        if csp:
            augmenter = CSP
        else:
            augmenter = None
    else:
        augmenter = set_augmenter(augment_list, fs=freq, winlen=window)
        augidx1 = get_aug_idx(augment_list[0])
        augidx2 = get_aug_idx(augment_list[1])
        if csp:
            augmenter1 = set_augmenter(augment_list, fs=freq, winlen=window)
            augmenter = aug.SequentialAug(augmenter1, CSP)
        if 'psdnet' in modelToEval:
            augmenter = TempPsdAugmenter(timelen, augmenter)

    
    # SET SEEDS FOR REPRODUCIBILITY
    _reset_seed_number(seed)
    
    # define model
    if modelToEval.casefold() == 'eegnet':
        Mdl = zoo.EEGNet(
            nb_classes, Chan, Samples,
            depthwise_max_norm    = None,
            norm_rate             = None
        )
    elif modelToEval.casefold() == 'shallownet':
        Mdl = zoo.ShallowNet(nb_classes, Chan, Samples)
    elif modelToEval.casefold() == 'xeegnet':
        xeegnet_custom_dict = {
            "F1":                     7,
            "K1":                     125,
            "F2":                     7,
            "Pool":                   75,
            "p":                      0.2,
            "log_activation_base":    "dB",
            "norm_type":              "batchnorm",
            "random_temporal_filter": False,
            "Fs":                     125 if downsample else 250 ,
            "freeze_temporal":        999999999,
            "dense_hidden":           None,
            "spatial_depthwise":      True,
            "spatial_only_positive":  False,
            "global_pooling":         True,
            "bias":                   [False, False, False],
            "return_logits":          True,
            "seed":                   seed
        }
        Mdl = XEEGNet(nb_classes, Chan, Samples, **xeegnet_custom_dict)
    elif modelToEval == 'deepconvnet':
        Mdl = zoo.DeepConvNet(
            nb_classes, Chan, Samples,
            kernLength      = 10,
            F               = 25,
            Pool            = 3,
            stride          = 3,
            batch_momentum  = 0.1,
            dropRate        = 0.5,
            max_norm        = None,
            max_dense_norm  = None
        )
    elif modelToEval == 'resnet':
        Mdl = EEGResNet18(nb_classes)
    elif modelToEval == 'eegconformer':
        Mdl = Conformer(40, depth=6, n_classes=nb_classes, chan=Chan, seed=seed)
    elif modelToEval == 'transformeeg':
        # Number of features is quadrupled due to double depthwise conv1d with D=2
        Mdl = TransformEEG(nb_classes, Chan, Chan*4, seed)
    elif modelToEval == 'atcnet':
        Mdl = zoo.ATCNet(nb_classes, Chan, Samples, freq)
    elif modelToEval == 'psdnet3':
        Mdl = PSDNetFinal(nb_classes, Chan, 128, timelen, freqlen, seed=seed)
    
    MdlBase = copy.deepcopy(Mdl)
    Mdl.to(device = device)
    Mdl.train()
    if verbose:
        print(' ')
        ParamTab = selfeeg.utils.count_parameters(Mdl, False, True, True)
        print(' ')
    
    if lr == 0:
        lr = GetLearningRate(modelToEval, taskToEval)
        if verbose:
            print(' ')
            print('used learning rate', lr)
    gamma = 0.99
    optimizer = torch.optim.Adam(
        Mdl.parameters(),
        betas = (0.75, 0.999),
        lr = lr,
        weight_decay = weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
    
    # Define selfEEG's EarlyStopper with large patience to act as a model checkpoint
    earlystop = selfeeg.ssl.EarlyStopping(
        patience  = 20,
        min_delta = 1e-04, 
        record_best_weights = True
    )
    
    # =============================
    #  Section 7: train the model
    # =============================
    _reset_seed_number(seed)
    loss_summary = train_model(
        model                 = Mdl,
        train_dataloader      = trainloader,
        epochs                = 300,
        optimizer             = optimizer,
        loss_func             = lossFnc,
        augmenter             = augmenter,
        lr_scheduler          = scheduler,
        EarlyStopper          = earlystop,
        validation_dataloader = valloader,
        validation_loss_func  = lossVal,
        validation_loss_args  = validation_loss_args,
        verbose               = verbose,
        device                = device,
        return_loss_info      = True
    )
    
    # ===============================
    #  Section 8: evaluate the model
    # ===============================
    scores = {}
    scores['loss_progression'] = loss_summary
    earlystop.restore_best_weights(Mdl)
    Mdl.to(device=device)
    Mdl.eval()

    # Evaluate model on the window level with the standard 0.5 threshold
    scores['th_standard'] = get_performances(
        loader2eval    = testloader, 
        Model          = Mdl, 
        device         = device,
        nb_classes     = nb_classes,
        return_scores  = True,
        verbose        = verbose,
        plot_confusion = False,
        class_labels   = class_labels,
        roc_correction = False,
        plot_roc       = False,
        th             = 0.5,
        subj_ratio      = None,
    )
    if not verbose:
        bal_acc = scores['th_standard']['accuracy_weighted']
        print(f'Balanced accuracy on windows with threshold 0.500 --> {bal_acc:.3f}')

    # Evalutate the model on the window level with a roc corrected threshold
    th_eval = get_performances(
        loader2eval    = valloader,
        Model          = Mdl,
        device         = device,
        nb_classes     = nb_classes,
        class_labels   = class_labels,
        verbose        = False,
        return_scores  = False,
        plot_confusion = False,
        plot_roc       = False,
        roc_correction = True
    )
    th = th_eval['best_th']
    scores['window_threshold'] = th
    scores['th_corrected'] = get_performances(
        loader2eval    = testloader, 
        Model          = Mdl, 
        device         = device,
        nb_classes     = nb_classes,
        return_scores  = True,
        verbose        = verbose,
        plot_confusion = False,
        class_labels   = class_labels,
        roc_correction = False,
        plot_roc       = False,
        th             = th,
        subj_ratio      = None,
    )
    if not verbose:
        bal_acc_roc = scores['th_corrected']['accuracy_weighted']
        print(f'Balanced accuracy on windows with threshold {th:.3f} --> {bal_acc_roc:.3f}')

    # Evalutate the model on the subject level
    subject_labeler = WinRatio()
    subject_labeler.add_data(valloader, Mdl, device='cpu', th = 0.5)
    subject_labeler.compute_ratio()
    subject_thresh = subject_labeler.get_ratio()
    scores['subject_threshold_05'] = subject_thresh
    scores['subject'] = get_performances(
        loader2eval    = testloader, 
        Model          = Mdl, 
        device         = device,
        nb_classes     = nb_classes,
        return_scores  = True,
        verbose        = verbose,
        plot_confusion = False,
        class_labels   = class_labels,
        roc_correction = False,
        plot_roc       = False,
        th             = 0.5,
        subj_ratio      = subject_thresh,
    )
    if not verbose:
        bal_acc_sub = scores['subject']['accuracy_weighted']
        print("using the following threshold for subject predictions", subject_thresh)
        print(f'Balanced accuracy on subject with threshold 0.500 --> {bal_acc_sub:.3f}')

    subject_labeler = WinRatio()
    subject_labeler.add_data(valloader, Mdl, device='cpu', th = th)
    subject_labeler.compute_ratio()
    subject_thresh = subject_labeler.get_ratio()
    scores['subject_threshold_th'] = subject_thresh
    scores['subject_corrected'] = get_performances(
        loader2eval    = testloader, 
        Model          = Mdl, 
        device         = device,
        nb_classes     = nb_classes,
        return_scores  = True,
        verbose        = verbose,
        plot_confusion = False,
        class_labels   = class_labels,
        roc_correction = False,
        plot_roc       = False,
        th             = th,
        subj_ratio      = subject_thresh,
    )
    if not verbose:
        bal_acc_sub_roc = scores['subject_corrected']['accuracy_weighted']
        print("using the following threshold for subject predictions", subject_thresh)
        print(f'Balanced accuracy on subject with roc & th  {th:.3f} --> {bal_acc_sub_roc:.3f}')
    
    # ==================================
    #  Section 9: Save model and metrics
    # ==================================

    # we will create a custom name summarizing
    # all the important parameters using for this training
    start_piece_mdl = 'PDClassification/Models/'
    start_piece_res = 'PDClassification/Results/'

    # For extra analyses.
    if flag_dir is not None:
        start_piece_mdl += flag_dir
        start_piece_res += flag_dir

    task_piece = 'pds'
    
    if modelToEval.casefold() == 'eegnet':
        mdl_piece = 'egn'
    elif modelToEval.casefold() == 'shallownet':
        mdl_piece = 'shn'
    elif modelToEval.casefold() == 'xeegnet':
        mdl_piece = 'xeg'
    elif modelToEval.casefold() == 'deepconvnet':
        mdl_piece = 'dcn'
    elif modelToEval.casefold() == "atcnet":
        mdl_piece = 'atc'
    elif modelToEval.casefold() == 'eegconformer':
        mdl_piece = 'con'
    elif modelToEval.casefold() == 'psdnet3':
        mdl_piece = 'ps3'
    elif modelToEval.casefold() == 'resnet':
        mdl_piece = 'res'
    elif modelToEval.casefold() == 'transformeeg':
        mdl_piece = 'etr'
    else:
        mdl_piece = 'UnknownModel'
    

    if pipelineToEval == 'ica':
        pipe_piece = 'ica'
    
    if downsample:
        freq_piece = '125'
    else:
        freq_piece = '250'

    if augment_list is None:
        aug1_piece = '000'
        aug2_piece = '000'
    else:
        aug1_piece = str(augidx1+1).zfill(3)
        aug2_piece = str(augidx2+1).zfill(3)

    if csp:
        csp_piece = str(Nfilters).zfill(3)
    else:
        csp_piece = '000'
    
    out_piece = str(outerFold+1).zfill(3)
    in_piece = str(innerFold+1).zfill(3)
    lr_piece = str(int(lr*1e6)).zfill(6)
    over_piece = str(int(overlap*100)).zfill(3)
    chan_piece = str(Chan).zfill(3)
    win_piece = str(round(window)).zfill(3)
    decay_piece = str(int(weight_decay*1e6)).zfill(6)
    
    file_name = '_'.join(
        [task_piece, pipe_piece, freq_piece, mdl_piece, 
         out_piece, in_piece, chan_piece, win_piece,
         over_piece, csp_piece, aug1_piece, aug2_piece, 
         lr_piece, decay_piece
        ]
    )
    model_path = start_piece_mdl + file_name + '.pt'
    results_path = start_piece_res + file_name + '.pickle'

    verbose = True
    if verbose:
        print('saving model and results in the following paths')
        print(model_path)
        print(results_path)
    
    # Save the model
    Mdl.eval()
    Mdl.to(device='cpu')
    torch.save(Mdl.state_dict(), model_path)
    
    # Save the scores
    with open(results_path, 'wb') as handle:
        pickle.dump(scores, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    if verbose:
        print('run complete')
