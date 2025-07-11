import os
import torch
import argparse
import itertools
import subprocess
import time
from AllFnc.utilities import (
    restricted_float,
    positive_float,
    positive_int_nozero,
    positive_int,
    makeGrid,
)
print_string = True
custom_env   = True

def run_single_training(arg_dict, print_string = False, custom_environment = None):
    # create args string
    arg_str = " -D " + arg_dict["dataPath"] + \
    " -p " + arg_dict["pipelineToEval"] + \
    " -t " + arg_dict["taskToEval"] + \
    " -m " + arg_dict["modelToEval"] + \
    " -o " + str(arg_dict["outer"]) + \
    " -i " + str(arg_dict["inner"]) + \
    " -d " + str(arg_dict["downsample"]) + \
    " -z " + str(arg_dict["z_score"]) + \
    " -r " + str(arg_dict["rem_interp"]) + \
    " -b " + str(arg_dict["batchsize"]) + \
    " -O " + str(arg_dict["overlap"]) + \
    " -l " + str(arg_dict["lr"]) + \
    " -a " + str(arg_dict["adamdecay"]) + \
    " -w " + str(arg_dict["window"]) + \
    " -c " + str(arg_dict["csp"]) + \
    " -f " + str(arg_dict["csp_filters"])
    
    if arg_dict["augmentations"] is None:
        arg_str += " -A " + str(arg_dict["augmentations"])
    else:
        arg_str += " -A " + "\"" + str(arg_dict["augmentations"]) + "\""
    
    arg_str += " -W " + str(arg_dict["workers"]) + \
    " -v " + str(arg_dict["verbose"]) + \
    " -g " + str(arg_dict["gpudevice"]) + \
    " -s " + str(arg_dict["seed"])

    # print argument string if needed
    if print_string:
        print(arg_str)

    p = subprocess.run(
        #"python3 RunSingleTraining.py" + arg_str,
        "python3 RunSingleTrainingFullPD.py" + arg_str,
        shell   = True, 
        check   = True,
        timeout = 7200,
        env     = custom_environment
    )    
    return

if __name__ == '__main__':

    if custom_env:
        env                            = os.environ.copy()
        env["OMP_NUM_THREADS"]         = "16" # export OMP_NUM_THREADS=1
        env["OPENBLAS_NUM_THREADS"]    = "16" # export OPENBLAS_NUM_THREADS=1
        env["MKL_NUM_THREADS"]         = "16" # export MKL_NUM_THREADS=1
        env["VECLIB_MAXIMUM_THREADS"]  = "16" # export VECLIB_MAXIMUM_THREADS=1
        env["NUMEXPR_NUM_THREADS"]     = "16" # export NUMEXPR_NUM_THREADS=1
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        env = None

    help_d = """
    RunCVDA is a copy of RunCV used to run a 10-10 N-LNSO
    for custom combinations of (models, data augmentation).
    """
    parser = argparse.ArgumentParser(description=help_d)
    
    parser.add_argument(
        "-d",
        "--datapath",
        dest      = "dataPath",
        metavar   = "datasets path",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = None,
        help      = """
        The dataset path. It must point to a directory containing a set of
        subdirecotries, with all the preprocessed EEGs stored as pickle files.
        Each subfolder is expected to contain the EEGs preprocessed with a 
        specific preprocessing pipeline.
        So, the path should look like.
        
         root_path
         |  + pipeline_1
         |  |  + EEG_1
         |  |  + EEG_2
         |  |  + EEG_3
         |  |  + ...
         |  |  + EEG_n
         |  + pipeline_2
         |  |  + EEG_1
         |  |  + ...
         |  |  + EEG_n
         |  + ...
         |  + pipeline_n

        """,
    )
    
    parser.add_argument(
        "-s",
        "--start",
        dest      = "start_idx",
        metavar   = "starting index",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 0,
        help      = """
        The starting index.
        It can be used to restart the trainings if one failed for some reasons. 
        """
    )
    parser.add_argument(
        "-e",
        "--end",
        dest      = "end_idx",
        metavar   = "ending index",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 0,
        help      = """
        The ending index.
        It can be used to stop the trainings at specific positions.
        You can use it if you want to split the total number of trainings on multiple
        GPUs or you want to run multiple training in parallel on the same GPU
        """
    )

    # basically we overwrite the dataPath if something was given
    args = vars(parser.parse_args())
    dataPathInput = args['dataPath']
    StartIdx = args['start_idx']
    EndIdx = args['end_idx']

    aug_list = [
        'flip_horizontal',
        'flip_vertical',
        'add_band_noise',
        'add_eeg_artifact',
        'add_noise_snr',
        'channel_dropout',
        'masking',
        'warp_signal',
        'random_FT_phase',
        'phase_swap',
    ]

    models = [
        "xeegnet", "eegnet", "shallownet", "eegconformer",
        "deepconvnet", "transformeeg", "resnet", "atcnet"
    ]
    augmns = [
        ["add_eeg_artifact", "phase_swap"],
        ["channel_dropout", "add_band_noise"],
        ["channel_dropout", "masking"],
        ["add_eeg_artifact", "add_band_noise"],
        ["phase_swap", "add_band_noise"],
        ["masking", "flip_horizontal"],
        ["random_FT_phase", "add_eeg_artifact"],
        ["phase_swap", "masking"]
    ]
    arg_list = []
    for model, aug in zip(models,augmns):
        PIPE_args = {
            "dataPath":       ['/data/delpup/datasets/eegpickle/'],
            "pipelineToEval": ["ica"],
            "taskToEval":     ["parkinson"],
            "modelToEval":    [model], 
            "downsample":     [True],
            "z_score":        [True],
            "rem_interp":     [True],
            "batchsize":      [64],
            "window":         [16.0],
            "overlap":        [0.25],
            "lr":             [2.5e-4],
            "adamdecay":      [0.0],
            "workers":        [0],
            "verbose":        [False],
            "csp":            [False],
            "csp_filters":    [10],
            "augmentations":  [aug],
            "gpudevice":      ["cuda:1"],
            "seed":           [42], #[42],
            "inner":          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "outer":          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
        if dataPathInput is not None:
            PIPE_args['dataPath'] = [dataPathInput]
        arg_list += makeGrid(PIPE_args)

    # print the final dictionary
    print("running trainings with the following set of parameters:")
    print(" ")
    for key in PIPE_args:
        if key == "augmentations":
            print(f"{key:15} ==> ", augmns)
        elif key == "modelToEval":
            print(f"{key:15} ==> ", models)
        else:
            print( f"{key:15} ==> {PIPE_args[key]}") 
    
    # Run each training in a sequential manner
    N = len(arg_list)
    print(f"the following setting requires to run {N:5} trainings")
    if StartIdx>0:
        print(f"Restart from training number {StartIdx:5}")
        StartIdx = StartIdx - 1
    if EndIdx>0:
        print(f"Will end at training number {EndIdx:5}")
    EndIdx = EndIdx - 1
    if EndIdx>0 and StartIdx>0 and EndIdx<=StartIdx:
        raise ValueError("ending index cannot be lower than the starting index")
    
    for i in range(StartIdx, N):
        if i==EndIdx:
            print(f"reached end idx. Stopping simulation at training number {i+1:<5}")
            break
        print(f"running training number {i+1:<5} out of {N:5}")
        Tstart = time.time()
        run_single_training(arg_list[i], print_string, env)
        Tend = time.time()
        Total = int(Tend - Tstart)
        print(f"training performed in    {Total:<5} seconds")
    
    print(f"Completed all {N:5} trainings")
    # Just a reminder to keep your GPU cool
    if (N-StartIdx)>1000:
        print(f"...Is your GPU still alive?")
