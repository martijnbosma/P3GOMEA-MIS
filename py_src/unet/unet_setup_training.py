# Code by Martijn Bosma

import os
from unet.utils_nas import *
from unet.unet_evaluate import *
import time
import shutil
import gc

import numpy as np
from torch.utils.data import Dataset, DataLoader
from unet.datasets import ImageDataset, DoubleImageDataset
from sklearn.model_selection import KFold
import torch
from collections import OrderedDict
import multiprocessing as mp


class UNet_parameters:
    n_seeds = 1
    folds = [i for i in range(1)]
    tl = 6
    tf = 8
    base_num_features = 32
    batch_size = 64
    patch_size = (128, 128)
    network = "2d"
    num_tr_evals = 2048
    num_val_evals = 1024 # not used
    initial_lr = 0.001
    activation_func = "LeakyReLU"
    weight_decay = 3e-5
    conv_per_stage = 2
    use_blocks = False
    use_convs = True
    fp16 = True
    save_final_checkpoint = False
    do_plot_progress = False
    plot_architecture = False
    score_segmentations = False
    do_segmentation = False
    calculate_network_size_and_complexity = True

    def __init__(self, dataset, task_id, max_epochs, seed, deterministic, epochs_per_round):
        self.dataset = dataset
        self.task_id = task_id
        self.num_input_channels = 1 if task_id == 9 or task_id == 6 else 2
        self.num_classes = 2 if task_id == 9 or task_id == 6 else 3
        self.epochs_per_round = epochs_per_round
        self.max_epochs = max_epochs
        self.seeds = [i + seed for i in range(self.n_seeds)]
        self.deterministic = deterministic

############################################################################################################

def evaluate(solution):
    results_dir = "src/../../../media/results/results_corr06_1/"
    dataset = "Task006_FEDMix"
    seed = 0
    epochs_per_round=40 
    max_epochs=40
    debug="1" 
    deterministic=True

    task_id = int(dataset[6])
    rounds = 0

    print("NAS-py:----------------BEGIN EVALUATION OF:", solution, "----------------")
    
    # Initialize directories
    DIRS = {
        'results_dir': results_dir,
        'edited_dir': os.environ.get('UNet_edited') if os.environ.get('UNet_edited') is not None else None,
        'raw_dir': os.environ.get('UNet_raw_data_base') if os.environ.get('UNet_raw_data_base') is not None else None,
        'preprocess_dir': os.environ.get('UNet_preprocessed') if os.environ.get('UNet_preprocessed') is not None else None,
        'trained_dir': os.environ.get('RESULTS_FOLDER') if os.environ.get('RESULTS_FOLDER') is not None else None
    }

    print('NAS-py: Preparing directory %s' % DIRS["results_dir"])
    os.makedirs(DIRS["results_dir"], exist_ok=True)

    # Initialize parameters
    UNET_PARAMS = UNet_parameters(dataset, task_id, max_epochs, seed, deterministic, epochs_per_round=epochs_per_round)
    DIRS['output_dir'] = join(DIRS['trained_dir'], UNET_PARAMS.network, UNET_PARAMS.dataset+"_{}".format(debug))

    # Remove existing data that may cause interference
    if isdir(join(DIRS['preprocess_dir'], UNET_PARAMS.dataset)):
        print("NAS-py: Deleting contents of "+join(DIRS['preprocess_dir'], UNET_PARAMS.dataset))
        shutil.rmtree(join(DIRS['preprocess_dir'], UNET_PARAMS.dataset))

    # start = time.time()
    if isdir(join(DIRS['edited_dir'], UNET_PARAMS.dataset)):
        patients_dataset = get_patients(join(DIRS['edited_dir'], UNET_PARAMS.dataset, 'images'))
    else:
        raise ValueError("Dataset doesn't exist")

    # if the split file does not exist we need to create it
    if not isfile(join(DIRS['results_dir'], "splits_final.pkl")):
        print("NAS-py: Creating new 5-fold cross-validation split")
        splits = []
        all_keys_sorted = np.sort(patients_dataset)
        kfold = KFold(n_splits=5, shuffle=True, random_state=12345+seed)
        for _, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            splits.append(OrderedDict())
            splits[-1]['train'] = train_keys
            splits[-1]['val'] = test_keys
        splits_file = join(DIRS['results_dir'], "splits_final.pkl")
        save_pickle(splits, splits_file)
        
    else:
        splits_file = join(DIRS['results_dir'], "splits_final.pkl")
        print("NAS-py: Using splits from existing split file:", splits_file)
        splits = load_pickle(splits_file)
        print("NAS-py: The split file contains %d splits." % len(splits))
    
    SPLITS = splits

    model_name, solution_str = '', ''
    for l in range(len(solution)):
        model_name += str(solution[l])
        if l < len(solution)-1:
            model_name += '_' 
        solution_str = solution_str + str(solution[l]) 

    if check_model_exist(DIRS["results_dir"], model_name):
        results = load_json(DIRS["results_dir"], model_name)
        print("NAS-py: Model already evaluated:", model_name)
        avg_val_score = float(np.average([results[str(seed)][str(fold)]["val_acc"][int(0.8*len(results[str(seed)][str(fold)]["val_acc"])):] for seed in UNET_PARAMS.seeds for fold in UNET_PARAMS.folds]))
        print("NAS-py: Results:", avg_val_score)
        print("NAS-py:----------------EVALUATION COMPLETE----------------")
        return avg_val_score

    dir_im = join(DIRS['edited_dir'], UNET_PARAMS.dataset, 'images')
    dir_masks = join(DIRS['edited_dir'], UNET_PARAMS.dataset, 'segmentations')
    
    results = {}

    for seed in UNET_PARAMS.seeds:
        results[seed] = {}

    for fold in UNET_PARAMS.folds:

        if fold < len(SPLITS):
            tr_keys = SPLITS[fold]['train']
            val_keys = SPLITS[fold]['val']
            print("NAS-py: Split %d has %d training and %d validation cases." % (fold, len(tr_keys), len(val_keys)))

        torch.cuda.empty_cache() 

        tr_dataset = ImageDataset([dir_im], [dir_masks], [tr_keys], augment=True, batch_size=UNET_PARAMS.batch_size, image_dim=UNET_PARAMS.patch_size, num_classes=UNET_PARAMS.num_classes)
        val_dataset = ImageDataset([dir_im], [dir_masks], [val_keys], augment=False, batch_size=UNET_PARAMS.batch_size, image_dim=UNET_PARAMS.patch_size, num_classes=UNET_PARAMS.num_classes)
        
        for seed in UNET_PARAMS.seeds:            
            try:
                results[seed][fold] = train(DIRS, UNET_PARAMS, model_name, fold, seed, tr_dataset, val_dataset, solution_str, catch_up_rounds=rounds)
            except Exception as e:
                raise(e)

            print("NAS-py: Results this eval:", float(np.average(results[seed][fold]["val_acc"][int(0.8*len(results[seed][fold]["val_acc"])):])))
        
            
    save_result(results, DIRS["results_dir"], model_name)
    print("NAS-py: Model evaluated:", model_name)
    avg_val_score = float(np.average([results[seed][fold]["val_acc"][int(0.8*len(results[seed][fold]["val_acc"])):] for seed in UNET_PARAMS.seeds for fold in UNET_PARAMS.folds]))
    print("NAS-py: Results:", avg_val_score)
    print("NAS-py:----------------EVALUATION COMPLETE----------------")
    
    return avg_val_score


def train(DIRS, UNET_PARAMS, model_name, fold, seed, tr_dataset, val_dataset, solution_str, catch_up_rounds=0):
    ctxt = mp.get_context('spawn')
    q = ctxt.Queue()
    p = ctxt.Process(target=train_one_cycle(q, DIRS, UNET_PARAMS, model_name, fold, seed, tr_dataset, val_dataset, solution_str, catch_up_rounds))
    p.start()
    p.join()
    result = q.get()
    q.close()
    p.terminate()
    p.close()
    gc.collect()
    return result


def train_one_cycle(q, DIRS, UNET_PARAMS, model_name, fold, seed, tr_dataset, val_dataset, solution_str, catch_up_rounds):

    if UNET_PARAMS.deterministic: 
        setup_torch(seed)
        g = torch.Generator()
        g.manual_seed(seed)

    torch.set_num_threads(UNET_PARAMS.tl)

    dataloader = {
        'val': DataLoader(
            val_dataset, #without augmentations
            batch_size=UNET_PARAMS.batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            drop_last=False),
        'train': DataLoader(
            tr_dataset, #with_augmentations
            batch_size=UNET_PARAMS.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g) #switch on for deterministic
        }

    output_folder = DIRS['output_dir']
    print("NAS-py: Temporary output file for training: {}".format(output_folder))
    print("NAS-py: Desired fold for training: %d, desired seed: %d" % (fold, seed))

    start = time.time()
    # Setup trainer
    trainer = UNetTrainer(output_folder)
    trainer.set_params(UNET_PARAMS)
    assert len(dataloader) > 0, 'NAS-py: Dataloaders empty'
    trainer.initialize(solution_str, dataloader, fold, seed) 
    assert trainer.network is not None, "UNET-py: Network not initialized"

    print("NAS-py: Network creation time: ", time.time() - start)

    output_data = DIRS["output_dir"]
    print("NAS-py: Deleting contents of ", output_data, "...")
    if os.path.exists(output_data):
        shutil.rmtree(output_data)
        print("NAS-py: Removed succesfully!")

    start = time.time()

    epochs_to_run = min((1 + catch_up_rounds) * UNET_PARAMS.epochs_per_round, UNET_PARAMS.max_epochs)
    print ("NAS-py: epochs: ", epochs_to_run)

    results = training_cycle(trainer, epochs_to_run)

    time_elapsed = time.time() - start
    print ('NAS-py: training took:', time_elapsed)

    maybe_mkdir(output_data)
    copy_and_add(output_data, join(DIRS["results_dir"], model_name))
    shutil.rmtree(output_data)
        
    return q.put(results)


def training_cycle(trainer, etr, model_path=None):
    trainer.set_etr(etr)
    if model_path is not None:
        trainer.load_latest_checkpoint_from_path(path=model_path)
        print("UNET-py: Loading the model from previous NAS round \n")
    trainer.set_starting_epoch()
    ret = trainer.run_training()
    return ret
    # trainer.validate(save_softmax=npz, validation_folder_name=val_folder, run_postprocessing_on_folds=not disable_postprocessing_on_folds)


if __name__ == '__main__':
    output_path = "src/../../../media/results/benchmarks_final_paper/"
    to_eval = ['0320100012202000320031203001232111210121'] #'0210132021102000100112102020100111220311' 
    for genotype in to_eval:
        if not isdir(join(output_path, "_".join(genotype))):
            maybe_mkdir(join(output_path, "_".join(genotype)))
            evaluate(genotype, output_path, "Task005_Prostate", seed=10, epochs_per_round=150, max_epochs=150, debug="test", deterministic=True)
        else:
            print(genotype)

# if __name__ == '__main__':
#     main()


    