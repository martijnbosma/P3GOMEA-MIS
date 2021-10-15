# Code by Martijn Bosma

import os
from random import seed
import sys
import time
import csv

from unet.utils_nas import *
from unet.losses import SoftDiceLoss, SoftDiceLossMultiClass, DC_and_CE_loss
from unet.metrics import *
from unet.models.NAS_WNet_blocks_convs import NAS_WNet_blocks_convs
from unet.models.NAS_WNet_blocks import NAS_WNet_blocks
from unet.models.NAS_WNet import NAS_WNet
from unet.models.blocks_and_utils import InitWeights_He

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import segmentation_models_pytorch as smp
from datetime import datetime
from thop.profile import profile
from tqdm import trange


class UNetTrainer():
    def __init__(self, output_folder, UNET_PARAMS=None):
        self.initialized = False
        self.pin_memory = True
        self.starting_epoch = 0
        self.output_folder = output_folder
        self.log_file = None
        self.epoch = 0
        self.network = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = self.val_gen = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_tr_eval_metrics = []
        self.all_val_eval_metrics = []
        self.online_eval_foreground_dc = []
        self.online_tr_foreground_dc = []
        self.eval_tp = []
        self.eval_fp = []
        self.eval_fn = []
        self.tr_tp = []
        self.tr_fp = []
        self.tr_fn = []
        self.nas_result = None
        self.num_batches_per_epoch = None
        self.num_val_batches_per_epoch = None
        self.pool_kernels = None
        self.conv_kernels = None
        self.nonscaling_kernels = None
        self.skip_connect_encodings = None
        self.depth = None
        self.scaler = None
        self.training_time = None

        if UNET_PARAMS is None: 
            self.patch_size = [128, 128]
            self.batch_size = 16
            self.max_num_epochs = 25
            self.initial_lr = 1e-3
            self.loss = SoftDiceLoss()
            self.activation_func = None
            self.architecture = None
            self.pin_memory = True
            self.net_nonscaling_operations = None
            self.epochs_this_round = self.max_num_epochs * 5
            self.starting_epoch = 0
            self.fold = 0
            self.epochs_per_round = None
            self.weight_decay = 3e-5
            self.num_input_channels = 1
            self.base_num_features = 32
            self.num_classes = 1
            self.conv_per_stage = 2
            self.tr_multiplier = 5
            self.val_multiplier = 1
            self.fp16 = False
        else: 
            self.set_params(UNET_PARAMS)

#--------------------------------------------SET PARAMETERS--------------------------------------------#

    def set_params(self, params):
        self.num_tr_evals = params.num_tr_evals
        self.num_val_evals = params.num_val_evals
        self.dataset = params.dataset
        self.patch_size = params.patch_size
        self.batch_size = params.batch_size
        self.max_num_epochs = params.max_epochs
        self.initial_lr = params.initial_lr
        self.loss = SoftDiceLossMultiClass() # DC_and_CE_loss(weight_ce=0.2) 
        self.activation_func = params.activation_func
        self.epochs_per_round = params.epochs_per_round
        self.weight_decay = params.weight_decay
        self.num_input_channels = params.num_input_channels
        self.base_num_features = params.base_num_features
        self.num_classes = params.num_classes
        self.use_blocks = params.use_blocks
        self.use_convs = params.use_convs
        self.conv_per_stage = params.conv_per_stage
        self.fp16 = params.fp16
        self.save_final_checkpoint = params.save_final_checkpoint
        self.do_plot_progress = params.do_plot_progress
        self.plot_architecture = params.plot_architecture
        self.do_segmentation = params.do_segmentation
        self.score_segmentations = params.score_segmentations
        self.calculate_network_size_and_complexity = params.calculate_network_size_and_complexity


    def set_etr(self, epochs_this_round):
        self.epochs_this_round = epochs_this_round

    def set_starting_epoch(self):
        if self.epoch == 0:
            self.starting_epoch = 0
        else:
            self.starting_epoch = self.epoch

#--------------------------------------------INITIALIZE NETWORK--------------------------------------------#

    def initialize(self, architecture, dataloaders, fold=0, seed=42):
        if not self.initialized:
            maybe_mkdir(self.output_folder)
            self.fold = fold
            self.seed = seed
            self.tr_gen = dataloaders['train']
            self.val_gen = dataloaders['val']
            self.num_batches_per_epoch = len(self.tr_gen)
            self.num_val_batches_per_epoch = len(self.val_gen)
            self.num_batchgens_tr = self.num_tr_evals // (self.num_batches_per_epoch * self.batch_size) + \
                                 (self.num_tr_evals % (self.num_batches_per_epoch * self.batch_size) > 0)
            self.num_batchgens_val = 1 #self.num_val_evals // (self.num_val_batches_per_epoch * self.batch_size)
            if self.use_convs and self.use_blocks:
                self.set_depth_convs_and_blocks(architecture)
            else:
                self.set_depth_and_ops(architecture, self.use_blocks)
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

        else:
            self.print_to_log_file('UNET-py: self.initialized is True, not running self.initialize again')
        self.initialized = True

    def initialize_network(self):
        conv_op = nn.Conv2d
        dropout_op = nn.Dropout2d
        norm_op = nn.InstanceNorm2d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'inplace': True}
        if self.use_blocks and not self.use_convs:
            self.network = NAS_WNet_blocks(self.num_input_channels, self.base_num_features, self.num_classes,
                            len(self.pool_kernels),
                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                            dropout_op_kwargs,
                            net_nonlin, net_nonlin_kwargs, False, False, InitWeights_He(1e-2),
                            self.pool_kernels, self.conv_kernels, False, True, True, depth=self.depth, 
                            skip_connects=self.skip_connect_encodings, block_types=self.blocks)
        elif not self.use_blocks and self.use_convs:
            self.network = NAS_WNet(self.num_input_channels, self.base_num_features, self.num_classes,
                            len(self.pool_kernels),
                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                            dropout_op_kwargs,
                            net_nonlin, net_nonlin_kwargs, False, False, InitWeights_He(1e-2),
                            self.pool_kernels, self.conv_kernels, False, True, True, depth=self.depth, 
                            skip_connects=self.skip_connect_encodings)
        elif self.use_blocks and self.use_convs:
                        self.network = NAS_WNet_blocks_convs(self.num_input_channels, self.base_num_features, self.num_classes,
                            len(self.pool_kernels),
                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                            dropout_op_kwargs,
                            net_nonlin, net_nonlin_kwargs, False, False, InitWeights_He(1e-2),
                            self.pool_kernels, self.conv_kernels, False, True, True, depth=self.depth, 
                            skip_connects=self.skip_connect_encodings, block_types=self.blocks)
        else:
            #Debugging, custom network
            self.network = smp.Unet(
                        encoder_name='vgg19_bn',# "resnet_50""resnet152", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,           # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=self.num_input_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=self.num_classes,                      # model output channels (number of classes in your dataset)
                        )

        if self.calculate_network_size_and_complexity:
            inputs = torch.randn((1, self.num_input_channels, self.base_num_features, self.base_num_features))
            self.total_ops, self.total_params = profile(self.network, (inputs,), verbose=False)
            self.mmacs = np.round(self.total_ops / (1000.0**2), 2)
            self.network_params = int(self.total_params)

        if torch.cuda.is_available():
            self.network.train()
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.initial_lr)
        # self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                                  momentum=0.99, nesterov=True)

    def set_depth_and_ops(self, architecture, use_blocks):
        if architecture is not None:
            depth = []
            if use_blocks:
                blocks = []
            else:
                blocks = None
                conv_operations = []  # nonscaling_operations = []
            skip_connect_encodings, pool_operations, conv_operations  = [], [], []
            for i in range(len(architecture)):
                if i % 3 == 0:
                    depth.append(int(architecture[i]))
                elif i % 3 == 1:
                    if use_blocks:
                        blocks.append(int(architecture[i]))
                        conv_operations.append([3,3])
                    else:
                        if architecture[i] == '0':
                            conv_operations.append([3, 3])
                        elif architecture[i] == '1':
                            conv_operations.append([5, 5])
                        elif architecture[i] == '2':
                            conv_operations.append([7, 7])
                        else:
                            conv_operations.append([1, 1])
                    pool_operations.append([2,2])
                else:
                    skip_connect_encodings.append(int(architecture[i]))
            self.pool_kernels = pool_operations
            self.conv_kernels = conv_operations
            self.blocks = blocks
            self.depth = depth
            self.skip_connect_encodings = skip_connect_encodings

    def set_depth_convs_and_blocks(self, architecture):
        if architecture is not None:
            depth = []
            blocks = []
            skip_connect_encodings, pool_operations, conv_operations  = [], [], []
            for i in range(len(architecture)):
                if i % 4 == 0:
                    depth.append(int(architecture[i]))
                elif i % 4 == 1:
                        blocks.append(int(architecture[i]))
                elif i % 4 == 2:
                    if architecture[i] == '0':
                        conv_operations.append([3, 3])
                    elif architecture[i] == '1':
                        conv_operations.append([5, 5])
                    elif architecture[i] == '2':
                        conv_operations.append([7, 7])
                    else:
                        conv_operations.append([1, 1])
                    pool_operations.append([2,2])
                else:
                    skip_connect_encodings.append(int(architecture[i]))
            self.pool_kernels = pool_operations
            self.conv_kernels = conv_operations
            self.blocks = blocks
            self.depth = depth
            self.skip_connect_encodings = skip_connect_encodings

#--------------------------------------------TRAINING CODE--------------------------------------------#

    def run_training(self):

        maybe_mkdir(self.output_folder)
        # don't train finished networks
        if self.starting_epoch >= self.max_num_epochs:
            self.epoch -= 1
            self.on_training_end()
            return

        self.update_lr(self.epoch)       

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.plot_architecture:
            self.plot_network_architecture()

        if not self.initialized:
            raise("The Network has not been initialized........\n")

        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        training_start_time = time.time()

        while self.epoch < self.max_num_epochs:

            # train one epoch
            epoch_start_time = time.time()
            train_losses_epoch = []
            self.network.train()
            with trange(self.num_batches_per_epoch * self.num_batchgens_tr) as tbar:
                for b in tbar:
                    if (b % self.num_batches_per_epoch == 0):
                        train_it = iter(self.tr_gen)
                    tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))
                    l = self.run_iteration(train_it, b=b, do_backprop=True, run_evaluation=True, fp16=self.fp16)
                    tbar.set_postfix(loss=l)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(float(np.mean(train_losses_epoch)))
            # validation 
            with torch.no_grad():
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch * self.num_batchgens_val):
                    if (b % self.num_val_batches_per_epoch == 0):
                        val_it = iter(self.val_gen)
                    l = self.run_iteration(val_it, b=b, do_backprop=False, run_evaluation=True, fp16=False)
                    val_losses.append(l)
            self.all_val_losses.append(float(np.mean(val_losses)))
            self.print_to_log_file("train loss : %.6f" % self.all_tr_losses[-1], \
                    "\nvalidation loss: %.6f" % self.all_val_losses[-1])

            continue_training = self.on_epoch_end()

            epoch_end_time = time.time()
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1

        self.training_time = time.time() - training_start_time
            
        return self.on_training_end()


    def on_epoch_end(self):
        self.finish_online_evaluation()
        self.update_lr()

        continue_training = (self.epoch < self.max_num_epochs) and (
                    self.epoch - self.starting_epoch < self.epochs_this_round - 1)

        return continue_training

    def on_training_end(self):
        if self.do_plot_progress: 
            self.plot_progress()
            self.nas_results()
        if self.do_segmentation:
            self.segment_results(self.score_segmentations)
        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))
        to_return = {"tr_loss": self.all_tr_losses, "val_loss": self.all_val_losses, 
                    "val_acc": self.all_val_eval_metrics, "tr_acc": self.all_tr_eval_metrics,
                    "tr_time": self.training_time}
        if self.calculate_network_size_and_complexity:
            to_return["mmacs"] = self.mmacs
            to_return["network_params"] = self.network_params
        return to_return

    def update_lr(self, epoch=None):
        """
        Update lr based on polynomial decay
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        adjust_learning_rate(self.optimizer, poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9))

    def run_iteration(self, data_generator, b, do_backprop=True, run_evaluation=False, fp16=False):
        data, target, _ = next(data_generator)

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        if do_backprop:
            self.network.train()

        self.optimizer.zero_grad()
        if fp16:
            with torch.cuda.amp.autocast():
                output = self.network.forward(data)
                l = self.loss(output, target)
        else: 
            output = self.network.forward(data)
            l = self.loss(output, target)

        # debug_outputs(self.output_folder, b, data, target, output)

        if do_backprop:
            if fp16:
                self.scaler.scale(l).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                l.backward()
                self.optimizer.step()
        
        if run_evaluation:
            self.network.eval()
            self.run_online_evaluation(output, target, do_backprop) #online

        return l.detach().cpu().numpy()
    
    def run_evaluation(self, output, target, training):
        if isinstance(output, list):
            target = target[0]
            output = output[0]
        with torch.no_grad():
            dc = DiceCoeff()
            if training:
                self.online_tr_foreground_dc.append(dc.calculate(target, output))
            else:
                self.online_eval_foreground_dc.append(dc.calculate(target, output))

    def finish_evaluation(self):
        if len(self.online_tr_foreground_dc) > 0:
            avg_tr_dice_score = np.mean(self.online_tr_foreground_dc)
            self.all_tr_eval_metrics.append(avg_tr_dice_score)
            self.print_to_log_file("avg tr dice score:", str(avg_tr_dice_score))
        avg_val_dice_score = np.mean(self.online_eval_foreground_dc)
        self.all_val_eval_metrics.append(avg_val_dice_score)
        self.print_to_log_file("avg val dice score:", str(avg_val_dice_score))
        self.online_eval_foreground_dc, self.online_tr_foreground_dc = [], []
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def run_online_evaluation(self, output, target, training=False):
        with torch.no_grad():
            dc = MultiClassDiceCoeff()
            soft_dice, tp_hard, fp_hard, fn_hard = dc.calculate(target, output)
            if training:
                self.online_tr_foreground_dc.append(list(soft_dice))
                self.tr_tp.append(list(tp_hard))
                self.tr_fp.append(list(fp_hard))
                self.tr_fn.append(list(fn_hard))
            else:
                self.online_eval_foreground_dc.append(list(soft_dice))
                self.eval_tp.append(list(tp_hard))
                self.eval_fp.append(list(fp_hard))
                self.eval_fn.append(list(fn_hard))

    def finish_online_evaluation(self):
        if len(self.online_tr_foreground_dc) > 0:
            self.tr_tp = np.sum(self.tr_tp, 0)
            self.tr_fp = np.sum(self.tr_fp, 0)
            self.tr_fn = np.sum(self.tr_fn, 0)
            global_dc_per_class_tr = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                    zip(self.tr_tp, self.tr_fp, self.tr_fn)] if not np.isnan(i)]
            self.all_tr_eval_metrics.append(np.mean(global_dc_per_class_tr))
            self.print_to_log_file("avg foreground tr dice score:", str(global_dc_per_class_tr)) #, np.mean(self.online_tr_foreground_dc, axis=0))

        self.eval_tp = np.sum(self.eval_tp, 0)
        self.eval_fp = np.sum(self.eval_fp, 0)
        self.eval_fn = np.sum(self.eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.eval_tp, self.eval_fp, self.eval_fn)] if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.print_to_log_file("avg foreground val dice score:", str(global_dc_per_class))# ,np.mean(self.online_eval_foreground_dc, axis=0))

        self.online_eval_foreground_dc, self.online_tr_foreground_dc = [], []
        self.eval_tp, self.eval_fp, self.eval_fn, self.tr_tp, self.tr_fp, self.tr_fn = [], [], [], [], [], []

#--------------------------------------------VISUALIZATION AND LOGGING--------------------------------------------#
    
    def plot_progress(self):
        plt.rcParams["figure.figsize"] = [14,14]
        try:
            font = {'weight': 'normal',
                    'size': 18}

            plt.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch+1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())


    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=False):
        if also_print_to_console:
            print(*args)
            return
        timestamp = time.time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(self.output_folder, "training_log_fold%d_seed%d.txt" % (self.fold, self.seed))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                time.sleep(0.5)
                ctr += 1

    def plot_network_architecture(self):
        try:
            from batchgenerators.utilities.file_and_folder_operations import join
            import hiddenlayer as hl
            if torch.cuda.is_available():
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                                   transforms=None)
            else:
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)),
                                   transforms=None)
            g.save(join(self.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            self.print_to_log_file("UNET-py: Unable to plot network architecture:")
            self.print_to_log_file(e)

            self.print_to_log_file("\nUNet-py: printing the network instead:\n")
            self.print_to_log_file(self.network)
            self.print_to_log_file("\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def segment_results(self, score_segmentations=False):
        n = len(self.val_gen)
        val_it = iter(self.val_gen)
        res = {} 
        for j in range(n):
            data, target, im_fname = next(val_it)
            self.network.eval()
            data = maybe_to_torch(data)
            if torch.cuda.is_available():
                data = to_cuda(data)
            with torch.cuda.amp.autocast():
                output = self.network.forward(data)
            # output = self.network(data)
            segment = torch.softmax(output, 1)
            
            edited_dir = os.environ.get('UNet_edited') if os.environ.get('UNet_edited') is not None else None
            edited_dir = join(edited_dir, self.dataset)
            filename = join(edited_dir, 'spacing.json')
            spacings = json.load(open(filename,'r'))

            for i in range(data.shape[0]):
                patient = im_fname[i][4:6]
                patient_slice = im_fname[i][7:].split(".")[0]
                dice = DiceCoeff() 
                surf_dice = SurfaceDice(2, spacing=spacings[patient])
                hausdorff = HausdorffDistance(95, spacing=spacings[patient])
                dice_score = dice.calculate(target[i, :, :, :], segment[i, :, :, :], False)
                surf_dice_score = surf_dice.calculate(target[i, :, :, :], segment[i, :, :, :], False) 
                hausdorff_score = hausdorff.calculate(target[i, :, :, :], segment[i, :, :, :], False)

                if patient not in res: 
                    res[patient] = {}
                res[patient][patient_slice] = {'dice':dice_score, 'surface dice':surf_dice_score, 'hausdorff':hausdorff_score} 
                with torch.no_grad():
                    save_segmentations(self.output_folder, data[i, :, :, :], target[i, :, :, :], segment[i, :, :, :], b=patient, i=patient_slice, batch_size=self.batch_size)
                if score_segmentations:
                    segment_with_scores(self.output_folder, dice_score, surf_dice_score, hausdorff_score, b=patient, i=patient_slice)
        json.dump(res, open(join(self.output_folder, "scores_seed{}_fold{}.json".format(self.seed, self.fold)), 'w'))
    
    def nas_results(self):
        if self.nas_result is None:
            maybe_mkdir(self.output_folder)
            self.nas_result = join(self.output_folder, "nas_result_seed{}_fold{}.csv".format(self.seed, self.fold))
        with open(self.nas_result, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(["tr_loss", self.all_tr_losses])
            writer.writerow(["val_loss", self.all_val_losses])
            if len(self.all_tr_eval_metrics)>0: writer.writerow(["tr_acc", self.all_tr_eval_metrics])
            writer.writerow(["val_acc", self.all_val_eval_metrics])
            writer.writerow(["epochs", len(self.all_tr_losses)])

#--------------------------------------------LOADING MODELS--------------------------------------------#

    def load_latest_checkpoint_from_path(self, path, train=True):
        print("UNET-py: ", path)
        if isfile(join(path, "model_final_checkpoint.model")):
            return self.load_checkpoint(join(path, "model_final_checkpoint.model"), train=train)
        raise RuntimeError("No checkpoint found")

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time.time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,'state_dict'):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_eval_metrics)
        }

        torch.save(save_this, fname)
        self.print_to_log_file("saved (took %.2f seconds)" % (time.time() - start_time))

    def load_checkpoint(self, fname, train=True):
        self.print_to_log_file("UNET-py: loading checkpoint")
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train)

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        """
        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch'] + 1
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

        self.all_tr_losses, self.all_val_losses, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("epoch = "+str(self.epoch)+", length of all_tr_lossen = "+str(len(self.all_tr_losses)))
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]