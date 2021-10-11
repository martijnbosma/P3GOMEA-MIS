import random
import numpy as np
import torch
import torch.nn.functional as F
import json
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
from distutils.dir_util import copy_tree
from torchvision.utils import save_image

def model_to_solution(model):
    solution = []
    for l in model:
        solution.append(str(l))
    return ''.join(solution)

def get_device(device_id):
    if device_id >= 0:
        device = torch.device('cuda:%d' % device_id)
    else:
        device = torch.device('cpu')
    return device

def save_result(result, dir, model_description):
    filename = '%s/model_%s.json' % (dir, model_description)
    json.dump(result, open(filename, 'w'))

def check_model_exist(dir, model_description):
    filename = '%s/model_%s.json' % (str(dir), str(model_description))
    return os.path.exists(filename)

def load_json(dir, model_description):
    filename = join(dir, 'model_%s.json' % (model_description))
    return json.load(open(filename,'r'))

def copy_solution(dir, model1_description, model2_description):
    filename1 = '%s/model_%s.json' % (dir, model1_description)
    filename2 = '%s/model_%s.json' % (dir, model2_description)
    shutil.copyfile(filename2, filename1)

def copy_and_overwrite(src_path, dst_path):
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    shutil.copytree(src_path, dst_path)

def copy_and_add(src_path, dst_path):
    maybe_mkdir(dst_path)
    copy_tree(src_path, dst_path)

def get_free_memory(device):
    total_memory = torch.cuda.get_device_properties(device).total_memory
    consumed_memory = torch.cuda.memory_allocated(device)
    return (total_memory - consumed_memory) // (1024*1024)

def setup_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled=True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d

def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_cyclic_lr(epoch, lrs, lr_init, lr_start_cycle, cycle_period):
    n_0 = lr_init
    l_0 = lr_start_cycle  + cycle_period
    if epoch < lr_start_cycle:         
        lr = 0.5 * n_0 * (1 + np.cos((np.pi * epoch) / l_0))
    else:
        lr = lrs[-cycle_period]
    return lr

def set_random_seeds(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_patients(dir_images):
    patients = set([])
    files = os.listdir(dir_images)
    for file in files:
        try:
            patient = int(file.split('_')[0].replace('Case',''))
            patients.add(patient)
        except Exception:
            pass
    return list(patients)
 
def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def save_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)

def load_json_file(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def save_json_file(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def load_dataset(dir_images):
    files = os.listdir(dir_images)
    return files

def load_patients(dir_images, patients):
    images = os.listdir(dir_images)
    patients_list = []
    for image in images:
        case_identifier = int(image.split('_')[0].replace('Case',''))
        if case_identifier in patients:
            patients_list.append(image)
    return patients_list

def maybe_mkdir(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                print("WARNING: Folder %s already existed and does not need to be created" % directory)

def debug_outputs(output_folder, b, data, target, output):
    # probs = F.softmax(output, 1)
    # print("X", data.min(), data.max(), data.size())
    # print("y", target.min(), target.max(), target.size())
    # print("output", output.min(), output.max(), output.size())
    # print("probs", probs.min(), probs.max(), probs.size())
    # maybe_mkdir(output_folder)
    # input_image = data[0,:,:]
    # save_image(input_image, join(output_folder,'input_{}.png'.format(b)))
    # output_image = target[0]
    # save_image(output_image,join(output_folder,'target_{}.png'.format(b)))
    # probs_image = output[0]
    # save_image(probs_image,join(output_folder,'prob_{}.png'.format(b)))

    probs = F.softmax(output, 1)
    print("X", data.min(), data.max(), data.size())
    print("y", target.min(), target.max(), target.size())
    print("output", output.min(), output.max(), output.size())
    print("probs", probs.min(), probs.max(), probs.size())
    segment = probs.argmax(1).float()
    if target.max() > 1:
        target = torch.mul(target, 0.5)
        segment = torch.mul(segment, 0.5)
    maybe_mkdir(output_folder)
    input_image = data[0,0,:,:]
    save_image(input_image, join(output_folder,'input_1_{}.png'.format(b)))
    input_image = data[0,1,:,:]
    save_image(input_image, join(output_folder,'input_2_{}.png'.format(b)))
    output_image = target[0]
    save_image(output_image,join(output_folder,'target_{}.png'.format(b)))
    probs_image = segment[0]
    save_image(probs_image,join(output_folder,'prob_{}.png'.format(b)))

def save_segmentations(output_folder, data, target, segment, b=0, i=0, batch_size=32):
    segment = segment.argmax(0).float()
    if data.shape[0] > 1: 
        segment = torch.mul(segment, 0.5)
        target = torch.mul(target, 0.5)
        input_image = data[0,:,:]
        save_image(input_image, join(output_folder,'input_{}_{}.png'.format(b, i)))
        input_image = data[1,:,:]
        save_image(input_image, join(output_folder,'input_2_{}_{}.png'.format(b, i)))
        output_image = target
        save_image(output_image,join(output_folder,'target_{}_{}.png'.format(b, i)))
        probs_image = segment
        save_image(probs_image,join(output_folder,'seg_{}_{}.png'.format(b, i)))
    else:
        input_image = data
        save_image(input_image, join(output_folder,'input_{}_{}.png'.format(b, i)))
        target_image = target
        save_image(target_image,join(output_folder,'target_{}_{}.png'.format(b, i)))
        output_image = segment
        save_image(output_image,join(output_folder,'seg_{}_{}.png'.format(b, i)))

def segment_with_scores(folder_with_masks, dice_score, surf_dice_score, hausdorff_score, b=0, i=0):
    # files = subfiles(folder_with_masks, suffix=".png", join=False)
    ref_image = Image.open(os.path.join(folder_with_masks, 'input_{}_{}.png'.format(b, i))).convert('RGBA')
    ref_mask = Image.open(os.path.join(folder_with_masks, 'target_{}_{}.png'.format(b, i))).convert('RGBA')
    pred_mask = Image.open(os.path.join(folder_with_masks, 'seg_{}_{}.png'.format(b, i))).convert('RGBA')
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    for _ in range(0, len(axs)):
        axs[0].imshow(ref_image)
        axs[0].set_title('Input image')
        ref_mask = change_color(ref_mask, (255, 255, 255), (250,20,20,90))
        ref_mask = change_color(ref_mask, (127.5, 127.5, 127.5), (20,20,250,90))
        ref_mask = change_color(ref_mask, (0, 0, 0), (0, 0, 0, 0))
        axs[1].imshow(ref_image)
        axs[1].imshow(ref_mask)
        axs[1].set_title('Reference')
        pred_mask = change_color(pred_mask, (255, 255, 255), (250,20,20,90))
        pred_mask = change_color(pred_mask, (127.5, 127.5, 127.5), (20,20,250,90))
        pred_mask = change_color(pred_mask, (0, 0, 0), (0, 0, 0, 0))
        axs[2].imshow(ref_image)
        axs[2].imshow(pred_mask)
        axs[2].set_title('NAS')
        axs[2].text(2, 126, "Dice: {:.2f}     Surface Dice: {:.2f}     Hausdorff: {:.2f}".format(dice_score, surf_dice_score, hausdorff_score), c='white')
        # test_mask2 = change_color(test_mask, (255, 255, 255), (250,20,20,90))
        # test_mask2 = change_color(test_mask, (127.5, 127.5, 127.5), (20,20,250,90))
        # test_mask2 = change_color(test_mask, (0, 0, 0), (0, 0, 0, 0))
        # axs[3].imshow(ref_image)
        # axs[3].imshow(test_mask2)
        # axs[3].set_title('U-NET')
    plt.savefig(join(folder_with_masks, "segmentation_with_scores_{}_{}.png".format(b, i)))
    plt.close()

def change_color(im, ex_color, new_color):
    # Process every pixel
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            current_color = im.getpixel( (x,y) )
            if current_color[0] > ex_color[0] - 1 and current_color[1] > ex_color[1] - 1 and \
                    current_color[2] > ex_color[2] - 1 and current_color[0] < ex_color[0] + 1 and \
                    current_color[1] < ex_color[1] + 1 and current_color[2] < ex_color[2] + 1:
                im.putpixel( (x,y), new_color)
    return im


def calculate_field_of_view(architecture, encoding_length):
    field_of_view = 3
    depth = int(architecture[0])
    for i in range(encoding_length, len(architecture)):
        if i % encoding_length == 0:
            conv_size = 3 if int(architecture[i+encoding_length-2]) == 0 else 5 if int(architecture[i+encoding_length-2]) == 1 else 7 if int(architecture[i+encoding_length-2]) == 2 else 1
            if int(architecture[i]) > depth: 
                field_of_view += conv_size-1
                field_of_view *= 2
            elif int(architecture[i]) > depth: 
                field_of_view += conv_size-1
                field_of_view += conv_size-1
            else: 
                field_of_view += conv_size-1
                field_of_view += conv_size-1
            depth = int(architecture[i])
        else:
            pass
    return field_of_view

join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir

def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def subfolders(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


if __name__ == '__main__':
    print(calculate_field_of_view('020120220210201120010011122002', 3))