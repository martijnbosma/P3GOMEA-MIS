import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import surface_distance
from unet.utils_nas import sum_tensor
import torch.nn.functional as F

class DiceCoeff:
    def __init__(self, eps=1e-5):
        self.smooth = eps

    def calculate(self, target, prediction, apply_nonlin=True):
        axis = 0 if len(prediction.shape) == 3 else 1
        if apply_nonlin:
            prediction = torch.softmax(prediction, axis)
        target = F.one_hot(target.squeeze(0).to(torch.int64), num_classes=prediction.shape[0]).permute(2,0,1).float()
        prediction = (prediction >= 0.5).float()
        target = target.to(prediction.device.index)

        d = []
        for c in range(1, int(prediction.shape[0])):
            t = target[c]
            p = prediction[c]
            inter = torch.dot(p.view(-1), t.contiguous().view(-1))
            union_sum = torch.sum(p) + torch.sum(t)
            union = union_sum + self.smooth
            t = (2 * inter.float() + self.smooth) / union.float()

            if union_sum.item() == 0:
                d.append(1.0)
            else:
                d.append(t.item())

        return sum(d)/len(d) if len(d)>0 else 1


class MultiClassDiceCoeff:
    def __init__(self, eps=1e-8):
        self.smooth = eps

    def calculate(self, target, prediction, apply_nonlin=True):
        num_classes = prediction.shape[1]
        if apply_nonlin:
            prediction = torch.softmax(prediction, 1)
        output_seg = prediction.argmax(1)
        target = target[:, 0]
        axes = tuple(range(1, len(target.shape)))
        target = target.to(output_seg.device.index)
        tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
            fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
            fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

        tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

        union = 2 * tp_hard + fp_hard + fn_hard 
        soft_dice = (2 * tp_hard) / (union + self.smooth)

        for i, un in enumerate(union):
            if un == 0:
                soft_dice[i] = 1.0
        
        return soft_dice, tp_hard, fp_hard, fn_hard
    

class SurfaceDice:
    def __init__(self, tolerance = 2, spacing = (0.6, 0.6)):
        self.tolerance = tolerance
        self.spacing = spacing
        
    def calculate(self, target, prediction, apply_nonlin=True):
        axis = 0 if len(prediction.shape) == 3 else 1

        if apply_nonlin:
            prediction = torch.softmax(prediction, axis)
        
        target = F.one_hot(target.squeeze(0).to(torch.int64), num_classes=prediction.shape[0]).permute(2,0,1).cpu().numpy()
        prediction = (prediction >= 0.5).cpu().numpy()

        if np.sum(target)+np.sum(prediction) == 0.0:
            return 1.0
        
        d = []
        for c in range(1, int(prediction.shape[0])):
            t = (target[c]).astype(np.bool)
            p = (prediction[c]).astype(np.bool)
            if np.sum(t)+np.sum(p) == 0.0:
                d.append(1.0)
            else:
                surface_distances = surface_distance.compute_surface_distances(t, p, spacing_mm=self.spacing)
                d.append(surface_distance.compute_surface_dice_at_tolerance(surface_distances, self.tolerance))
        return sum(d)/len(d) if len(d)>0 else 1


class HausdorffDistance:
    def __init__(self, percent=95, spacing = (0.6, 0.6)):
        self.percent = percent
        self.spacing = spacing
        
    def calculate(self, target, prediction, apply_nonlin=True):
        axis = 0 if len(prediction.shape) == 3 else 1

        if apply_nonlin:
            prediction = torch.softmax(prediction, axis)
        
        target = F.one_hot(target.squeeze(0).to(torch.int64), num_classes=prediction.shape[0]).permute(2,0,1).cpu().numpy()
        prediction = (prediction >= 0.5).cpu().numpy()

        if np.sum(target)+np.sum(prediction) == 0.0:
            return 0.0
        
        d = []
        for c in range(1, int(prediction.shape[0])):
            t = (target[c]).astype(np.bool)
            p = (prediction[c]).astype(np.bool)
            if np.sum(t)+np.sum(p) == 0.0:
                d.append(0)
            else:
                surface_distances = surface_distance.compute_surface_distances(t, p, spacing_mm=self.spacing)
                d.append(surface_distance.compute_robust_hausdorff(surface_distances, self.percent))
        return max(d) if len(d)>0 else 0