import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftDiceLossMultiClass(nn.Module):
    def __init__(self, apply_nonlin=F.softmax, batch_dice=True, do_bg=False, smooth=1.):
        """
        """
        super(SoftDiceLossMultiClass, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x, dim=1)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (denominator + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        smooth = 1e-5

        probs = torch.sigmoid(probs)

        m1 = probs.view(-1)
        m2 = targets.view(-1)
        intersection = torch.dot(m1, m2)
        union = torch.sum(m1) + torch.sum(m2)
        
        score = (2. * intersection + smooth) / (union + smooth)
        score = -score
        return score

class BCELoss3d(nn.Module):
    def __init__(self):
        super(BCELoss3d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, probs, targets):
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)

class CombinedLoss(nn.Module):
    def __init__(self, is_dice_log):
        super(CombinedLoss, self).__init__()
        self.is_dice_log = is_dice_log
        self.bce = BCELoss3d()
        self.soft_dice = SoftDiceLoss()

    def forward(self, probs, targets):

        bce_loss = self.bce(probs, targets)
        dice_loss = self.soft_dice(probs, targets)

        if self.is_dice_log:
            l = bce_loss - torch.log(-dice_loss)
        else:
            l = bce_loss + dice_loss

        return l, bce_loss, dice_loss

class DC_and_CE_loss(nn.Module):
    def __init__(self, weight_ce=1, weight_dice=1, log_dice=False):
        super(DC_and_CE_loss, self).__init__()

        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = nn.CrossEntropyLoss()
        self.dc = SoftDiceLossMultiClass()

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0 
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss

        return result

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn




