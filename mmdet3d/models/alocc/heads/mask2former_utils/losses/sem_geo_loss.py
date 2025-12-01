import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from mmcv.runner import BaseModule, force_fp32
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

from torch.cuda.amp import autocast
from torch.autograd import Variable


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)


def inverse_sigmoid(x, sign='A'):
    x = x.to(torch.float32)
    while x >= 1-1e-5:
        x = x - 1e-5

    while x< 1e-5:
        x = x + 1e-5

    return -torch.log((1 / x) - 1)

def sem_scal_loss(pred, ssc_target, ignore_index=255):
    # Get softmax probabilities
    with autocast(False):
        # pred = F.softmax(pred_, dim=1)
        loss = 0
        count = 0
        # mask = ssc_target != ignore_index

        n_classes = pred.shape[1]
        # begin = 1 if n_classes == 19 else 0
        for i in range(0, n_classes-1):   

            # Get probability of class i
            p = pred[:, i]  

            # Remove unknown voxels
            target = ssc_target
            # p = p[mask]
            # target = ssc_target[mask]   

            # completion_target = torch.ones_like(target)
            # completion_target[target != i] = 0
            # completion_target = torch.ones_like(target).float()
            # completion_target[target != i] = 0
            completion_target=target[:,i]

            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p)+ 1e-5)
                    loss_precision = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(precision, 'D'), torch.ones_like(precision)
                        )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target) +1e-5)
                    # loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))

                    loss_recall = F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'E'), torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target) +  1e-5
                    )

                    loss_specificity = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(specificity, 'F'), torch.ones_like(specificity)
                        )
                    loss_class += loss_specificity
                loss += loss_class
                # print(i, loss_class, loss_recall, loss_specificity)
        l = loss/count
        if torch.isnan(l):
            from IPython import embed
            embed()
            exit()
        return l


def geo_scal_loss(pred, ssc_target, ignore_index=255, non_empty_idx=0):

    # Get softmax probabilities
    # pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred#[:, -1]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    # import pdb; pdb.set_trace()
    # mask = ssc_target != ignore_index
    # nonempty_target = ssc_target != non_empty_idx
    # nonempty_target = nonempty_target[mask].float()
    # nonempty_probs = nonempty_probs[mask]
    # empty_probs = empty_probs[mask]
    nonempty_target=(1-ssc_target).float()

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum()+eps)
    recall = intersection / (nonempty_target.sum()+eps)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
    with autocast(False):
        return (
            F.binary_cross_entropy_with_logits(inverse_sigmoid(precision, 'A'), torch.ones_like(precision))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'B'), torch.ones_like(recall))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(spec, 'C'), torch.ones_like(spec))
        )



def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        with autocast(False):
            loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    # import pdb; pdb.set_trace()
    for c in class_to_sum:
        fg = labels[:,c].float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(class_to_sum) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 2:
        # import pdb; pdb.set_trace()
        if ignore is not None:
            valid = (labels != ignore)
            probas = probas[valid]
            labels = labels[valid]
        return probas, labels

    elif probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        #3D segmentation
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H*W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
