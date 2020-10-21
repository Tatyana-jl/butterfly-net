import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from numpy import linalg as LA
from sklearn import metrics



def get_roc_auc(target, prediction):
    y_true = target.view(-1).numpy()
    y_score = prediction.view(-1).cpu().detach().numpy()
    roc_auc_score = metrics.roc_auc_score(y_true, y_score)
    return roc_auc_score


def get_precission_recall_auc(target, prediction):
    y_true = target.view(-1).numpy()
    y_score = prediction.view(-1).cpu().detach().numpy()
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    precission_recall_auc = metrics.auc(recall, precision)
    return precission_recall_auc



def dice_coef(target, prediction):
    pred_flat = prediction.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat + target_flat)
    coef = (2 * intersection) / union
    return coef


def hausdorff_distance(target_coord, prediction_coord):
    if len(prediction_coord) >= 1:
        hausdorff_distance = max(directed_hausdorff(target_coord, prediction_coord)[0], directed_hausdorff(prediction_coord, target_coord)[0])
    else:
        hausdorff_distance = None
    return hausdorff_distance


def jaccard_coef(target_fg, prediction_fg):
    intersection = torch.sum(prediction_fg * target_fg)
    union = torch.sum(prediction_fg + target_fg)
    coef_fg = intersection/(union - intersection)
    return coef_fg

# def gt_hot_encoding(ground_truth):
#     return (ground_truth-1)*-1


def mean_surface_distance(target_coord, prediction_coord):
    surface_sum_distance = 0
    if len(prediction_coord) != 0:
        for point in target_coord:
            min_distances = min([LA.norm(coord) for coord in np.array(point)-np.array(prediction_coord)])
            surface_sum_distance += min_distances
        for point in prediction_coord:
            min_distances = min([LA.norm(coord) for coord in np.array(point) - np.array(target_coord)])
            surface_sum_distance += min_distances
        return surface_sum_distance/(len(target_coord) + len(prediction_coord))
    else:
        return None


def convert_to_coordinates(target, prediction):
    target = target.squeeze_(0).numpy()
    prediction = prediction.squeeze_(0).numpy()

    target_coord = [(x, y, z) for x, y, z in zip(np.where(target==1)[0], np.where(target==1)[1], np.where(target==1)[2])]
    prediction_coord = [(x, y, z) for x, y, z in zip(np.where(prediction==1)[0], np.where(prediction==1)[1], np.where(prediction==1)[2])]
    return target_coord, prediction_coord


def get_relative_volume(mask):
    relative_volume = 100 * torch.sum(mask)/mask.numel()
    return relative_volume


def plot_ct_and_mask(query, mask, pred, title, path):
    query = query.squeeze_(0).squeeze_(0).detach().cpu()
    pred = pred.squeeze_(0)
    fig1 = plt.figure()
    fig2 = plt.figure()
    subplot_1 = 1
    subplot_2 = 1

    slices = np.random.choice(np.arange(query.shape[2]), 5)
    for i in slices:
        fig = plt.figure(figsize=(10, 10))
        ax_gt = fig.add_subplot(1, 2, 1)
        ax_gt.imshow(query[:, :, i] + mask[:, :, i] * 5, cmap=plt.cm.bone, aspect='auto')
        ax_pred = fig.add_subplot(1, 2, 2)
        ax_pred.imshow(query[:, :, i] + pred[:, :, i] * 5, cmap=plt.cm.bone, aspect='auto')
        plt.title(title + ' slice ' + str(i))
        plt.savefig(path + '/' + title + ' slice ' + str(i) + '.png')


def save_images(prediction, path, title, in_memory=None):
    prediction = prediction.squeeze_(0)
    if in_memory is not None:
        title = title + in_memory
    np.save(path + '/' + title, prediction)
