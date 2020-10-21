import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import argparse
import pandas as pd
import glob
import os
import evaluation_utils
import Vnetwork
import utils


def load_config_files():
    files = {}
    for file in config_files:
        with open(file) as f:
            files[file.split('.')[0]] = yaml.load(f)
    return files['data_config'], files['network_config'], files['training_config']


def main():
    data_evaluate = DataLoader(
        utils.CTandContourLoader(data_config,
                                 dataset_type='validate',
                                 n_samples=training_config['training_mode']['n_samples']),
        batch_size=1
    )
    assert data_evaluate.__len__() > 0, 'data folder is empty, check the path'
    net_file = glob.glob(path + '/*.pt')[0]
    net = Vnetwork.VNet(network_config)
    checkpoint = torch.load(net_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print('Model is saved on epoch: ', epoch)
    net.to(device)
    net.eval()
    data = pd.DataFrame()
    if save_img and 'imgs' not in os.listdir(path):
        os.mkdir(path + '/imgs')
    eval_Vnet(data_evaluate, net, data)


def eval_Vnet(data_evaluate, net, data):
    for v in range(data_evaluate.__len__()):
        video = data.videos[v]
        mouse = data.mice[v][0]
        if save_img and mouse not in os.listdir(path + '/imgs'):
            os.mkdir(path + '/imgs/' + mouse)
        for sample in range(video['ct'].shape[1]):
            query = video['ct'][:, [sample], :, :, :].to(device)
            ground_truth = video['mask'][:, sample, :, :, :]

            output = net(query=query)
            output = output[:, 0, :, :, :]

            roc_auc = evaluation_utils.get_roc_auc(ground_truth, output)
            precission_recall_auc = evaluation_utils.get_precission_recall_auc(ground_truth, output)

            t = 0.5  # threshold
            output = (output > t).float()
            output = output.detach().cpu()

            target_coord, prediction_coord = evaluation_utils.convert_to_coordinates(target=ground_truth,
                                                                                     prediction=output)

            day = int(video['ids'][sample])
            print(mouse, day)

            tumor_volume = evaluation_utils.get_relative_volume(ground_truth)
            print('Tumor rel volume: ', float(tumor_volume))
            prediction_volume = evaluation_utils.get_relative_volume(output)
            print('Prediction rel volume: ', float(prediction_volume))
            dice_coef = evaluation_utils.dice_coef(target=ground_truth, prediction=output)
            print('Dice coef: ', float(dice_coef))
            hausdorff = evaluation_utils.hausdorff_distance(target_coord, prediction_coord)
            print('Hausdorf distance: ', hausdorff)
            jaccard_coef = evaluation_utils.jaccard_coef(target_fg=ground_truth,
                                                         prediction_fg=output)
            print('Jaccard coef: ', float(jaccard_coef))
            print('ROC AUC: ', roc_auc)
            print('Precissio-recall AUC: ', precission_recall_auc)
            msd = None #evaluation_utils.mean_surface_distance(target_coord, prediction_coord)
            print('MSD: ', msd)
            print()

            data = data.append({
                'mouse': mouse,
                'day': day,
                'tumor_v': float(tumor_volume),
                'pred_v': float(prediction_volume),
                'dice_coef': float(dice_coef),
                'hausdorff': hausdorff,
                'jaccard': float(jaccard_coef),
                'roc_auc': roc_auc,
                'precission_recall_auc': precission_recall_auc,
                'msd': msd
            }, ignore_index=True)

            if save_img:
                title = str(int(video['ids'][sample]))
                evaluation_utils.save_images(output, path + '/imgs/' + mouse, title)

    data.to_excel(path + '/' + 'eval_results_VNet.xlsx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config-files', nargs='+', required=True)
    parser.add_argument('-path', required=True)
    parser.add_argument('-save-img', required=False, type=bool, default=True)

    args = parser.parse_args()
    config_files = args.config_files
    path = args.path
    save_img = args.save_img

    data_config, network_config, training_config = load_config_files()

    if torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device('cuda:0')
    else:
        print('Using CPU')
        device = torch.device('cpu')

    main()