import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import argparse
import pandas as pd
import glob
import os
import evaluation_utils
import STnetwork
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
    net = STnetwork.STNet(network_config)
    checkpoint = torch.load(net_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print('Model is saved on epoch: ', epoch)
    net.to(device)
    net.eval()

    data = pd.DataFrame()
    if save_img and 'imgs' not in os.listdir(path):
        os.mkdir(path + '/imgs')
    eval_STnet(data_evaluate, net, data)


def eval_STnet(data_evaluate, net, data):
    for v in range(data_evaluate.__len__()):
        video = data_evaluate.dataset.videos[v]
        mouse = data_evaluate.dataset.mice[v][0]
        if save_img and mouse not in os.listdir(path + '/imgs'):
            os.mkdir(path + '/imgs/' + mouse)
        for q_sample in range(1, video['ct'].shape[0]):
            query = video['ct'][[q_sample], :, :, :].unsqueeze_(0).to(device)
            ground_truth = video['mask'][[q_sample], :, :, :]
            for m_sample in range(q_sample):
                net.memoryUnit.reset_memory()
                prev_sample = [m_sample]
                try:
                    add_timepoint = np.random.choice(np.arange(m_sample),
                                                     training_config['training_mode']['n_samples']-2,
                                                     replace=False)
                    prev_sample += [point for point in add_timepoint]
                    prev_sample.sort()
                except ValueError:
                    continue
                previous_data = torch.stack((video['ct'][prev_sample, :, :, :],
                                             video['mask'][prev_sample, :, :, :]),
                                            dim=1).unsqueeze_(0).to(device)
                day_m = int(video['ids'][m_sample])
                print('To memory: ', mouse, day_m)
                m_tumor_volume = evaluation_utils.get_relative_volume(video['mask'][m_sample, :, :, :])
                print('Memory Tumor rel volume: ', float(m_tumor_volume))

                output = net(query=query, previous_data=previous_data)
                output = output[:, 0, :, :, :]

                roc_auc = evaluation_utils.get_roc_auc(ground_truth, output)
                precission_recall_auc = evaluation_utils.get_precission_recall_auc(ground_truth, output)

                t = 0.5  # threshold
                output = (output > t).float()
                output = output.detach().cpu()

                target_coord, prediction_coord = evaluation_utils.convert_to_coordinates(target=ground_truth,
                                                                                         prediction=output)
                day_q = int(video['ids'][q_sample])
                print('Query: ', mouse, day_q)

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
                msd = None #evaluation_utils.mean_surface_distance(target_coord, prediction_coord)
                print('MSD: ', msd)
                print()

                data = data.append({
                    'mouse_m': mouse,
                    'day_m': day_m,
                    'mouse_q': mouse,
                    'day_q': day_q,
                    'm_tumor_v': float(m_tumor_volume),
                    'q_tumor_v': float(tumor_volume),
                    'pred_v': float(prediction_volume),
                    'dice_coef': float(dice_coef),
                    'hausdorff': hausdorff,
                    'jaccard': float(jaccard_coef),
                    'roc_auc': roc_auc,
                    'precission_recall_auc': precission_recall_auc,
                    'msd': msd
                }, ignore_index=True)

                if save_img:
                    title = str(int(video['ids'][q_sample]))
                    to_memory = '_' + str(day_m)
                    evaluation_utils.save_images(output, path + '/imgs/' + mouse, title, to_memory)

    # data.to_excel(path + '/' + 'eval_results_STNet_2.xlsx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config-files', nargs='+', required=True)
    parser.add_argument('-path', required=True)
    parser.add_argument('-save-img', required=False, default=True)


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