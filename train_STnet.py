import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import argparse
import glob
from tqdm import tqdm

import STnetwork
import utils


def load_config_files():
    files = {}
    for file in config_files:
        with open(file) as f:
            files[file.split('.')[0]] = yaml.load(f)
    return files['data_config'], files['network_config'], files['training_config']


def main():
    # set data loader
    data_train = DataLoader(
        utils.CTandContourLoader(data_config,
                                 dataset_type='train',
                                 n_samples=training_config['training_mode']['n_samples']),
        batch_size=1,
        shuffle=True
    )
    data_validate = DataLoader(
        utils.CTandContourLoader(data_config,
                                 dataset_type='validate',
                                 n_samples=training_config['training_mode']['n_samples']),
        batch_size=1,
        shuffle=True
    )
    assert data_train.__len__() > 0, 'data folder (train) is empty, check the path'
    assert data_validate.__len__() > 0, 'data folder (validate) is empty, check the path'

    net = STnetwork.STNet(network_config)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=training_config['optimizer']['start_learning_rate'])

    if cntd:
        net, optimizer, start_epoch, loss_train, loss_val = continue_training(net, optimizer)
    else:
        net, start_epoch, loss_train, loss_val = init_new(net)

    loss_fnct = utils.dice_loss

    net.to(device)

    for name, param in net.named_parameters():
        if param.device.type != 'cuda':
            print('param {}, not on GPU'.format(name))

    train(data_train=data_train,
          data_validate=data_validate,
          start_epoch=start_epoch,
          optimizer=optimizer,
          net=net,
          loss_fnct=loss_fnct,
          loss_train=loss_train,
          loss_val=loss_val,
          path=save_folder)


def continue_training(net, optimizer):
    net_file = glob.glob(save_folder + '/*pt')[0]
    checkpoint = torch.load(net_file, map_location='cuda:0')
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print('Model is saved on epoch: ', start_epoch)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    loss_train = np.load(save_folder + '/train_loss.npy')
    loss_val = np.load(save_folder + '/val_loss.npy')
    return net, optimizer, start_epoch, loss_train[:start_epoch+1], loss_val[:start_epoch+1]


def init_new(net):
    start_epoch = -1
    net.apply(weights_init)
    loss_train = np.array([])
    loss_val = np.array([])
    return net, start_epoch, loss_train, loss_val


def weights_init(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)


def train(data_train, data_validate, start_epoch, net, optimizer, loss_fnct, loss_train, loss_val, path):

    for epoch in tqdm(range(start_epoch+1, training_config['n_epochs']+1)):
        loss_on_epoch = 0
        validation_loss = 0

        # TRAIN
        net.train()
        for i, (ct, mask, _) in enumerate(data_train):
            net.memoryUnit.reset_memory()
            assert ct.shape[1] >= 2, 'At least two frames are needed for the Memory Network'
            #TODO: implement n-size minibatch
            previous_data = torch.stack((ct[:, :-1, :, :, :],
                                         mask[:, :-1, :, :, :]), dim=2).to(device)
            query = ct[:, -1, :, :, :].unsqueeze_(1).to(device)
            ground_truth = utils.gt_hot_encoding(mask[:, -1, :, :, :]).to(device)
            optimizer.zero_grad()
            output = net(query=query, previous_data=previous_data)
            loss = loss_fnct(output, ground_truth)
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_on_epoch += loss.item()

        loss_train = np.append(loss_train, loss_on_epoch/data_train.__len__())

        # VALIDATE
        net.eval()
        for k, (ct, mask, mouse) in enumerate(data_validate):
            net.memoryUnit.reset_memory()
            previous_data = torch.stack((ct[:, :-1, :, :, :],
                                         mask[:, :-1, :, :, :]), dim=2).to(device)
            query = ct[:, -1, :, :, :].unsqueeze_(1).to(device)
            ground_truth = utils.gt_hot_encoding(mask[:, -1, :, :, :]).to(device)
            output = net(query=query, previous_data=previous_data)
            loss = loss_fnct(output, ground_truth, weight=training_config['loss']['weight'])
            validation_loss += loss.item()

        loss_val = np.append(loss_val, validation_loss/data_validate.__len__())

        if epoch % training_config['check_every_x_epoch'] == 0:
            utils.checkpoint_save(epoch, net, optimizer, save_folder, net_name='ST_net')
            utils.checkpoint_record_loss(loss_train, loss_val, path)

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config-files', nargs='+', required=True)
    parser.add_argument('-save-to', required=True)
    parser.add_argument('-cntd', type=bool, required=False, default=False)


    args = parser.parse_args()
    config_files = args.config_files
    save_folder = args.save_to
    cntd = args.cntd
    data_config, network_config, training_config = load_config_files()

    if torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device('cuda:0')
    else:
        print('Using CPU')
        device = torch.device('cpu')

    main()

