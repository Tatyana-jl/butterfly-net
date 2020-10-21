import glob
import torch
import numpy as np
import re
from torch.utils.data import Dataset


class CTandContourLoader(Dataset):

    def __init__(self, config, dataset_type, n_samples):
        list_of_videos = glob.glob(config['folder_'+dataset_type] + '/*')
        self.videos = {}
        self.mice = []
        for v, video in enumerate(list_of_videos):
            mouse_num = re.findall(r"Mouse\d{0,100}", video)
            self.mice.append(mouse_num)
            self.videos[v] = {}
            list_of_files = glob.glob(video + '/*npy')
            self.videos[v]['ct'] = torch.empty((len(list_of_files), *config['image_shape']))
            self.videos[v]['mask'] = torch.empty((len(list_of_files), *config['image_shape']))
            self.videos[v]['ids'] = []
            list_of_files = self.sort_in_order(list_of_files)
            for f, file in enumerate(list_of_files):
                day = np.int(re.findall(r"d(\d+)", file)[0])
                self.videos[v]['ids'].append(day)
                image_and_contour = np.load(file)
                ct = image_and_contour[:, :, :, 0]
                self.videos[v]['ct'][f, :, :, :] = torch.tensor((ct - np.min(ct))/(np.max(ct)-np.min(ct)))
                self.videos[v]['mask'][f, :, :, :] = torch.tensor(image_and_contour[:, :, :, 1])
        self.len = len(self.videos.keys())
        self.n_samples = n_samples

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        the_video = self.videos[idx]
        the_mouse = self.mice[idx]
        time_points = the_video['ct'].shape[0]
        samples = np.random.choice(np.arange(time_points), self.n_samples, replace=False)
        samples = np.sort(samples)
        ct = the_video['ct'][samples, :, :, :]
        mask = the_video['mask'][samples, :, :, :]
        return ct, mask, the_mouse


    def sort_in_order(self, list_of_files):
        time_points = {}
        files_sorted = []
        for file in list_of_files:
            day = np.int(re.findall(r"d(\d+)", file)[0])
            time_points[day] = file
        for key in sorted(time_points):
            files_sorted.append(time_points[key])
        return files_sorted

    # def get_samples_for_Vnet(self, video):
    #     time_points = video['ct'].shape[1]
    #     samples = np.random.choice(np.arange(time_points), 1)
    #     return samples
    #
    # def get_samples_for_STnet(video, n_samples):
    #     time_points = video['ct'].shape[1]
    #     samples = np.random.choice(np.arange(time_points), n_samples, replace=False)
    #     samples = np.sort(samples)
    #     return samples[0:-1], samples[-1]

    # def get_previous_data(video, samples):
    #     previous_data = torch.stack((video['ct'][0, samples, :, :, :],
    #                                  video['mask'][0, samples, :, :, :]),
    #                                 ).unsqueeze_(0).cuda()
    #     return previous_data

    # def get_query_and_ground_truth(video, sample):
    #     query = video['ct'][:, sample, :, :, :].unsqueeze_(0).cuda()
    #     ground_truth = video['mask'][:, sample, :, :, :].cuda()
    #     ground_truth = gt_hot_encoding(ground_truth)
    #     return query, ground_truth

def gt_hot_encoding(ground_truth):
    return torch.cat((ground_truth, (ground_truth - 1) * -1), dim=1)


def dice_loss(prediction, target, weight=False):
    smooth = 1
    pred_flat = prediction.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = torch.sum(pred_flat*target_flat)
    union = torch.sum(pred_flat+target_flat)
    loss = torch.tensor([1]).cuda() - ((2*intersection)+smooth)/(union + smooth + 0.00001)
    if weight:
        weight = 1/(1+torch.sum(target_flat)/target_flat.numel())
        return loss*(1+weight)
    else:
        return loss


def get_data_description(data_config):
    data_description = {} # {mouse: {tumor_type: 0/1, irradiation: 0/1}}
    tumor_types = dict([(type_t, torch.tensor([i]).cuda()) for i, type_t in enumerate(data_config['tumor_type'].keys())])
    irradiation = dict([(irr, torch.tensor([i]).cuda()) for i, irr in enumerate(data_config['irradiation'].keys())])
    for mouse in data_config['mice']:
        data_description[mouse] = {
            'tumor_type': [tumor_types[type_t] for type_t in data_config['tumor_type'].keys() if mouse in data_config['tumor_type'][type_t]][0],
            'irradiation': [irradiation[irr] for irr in data_config['irradiation'].keys() if mouse in data_config['irradiation'][irr]][0]
        }
    return data_description


def get_clinical_factor(mice_data_description, mouse, day):
    mouse = np.int(re.findall(r"(\d+)", mouse)[0])
    factor = mice_data_description[mouse].copy()
    factor['day'] = torch.tensor([day]).cuda()
    return factor


def checkpoint_save(epoch, net, optimizer, path, net_name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path + '/' + net_name + '_model.pt')


def checkpoint_hist(writer, net, epoch):
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        writer.add_histogram(tag, value.data.cpu().numpy(), epoch + 1)
        writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)


def checkpoint_images(writer, query, ground_truth, output, epoch, step):
    writer.add_images('images_val_query' + str(epoch),
                      image_process(query.squeeze_(0)),
                      global_step=step,
                      dataformats='CHWN')

    writer.add_images('images_val_output_fg' + str(epoch),
                      image_process(output[:, 0, :, :, :]),
                      global_step=step,
                      dataformats='CHWN')

    writer.add_images('images_val_output_bg' + str(epoch),
                      image_process(output[:, 1, :, :, :]),
                      global_step=step,
                      dataformats='CHWN')


def checkpoint_record_loss(train_loss, val_loss, path):
    np.save(path+'/train_loss', train_loss)
    np.save(path+'/val_loss', val_loss)


def image_process(image):
    image = image.detach().cpu()
    return image



