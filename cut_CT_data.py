import pydicom
import numpy as np
from scipy import ndimage as ndi
import zipfile
import argparse
import glob
import os
import shutil
from matplotlib import pyplot as plt
from dicom_contour.contour import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data-folder', type=str, required=False,
                        default='D://masterThesis/Data/group2',
                        help='folder with zip files')
    parser.add_argument('-initial-size', required=False, type=tuple, default=(412, 412),
                        help='initial size of data (high, width, depth)')
    parser.add_argument('-target-size', required=False, type=tuple, default=(128, 128, 64),
                        help='target size of data (high, width, depth)')
    parser.add_argument('-cut-coordinates', required=False, type=tuple,
                        default=((250, 378), (150, 278), (80, 144)),
                        help='cut coordinates for images')
    parser.add_argument('-coregister', required=False, type=bool,
                        default=False,
                        help='coregister wrt mask')
    parser.add_argument('-folder-to-save', required=False, type=str, default='D://masterThesis/mice_data_coregistered/')

    arguments = parser.parse_args()
    data_folder = arguments.data_folder
    target_size = arguments.target_size
    cut_coord = arguments.cut_coordinates
    coregister = arguments.coregister

    folder_to_save = parser.parse_args().folder_to_save

    for subject_folder in glob.glob(data_folder+'/*'):
        subject = subject_folder.split('\\')[-1]
        zip_files = glob.glob(subject_folder+'/*zip')
        print(subject)
        # os.mkdir(folder_to_save+subject+'/')
        for file in zip_files:
            day = get_day(file)
            print(day)
            path_to_temp_file = unzip(file, subject_folder)
            path_to_dcm = glob.glob(path_to_temp_file + '/*')[0]
            contour_file = find_contour_file(path_to_dcm)
            image_contour = map_contour_to_ct(contour_file, path_to_dcm, cut_coord)

            image_contour = cut_image(image_contour, cut_coord)
            image_contour = process_in_coronal(image_contour)
            image_contour = process_in_sagittal(image_contour)
            image_contour = process_in_axial(image_contour)

            if coregister:
                print('Coregistration wrt mask')
                cut_coord = find_coregistered_coord(image_contour[:, :, :, 1])
            assert image_contour.shape == (128, 128, 64, 2)
            show_all_slices(image_contour, subject, day)
            # np.save(folder_to_save+subject+'/'+day, image_contour)
            delete_temp(path_to_temp_file)


def find_coregistered_coord(contour):
    center = find_center(contour)
    d_coord = (center[2]-32, center[2]+32)
    h_coord = (center[0]-64, center[0]+64)
    w_coord = (center[1]-64, center[1]+64)
    assert d_coord[0] > 0
    assert h_coord[0] > 0
    assert w_coord[0] > 0
    assert d_coord[1] < contour.shape[2]
    assert h_coord[1] < contour.shape[0]
    assert w_coord[1] < contour.shape[1]
    return h_coord, w_coord, d_coord


def find_center(contour):
    mask_coord = np.where(contour == 1)
    d_center = np.min(mask_coord[2])+(np.max(mask_coord[2]) - np.min(mask_coord[2]))//2
    contour_2d = contour[:, :, d_center]
    if 1 not in contour_2d:
        contour_2d = contour[:, :, np.min(mask_coord[2])]
    mask_coord_2d = np.where(contour_2d == 1)
    h_center = np.min(mask_coord_2d[0]) + (np.max(mask_coord_2d[0]) - np.min(mask_coord_2d[0]))//2
    w_center = np.min(mask_coord_2d[1]) + (np.max(mask_coord_2d[1]) - np.min(mask_coord_2d[1]))//2
    return h_center, w_center, d_center


def unzip(file, data_folder):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(data_folder + '/temp')
    return data_folder + '/temp'


def get_day(file):
    day_mark = file.find('d')
    day = file[day_mark:day_mark + 3]
    try:
        int(day[-1])
    except ValueError:
        day = day[:-1]
    return day


def find_contour_file(path):
    contour_file = [file for file in glob.glob(path+'/*dcm')
                    if 'RS' in file]
    return contour_file[0]


def find_ct_images(path):
    return [file for file in glob.glob(path) if 'CT' in file]


def map_contour_to_ct(contour_file, path, cut_coord):
    path = path + '/*dcm'
    contours = get_contour_data(contour_file)
    ct_images = find_ct_images(path)
    initial_size = get_size(ct_images[0])
    print('Size: ', (*initial_size, len(ct_images)))
    ct_and_masks = np.empty((*initial_size, len(ct_images), 2))
    for i, image_file in enumerate(ct_images):
        image = pydicom.read_file(image_file)
        image_ind = image.InstanceNumber
        ct_and_masks[:, :, image_ind, 0] = image.pixel_array
        image_id = image_file.partition('CT.')[-1].partition('.dcm')[0]
        if image_id in contours.keys():
            if image_ind > cut_coord[2][1] or image_ind < cut_coord[2][0]:
                print('You will cut the mask: ', image_ind)
            ct_and_masks[:, :, image_ind, 1] = put_contour(contours[image_id], image, initial_size)
        else:
            ct_and_masks[:, :, image_ind, 1] = np.zeros(initial_size[:2])
    return ct_and_masks


def process_in_axial(image_contour):
    mask = image_contour[:, :, :, 1]
    for s in range(mask.shape[0]):
        mask_s = mask[s, :, :]
        interpol_mask = ndi.binary_closing(mask_s, structure=np.ones((4, 4)), brute_force=True).astype(np.int)
        filled_mask = ndi.binary_fill_holes(interpol_mask).astype(np.int)
        image_contour[s, :, :, 1] = filled_mask
    return image_contour


def process_in_sagittal(image_contour):
    mask = image_contour[:, :, :, 1]
    for s in range(mask.shape[1]):
        mask_s = mask[:, s, :]
        interpol_mask = ndi.binary_closing(mask_s, structure=np.ones((4, 4)), brute_force=True).astype(np.int)
        filled_mask = ndi.binary_fill_holes(interpol_mask).astype(np.int)
        image_contour[:, s, :, 1] = filled_mask
    return image_contour


def process_in_coronal(image_contour):
    mask = image_contour[:, :, :, 1]
    for s in range(mask.shape[2]):
        mask_s = mask[:, :, s]
        interpol_mask = ndi.binary_closing(mask_s, structure=np.ones((4, 4)), brute_force=True).astype(np.int)
        filled_mask = ndi.binary_fill_holes(interpol_mask, structure=np.ones((4, 4))).astype(np.int)
        filled_mask = fill_contour(filled_mask)
        filled_mask = construct_mask(filled_mask, (mask_s.shape[0], mask_s.shape[1]))
        image_contour[:, :, s, 1] = filled_mask
    return image_contour


def get_size(image_file):
    image = pydicom.read_file(image_file)
    image = image.pixel_array
    image_size = (image.shape[0], image.shape[1])
    return image_size


def put_contour(contour, image, initial_size):
    pixel_coord = convert_to_pixels(contour, image)
    mask = construct_mask(pixel_coord, initial_size)
    return mask



def convert_to_pixels(contour, image):
    coord = []
    for i in range(0, len(contour), 3):
        coord.append((contour[i], contour[i+1], contour[i+2]))
    x_spacing, z_spacing = float(image.PixelSpacing[0]), float(image.PixelSpacing[1])
    origin_x, _, origin_z = image.ImagePositionPatient
    pixel_coord = [(np.rint(abs(z - origin_z) / z_spacing).astype(int), np.rint((x - origin_x) / x_spacing).astype(int))
                   for x, _, z in coord]
    return pixel_coord


def fill_contour(mask, fill_vertically=True, fill_horizontally=True):
    pixel_coord = [(x, y) for (x, y) in zip(np.where(mask == 1)[0], np.where(mask == 1)[1])]
    if fill_vertically:
        x_coord = np.unique([coord[1] for coord in pixel_coord])
        for x in x_coord:
            row_coord = np.array([coord[0] for coord in pixel_coord if coord[1] == x])
            row_coord = np.sort(row_coord)
            for i in range(len(row_coord) - 1):
                if row_coord[i + 1] - row_coord[i] > 1:
                    for y in range(row_coord[i], row_coord[i + 1]):
                        if (y, x) not in pixel_coord:
                            pixel_coord.append((y, x))

    if fill_horizontally:
        y_coord = np.unique([coord[0] for coord in pixel_coord])
        for y in y_coord:
            row_coord = np.array([coord[1] for coord in pixel_coord if coord[0] == y])
            row_coord = np.sort(row_coord)
            for i in range(len(row_coord)-1):
                if row_coord[i+1] - row_coord[i] > 1:
                    for x in range(row_coord[i], row_coord[i+1]):
                        if (y, x) not in pixel_coord:
                            pixel_coord.append((y, x))

    return pixel_coord


def construct_mask(pixel_coord, size):
    mask = np.zeros(size)
    for (x, y) in pixel_coord:
        mask[x, y] = 1
    return mask


def get_contour_data(contour_file):
    contour_file = pydicom.read_file(contour_file)
    rtv = contour_file.ROIContourSequence[0]
    contours = {}
    for contour in rtv.ContourSequence:
        contours[contour.ContourImageSequence[0].ReferencedSOPInstanceUID] = contour.ContourData
    return contours


def cut_image(image_contour, cut_coord):
    x = cut_coord[0]
    y = cut_coord[1]
    depth = cut_coord[2]
    return image_contour[x[0]:x[1], y[0]:y[1], depth[0]:depth[1], :]


def delete_temp(data_folder):
    shutil.rmtree(data_folder)



############## Vizualization ###########################

def show_contour_slices(image_contour, subject, day):
    fig = plt.figure(figsize=(50, 100))
    next_subplot = 1
    for i in range(image_contour.shape[2]):
        if 1 in image_contour[:, :, i, 1]:
            ax = fig.add_subplot(3, 10, next_subplot)
            ax.imshow(image_contour[150:278, 136:264, i, 0]+image_contour[150:278, 136:264, i, 1]*10000,
                      cmap=plt.cm.bone, aspect='auto')
            next_subplot += 1
    fig.suptitle(str(subject) + ', day ' + str(day))
    plt.show()


def plot_sample(initial, interpolated, mask, title):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(initial, cmap=plt.cm.bone, aspect='auto')
    ax1.set_title('Initial')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(interpolated, cmap=plt.cm.bone, aspect='auto')
    ax2.set_title('Interpolated')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(mask, cmap=plt.cm.bone, aspect='auto')
    ax3.set_title('Filled')

    plt.show()


def show_all_slices(image_contour, subject, day):
    fig = plt.figure()
    subplot = 1
    for i in range(image_contour.shape[2]):
        ax = fig.add_subplot(6, 11, subplot)
        ax.imshow(image_contour[:, :, i, 0]+image_contour[:, :, i, 1]*10000, cmap='gray', aspect='auto')
        fig.suptitle(str(subject) + ', day ' + str(day))
        subplot += 1
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    plt.show()

    fig = plt.figure()
    subplot = 1
    for i in range(image_contour.shape[1]):
        if 1 in image_contour[:, i, :, 1]:
            ax = fig.add_subplot(6, 10, subplot)
            ax.imshow(image_contour[:, i, :, 0] + image_contour[:, i, :, 1] * 10000, cmap='gray', aspect='auto')
            fig.suptitle(str(subject) + ', day ' + str(day))
            subplot += 1
            ax.set_xticks([], [])
            ax.set_yticks([], [])
    plt.show()

    fig = plt.figure()
    subplot = 1
    for i in range(image_contour.shape[0]):
        if 1 in image_contour[i, :, :, 1]:
            ax = fig.add_subplot(6, 10, subplot)
            ax.imshow(image_contour[i, :, :, 0] + image_contour[i, :, :, 1] * 10000, cmap='gray', aspect='auto')
            fig.suptitle(str(subject) + ', day ' + str(day))
            subplot += 1
            ax.set_xticks([], [])
            ax.set_yticks([], [])
    plt.show()

def show_slice_and_mask(ct, mask, title):
    fig = plt.figure(figsize=(10, 10))
    ax_ct = fig.add_subplot(1, 2, 1)
    ax_ct.imshow(ct+mask*10000, cmap=plt.cm.bone, aspect='auto')
    # ax_mask = fig.add_subplot(1, 2, 2)
    # ax_mask.imshow(mask, cmap=plt.cm.bone, aspect='auto')
    # ax_ct.hlines(150, xmin=150, xmax=278, colors='r')
    # ax_ct.hlines(278, xmin=150, xmax=278, colors='r')
    # ax_ct.vlines(150, ymin=150, ymax=278, colors='r')
    # ax_ct.vlines(278, ymin=150, ymax=278, colors='r')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    main()