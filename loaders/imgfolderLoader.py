import glob
import random
import os
import sys
ROOT = r"E:\ZimengZhao_Program\RebuidZoo\ConvDetect" #os.getcwd()
sys.path.append(ROOT)
import custom_utils.parser as ut_prs

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class_filename = os.path.join(ROOT, r"official_yolo_files\data_names\voc.names") 

def view_tensor(p_img_Tsor):
    # p_img_Tsor = p_img_Tsor / 2 + 0.5     # unnormalize
    img_Arr = p_img_Tsor.numpy()
    plt.imshow(np.transpose(img_Arr, (1, 2, 0)))
    plt.show()

def view_tensor_withBBX(img_Tsor_batch, tar_Tsor_batch, n_rows, return_img = False):
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    bbox_colors = random.sample(colors, 20)

    classname_Lst = ut_prs.load_classes(class_filename)
    netin_size = img_Tsor_batch.shape[-1]
    

    img_idx_batch = tar_Tsor_batch[: , 0].int()
    cls_idx_batch = tar_Tsor_batch[: , 1].int()
    x_c_batch = (netin_size * tar_Tsor_batch[: , 2]).int(); 
    y_c_batch = (netin_size * tar_Tsor_batch[: , 3]).int(); 
    box_w_batch = (netin_size * tar_Tsor_batch[: , 4]).int(); 
    box_h_batch = (netin_size * tar_Tsor_batch[: , 5]).int(); 

    x1_batch = x_c_batch - box_w_batch // 2
    y1_batch = y_c_batch - box_h_batch // 2
    n_cols = img_Tsor_batch.shape[0] // n_rows
    
    fig = plt.figure(figsize= [n_rows*3 ,n_rows*3]) # (default: [6.4, 4.8])
    tar_idx = 0
    for img_idx, img_Tsor in enumerate(img_Tsor_batch):
        img_Arr = img_Tsor.numpy()
        img_Arr = np.transpose(img_Arr, (1, 2, 0))
        ax = fig.add_subplot(n_rows, n_cols, img_idx+1)
        ax.imshow(img_Arr)
        plt.axis('off')

        while img_idx_batch[tar_idx] == img_idx:
            cls_gt = cls_idx_batch[tar_idx]
            color = bbox_colors[cls_gt]
            x1 = x1_batch[tar_idx]; y1 = y1_batch[tar_idx]
            bbox = patches.Rectangle((x1, y1), box_w_batch[tar_idx], box_h_batch[tar_idx], 
                linewidth=2, edgecolor = bbox_colors[cls_gt], facecolor="none")
            ax.add_patch(bbox)
            plt.text(x1, y1,
                    fontsize=12, 
                    s = classname_Lst[cls_gt],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
            tar_idx += 1
            if tar_idx == len(tar_Tsor_batch) : break
    plt.tight_layout(h_pad = 0.1, w_pad= 0.00)
    if not return_img:
        plt.show()
    else : 
        fig.canvas.draw()
        img_fig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img_fig = img_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_fig = img_fig / 255.0
        plt.close(fig)
        # plt.imshow(img_fig)
        # plt.show()
        return img_fig



def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

class ImageFolderLoader(torch.utils.data.Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

class ListDataLoader(torch.utils.data.Dataset):
    '''
    Deal with the dataset in COCO uniform.
    For VOC, download script https://pjreddie.com/media/files/voc_label.py and 
    put into where 'VOCdevkit' lies(/VOC2012/JPEGImages, Annotations, ...)
    '''
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        
        self.label_files = [
            path.replace("images", "labels").replace("JPEGImages", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes # **place 0 are padded by collate_fn in dataloader with batch index

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


def check_ImageFolderLoader():
    imgs_dir = os.path.join( r"E:\ZimengZhao_Program\RebuidZoo\ConvDetect", "test_samples", "yolo_samples")
    g_dataset = ImageFolderLoader(imgs_dir, 416)

    g_testloader = torch.utils.data.DataLoader(dataset = g_dataset, batch_size= 16,shuffle= True,num_workers= 1) 
    for batch_i, (img_paths, img_Tsor_batch_i) in enumerate(g_testloader):
        view_Tsor = torchvision.utils.make_grid(tensor = img_Tsor_batch_i, nrow= 4) # (3, H, W)
        view_tensor(view_Tsor)

def check_ListDataLoader():
    listfile_path = r"F:\ZimengZhao_Data\VOC2012\VOCtrainval_11-May-2012\2012_val.txt" # train
    g_dataset = ListDataLoader(list_path = listfile_path, img_size = 416, multiscale = False)
    g_testloader = torch.utils.data.DataLoader(dataset = g_dataset, batch_size= 4, shuffle= False,num_workers= 1, collate_fn = g_dataset.collate_fn) 
    for batch_i, (img_paths, img_Tsor_batch_i, tar_Tsor_batch_i) in enumerate(g_testloader):
        print(img_Tsor_batch_i.shape)
        # print(tar_Tsor_batch_i)
        # view_Tsor = torchvision.utils.make_grid(tensor = img_Tsor_batch_i, nrow= 4) # (3, H, W)
        # view_tensor(view_Tsor)
        view_tensor_withBBX(img_Tsor_batch_i, tar_Tsor_batch_i, 2)
        

if __name__ == "__main__":
    
    # check_ImageFolderLoader()
    check_ListDataLoader()
