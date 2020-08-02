import time 
import os 
import sys 
ROOT = os.getcwd()
sys.path.append(ROOT)

import arch.darknet as dk_detect
import loaders.imgfolderLoader as fd_ld
import custom_utils.config as ut_cfg
import custom_utils.parser as ut_prs

import torch 
import torchvision 
from torch.utils.data import DataLoader

import numpy as np 
import random
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class YOLOdetect_config(ut_cfg.config):
    def __init__(self):
        super(YOLOdetect_config, self).__init__(pBs = 1, pWn = 2, p_force_cpu = False)

        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs", "yolov3"))
        self.path_yolocfg_file = r"official_yolo_files\configs\yolov3.cfg"
        self.path_weight_file = r"official_yolo_files\weights\yolov3.weights"
        self.path_class_file =  r"official_yolo_files\data_names\coco.names"
        self.class_Lst = self.load_classes(self.path_class_file)

        self.imgs_dir = self.check_path_valid(r"test_samples\yolo_samples")
        self.res_dir =self.check_path_valid(os.path.join(self.imgs_dir, "results"))

        self.method_init = "preTrain"

        # for non_max_suppression:
        self.iou_thres = 0.5 # iou threshold required to qualify as detected
        self.conf_thres = 0.85 # object confidence threshold
        self.nms_thres = 0.4 # iou thresshold for non-maximum suppression
        # for net input map size
        self.netin_size = 416
        
    
    def load_classes(self, path):
        """
        Loads class labels at 'path'
        """
        with  open(path, "r") as fp:
            name_Lst = fp.read().split("\n")[:-1]
        return name_Lst

    def init_net(self, pNet):
        if self.method_init == "preTrain":
            assert self.path_weight_file is not None, "weight path ungiven"
            # pNet.load_state_dict(torch.load(self.preTrain_model_path))
            if ".weight" in self.path_weight_file:
                pNet.load_darknet_weights(self.path_weight_file)
            elif ".pth" in self.path_weight_file:
                pNet.load_state_dict(torch.load(self.path_weight_file))

        pNet.to(self.device).eval()

    def create_dataset(self, istrain):
        
        q_dataset = fd_ld.ImageFolderLoader(self.imgs_dir, self.netin_size)

        return q_dataset
        


if __name__ == "__main__":
    
    gm_cfg = YOLOdetect_config()

    gm_testloader = torch.utils.data.DataLoader(
        dataset = gm_cfg.create_dataset(istrain = True), 
        batch_size= gm_cfg.ld_batchsize,
        shuffle= True,
        num_workers= gm_cfg.ld_workers
    ) 

    gm_net = dk_detect.Darknet(gm_cfg.path_yolocfg_file, img_size=gm_cfg.netin_size).to(gm_cfg.device)

    gm_cfg.init_net(gm_net)

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    for batch_i, (img_paths, img_Tsor_batch_i) in enumerate(gm_testloader):
        # print(img_paths, input_imgs.shape)
        # fd_ld.view_tensor(torchvision.utils.make_grid(
        #                 tensor = input_imgs, 
        #                 nrow= 1)
        #     )
        prev_time = time.time()
        img_Tsor_batch_i = img_Tsor_batch_i.to(gm_cfg.device)
        with torch.no_grad():
            detections = gm_net(img_Tsor_batch_i)
            detections = ut_prs.non_max_suppression(detections, gm_cfg.conf_thres, gm_cfg.nms_thres)
        inference_time = time.time() - prev_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
    
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = ut_prs.rescale_boxes(detections, gm_cfg.netin_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (gm_cfg.class_Lst[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s = gm_cfg.class_Lst[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("\\")[-1].split(".")[0]
        # plt.show()
        res_filename = os.path.join(gm_cfg.res_dir,  f"{filename}.png")

        plt.savefig(res_filename, bbox_inches="tight", pad_inches=0.0)
        plt.close()
        






