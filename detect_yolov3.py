import time 
import os 
import sys 
sys.path.append(os.getcwd())
import arch.darknet as dk_detect
import loaders.imgfolderLoader as fd_ld
import custom_utils.config as ut_cfg

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

        self.netin_size = 416

        self.yolocfg_filename = r"official_yolo_files\configs\yolov3.cfg"
        self.weight_filename = r"official_yolo_files\weights\yolov3.weights"
        self.class_filename =  r"official_yolo_files\data_names\coco.names"

        self.imgs_dir = r"test_samples\yolo_samples"
        self.res_dir = os.path.join(self.imgs_dir, "results")
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        self.method_init = "preTrain"


        self.class_Lst = self.load_classes(self.class_filename)
    
    def load_classes(self, path):
        """
        Loads class labels at 'path'
        """
        with  open(path, "r") as fp:
            name_Lst = fp.read().split("\n")[:-1]
        return name_Lst

    def non_max_suppression(self, prediction, conf_thres=0.85, nms_thres=0.4):
        
        """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_score, class_pred)
        """
        def xywh2xyxy(x):
            y = x.new(x.shape)
            y[..., 0] = x[..., 0] - x[..., 2] / 2
            y[..., 1] = x[..., 1] - x[..., 3] / 2
            y[..., 2] = x[..., 0] + x[..., 2] / 2
            y[..., 3] = x[..., 1] + x[..., 3] / 2
            return y
        def bbox_iou(box1, box2, x1y1x2y2=True):
            """
            Returns the IoU of two bounding boxes
            """
            if not x1y1x2y2:
                # Transform from center and width to exact coordinates
                b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
                b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
                b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
                b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
            else:
                # Get the coordinates of bounding boxes
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

            # get the corrdinates of the intersection rectangle
            inter_rect_x1 = torch.max(b1_x1, b2_x1)
            inter_rect_y1 = torch.max(b1_y1, b2_y1)
            inter_rect_x2 = torch.min(b1_x2, b2_x2)
            inter_rect_y2 = torch.min(b1_y2, b2_y2)
            # Intersection area
            inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
                inter_rect_y2 - inter_rect_y1 + 1, min=0
            )
            # Union Area
            b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
            b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

            iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

            return iou
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        output = [None for _ in range(len(prediction))]
        for image_i, image_pred in enumerate(prediction):
            # Filter out confidence scores below threshold
            image_pred = image_pred[image_pred[:, 4] >= conf_thres]
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Object confidence times class confidence
            score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
            # Sort by it
            image_pred = image_pred[(-score).argsort()]
            class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
            detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
            # Perform non-maximum suppression
            keep_boxes = []
            while detections.size(0):
                large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
                label_match = detections[0, -1] == detections[:, -1]
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                invalid = large_overlap & label_match
                weights = detections[invalid, 4:5]
                # Merge overlapping bboxes by order of confidence
                detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
                keep_boxes += [detections[0]]
                detections = detections[~invalid]
            if keep_boxes:
                output[image_i] = torch.stack(keep_boxes)

        return output

    def rescale_boxes(self, boxes, current_dim, original_shape):
        """ Rescales bounding boxes to the original shape """
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x
        # Rescale bounding boxes to dimension of original image
        boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
        boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
        return boxes
    
    def init_net(self, pNet):
        if self.method_init == "preTrain":
            assert self.weight_filename is not None, "weight path ungiven"
            # pNet.load_state_dict(torch.load(self.preTrain_model_path))
            if ".weight" in self.weight_filename:
                pNet.load_darknet_weights(self.weight_filename)
            elif ".pth" in self.weight_filename:
                pNet.load_state_dict(torch.load(self.weight_filename))

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

    gm_net = dk_detect.Darknet(gm_cfg.yolocfg_filename, img_size=gm_cfg.netin_size).to(gm_cfg.device)

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
            detections = gm_cfg.non_max_suppression(detections)
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
            detections = gm_cfg.rescale_boxes(detections, gm_cfg.netin_size, img.shape[:2])
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
        






