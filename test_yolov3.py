import os
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)
import time
import tqdm


import arch.darknet as dk_detect
import loaders.imgfolderLoader as fd_ld
import custom_utils.config as ut_cfg
import custom_utils.parser as ut_prs


import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim


class YOLOtest_config(ut_cfg.config):
    def __init__(self):
        super(YOLOtest_config, self).__init__(pBs = 8, pWn = 2, p_force_cpu = False)

        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs", "yolov3"))
        localtime = time.localtime(time.time())
        self.path_save_mdid = "yoloVOC" + "%02d%02d"%(localtime.tm_mon, localtime.tm_mday)

        self.path_yolocfg_file = r"official_yolo_files\configs\yolov3.cfg"
        self.path_weight_file = r"official_yolo_files\weights\yolov3.weights"

        # self.datacfg_filename = r"official_yolo_files\configs\coco.data"
        self.path_classname_file =  r"official_yolo_files\data_names\coco.names"
        self.class_Lst = ut_prs.load_classes(self.path_classname_file)
        self.path_trainlist_file = r"F:\ZimengZhao_Data\VOC2012\VOCtrainval_11-May-2012\2012_train.txt"
        self.path_vallist_file = r"F:\ZimengZhao_Data\VOC2012\VOCtrainval_11-May-2012\2012_val.txt"

        self.method_init = "preTrain"

        # for get_batch_statistics and non-maximum suppression
        self.iou_thres = 0.5 # iou threshold required to qualify as detected
        self.conf_thres = 0.001 # object confidence threshold
        self.nms_thres = 0.5 # iou thresshold for non-maximum suppression

        self.netin_size = 416

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
        
        q_dataset = fd_ld.ListDataLoader(path = self.path_datavali_dir, img_size = self.netin_size, augment = False, multiscale = False)

        return q_dataset

    def evaluate_net(self, pNet):
        pNet.eval()

        testloader = torch.utils.data.DataLoader(
            dataset = gm_cfg.create_dataset(istrain = True), 
            batch_size= gm_cfg.ld_batchsize,
            shuffle= True,
            num_workers= gm_cfg.ld_workers
        )

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = ut_prs.xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size

            imgs = imgs.to(self.device)

            with torch.no_grad():
                outputs = pNet(imgs)
                outputs = ut_prs.non_max_suppression(outputs, conf_thres = self.conf_thres, nms_thres = self.nms_thres)
            
            sample_metrics += ut_prs.get_batch_statistics(outputs, targets, iou_threshold = self.iou_thres)
            
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        return precision, recall, AP, f1, ap_class



if __name__ == "__main__":
    

    gm_cfg = YOLOtest_config()

    gm_net = dk_detect.Darknet(gm_cfg.path_yolocfg_file, img_size=gm_cfg.netin_size).to(gm_cfg.device)

    gm_cfg.init_net(gm_net)

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = gm_cfg.evaluate_net(gm_net)

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")







