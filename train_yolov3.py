import os
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)
import time
import datetime
import tqdm
from terminaltables import AsciiTable
import numpy as np 
import arch.darknet as dk_detect
import loaders.imgfolderLoader as fd_ld
import custom_utils.config as ut_cfg
import custom_utils.parser as ut_prs
import custom_utils.initializer as ut_init
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("please pip install tensorboard==2.0.2")
'''
tensorboard --logdir outputs --port 8890
'''

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim

class YOLOtrain_config(ut_cfg.config):
    def __init__(self):
        super(YOLOtrain_config, self).__init__(pBs = 16, pWn = 2, p_force_cpu = False)

        self.path_save_mdroot = self.check_path_valid(os.path.join(ROOT, "outputs", "yolov3"))
        localtime = time.localtime(time.time())
        self.path_save_mdid = "vocimg416_" + "%02d%02d"%(localtime.tm_mon, localtime.tm_mday)

        self.path_yolocfg_file = r"official_yolo_files\configs\yolov3-tiny_cls20.cfg" # yolov3_cls20.cfg
        self.path_classname_file =  r"official_yolo_files\data_names\voc.names"
        self.class_Lst = ut_prs.load_classes(self.path_classname_file)
        self.path_trainlist_file = r"F:\ZimengZhao_Data\VOC2012\VOCtrainval_11-May-2012\2012_train.txt"
        self.path_vallist_file = r"F:\ZimengZhao_Data\VOC2012\VOCtrainval_11-May-2012\2012_val.txt"
        # self.pretrained_weights = r""
        
        self.log_epoch_txt = open(os.path.join(self.path_save_mdroot, "infomnist_z10unspv_MSE_epoch_loss_log.txt"), 'a+')
        self.writer = SummaryWriter(log_dir=os.path.join(self.path_save_mdroot, "board"))

        self.training_epoch_amount = 150
        self.save_epoch_begin = 50
        self.save_epoch_interval = 20
        self.val_epoch_interval = 20

        self.gradient_accumulations = 2 # number of gradient accums before step

         # for get_batch_statistics and non-maximum suppression
        self.iou_thres = 0.5 # iou threshold required to qualify as detected
        self.conf_thres = 0.001 # object confidence threshold
        self.nms_thres = 0.5 # iou thresshold for non-maximum suppression

        self.netin_size = 416

        self.compute_mAP = False # if True computes mAP every 10th batch. 
        self.multiscale_training = True

        self.method_init ="norm"  #"preTrain" #"kaming" #"xavier" # "norm"

        self.opt_baseLr = 1e-3
        self.opt_beta1 = 0.5
        self.opt_weightdecay = 3e-6

    def init_net(self, pNet):
        if self.method_init == "xavier":
            ut_init.init_xavier(pNet)
        elif self.method_init == "kaiming":
            ut_init.init_kaiming(pNet)
        elif self.method_init == "norm":
            ut_init.init_norm(pNet)
        elif self.method_init == "preTrain":
            assert self.path_weight_file is not None, "weight path ungiven"
            # pNet.load_state_dict(torch.load(self.preTrain_model_path))
            if ".weight" in self.path_weight_file:
                pNet.load_darknet_weights(self.path_weight_file)
            elif ".pth" in self.path_weight_file:
                pNet.load_state_dict(torch.load(self.path_weight_file))

        pNet.to(self.device).train()

    def create_dataset(self, istrain):
        list_file = None
        if istrain:
            list_file = self.path_trainlist_file
        else:
            list_file = self.path_vallist_file

        q_dataset = fd_ld.ListDataLoader(list_path = list_file, img_size = self.netin_size, augment = True, multiscale = self.multiscale_training)

        return q_dataset

    def name_save_model(self, save_mode, epochX = None):
        model_type = save_mode.split("_")[1] # netD / netG
        model_filename = self.path_save_mdid + model_type
        
        if "processing" in save_mode:
            assert epochX is not None, "miss the epoch info" 
            model_filename += "_%03d"%(epochX) + ".pth"
        elif "ending" in save_mode:
            model_filename += "_%03d"%(self.training_epoch_amount) + ".pth"
        elif "interrupt" in save_mode:
            model_filename += "_interrupt"+ ".pth"
        assert os.path.splitext(model_filename)[-1] == ".pth"
        q_abs_path = os.path.join(self.path_save_mdroot, model_filename)
        return q_abs_path

    def log_in_file(self, *print_paras):
        for para_i in print_paras:
            print(para_i, end= "")
            print(para_i, end= "", file = self.log_epoch_txt)
        print("")
        print("", file = self.log_epoch_txt)

    def log_in_board(self, chartname, data_Dic, epoch):
        # for key_i, val_i in data_Dic:
        self.writer.add_scalars(chartname, 
            data_Dic, epoch)

    def evaluate_net(self, pNet):
        pNet.eval()
        testdataset = self.create_dataset(istrain = False)
        testloader = torch.utils.data.DataLoader(
            dataset = testdataset, 
            batch_size= self.ld_batchsize,
            shuffle= False,
            num_workers= self.ld_workers, 
            collate_fn = testdataset.collate_fn
        )

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(testloader, desc="Detecting objects")):

            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = ut_prs.xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= self.netin_size

            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                outputs = pNet(imgs)
                outputs = ut_prs.non_max_suppression(outputs, conf_thres = self.conf_thres, nms_thres = self.nms_thres)
            
            sample_metrics += ut_prs.get_batch_statistics(outputs, targets, iou_threshold = self.iou_thres)
            
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ut_prs.ap_per_class(true_positives, pred_scores, pred_labels, labels)

        return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    
    gm_cfg = YOLOtrain_config()
    gm_dataset = gm_cfg.create_dataset(istrain = True)
    gm_trainloader = torch.utils.data.DataLoader(
        dataset = gm_dataset, 
        batch_size= gm_cfg.ld_batchsize,
        shuffle= True,
        num_workers= gm_cfg.ld_workers, 
        collate_fn = gm_dataset.collate_fn
    ) 

    gm_net = dk_detect.Darknet(gm_cfg.path_yolocfg_file, img_size=gm_cfg.netin_size).to(gm_cfg.device)

    gm_cfg.init_net(gm_net)

    gm_optimizer = optim.Adam(
        params = gm_net.parameters(),
        lr = gm_cfg.opt_baseLr,
        betas= (gm_cfg.opt_beta1, 0.99),
        # weight_decay = gm_cfg.opt_weightdecay
    )

    gm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = gm_optimizer,
        mode='min',
        factor=0.8, patience=5, verbose=True, 
        threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=0, eps=1e-08
    )

    gm_metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    loss_an_epoch_Lst = []
    try:
        print("Train_Begin".center(40, "*"))
        print("Darknet para:", end="")
        gm_cfg.check_arch_para(gm_net)
        gm_cfg.log_in_file("net_id = ", gm_cfg.path_save_mdid, ", batchsize = ", gm_cfg.ld_batchsize, ", workers = ", gm_cfg.ld_workers)
        gm_cfg.log_in_file("criterion_use: self included; ", ", init: ", gm_cfg.method_init)
        for epoch_i in range(gm_cfg.training_epoch_amount):
            start=time.time()
            gm_net.train()
            for batch_i, (_, imgs, targets) in enumerate(gm_trainloader):

                batches_done = len(gm_trainloader) * epoch_i + batch_i

                imgs = imgs.to(gm_cfg.device)
                targets = targets.to(gm_cfg.device)
                targets.requires_grad = False

                loss, outputs = gm_net(imgs, targets)
                loss.backward()
                # loss, outputs = torch.rand(1), torch.rand(1)
                loss_an_epoch_Lst.append(loss.item())
                gm_net.seen += imgs.shape[0]
                if batches_done % gm_cfg.gradient_accumulations:
                    # Accumulates gradient before each step
                    gm_optimizer.step()
                    gm_optimizer.zero_grad()
                    log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch_i, gm_cfg.training_epoch_amount, batch_i, len(gm_trainloader))
                    print(log_str, loss.item())
                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(gm_net.yolo_layers))]]]
                
            # end an epoch
            delta_t = (time.time()- start)/60
            avg_loss = sum(loss_an_epoch_Lst)/len(loss_an_epoch_Lst)
            loss_an_epoch_Lst.clear()
            gm_cfg.log_in_board("vocimg416_training loss", {"avg_loss": avg_loss}, epoch_i)
            # Log metrics at each YOLO layer
            for i, metric in enumerate(gm_metrics):
                formats = {m: "%.6f" for m in gm_metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in gm_net.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = {}
                for j, yolo in enumerate(gm_net.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            # tensorboard_log += [(f"{name}_{j+1}", metric)]
                            gm_cfg.log_in_board("vocimg416_yololayer%d_"%(j) + name, {name: metric}, epoch_i)
            
            log_str = AsciiTable(metric_table).table
            log_str += f"\nTotal loss {avg_loss}"
            # Determine approximate time left for epoch
            epoch_batches_left = len(gm_trainloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            print(log_str)

            if epoch_i % gm_cfg.val_epoch_interval == 0:
                print("\n---- Evaluating Model ----")
                precision, recall, AP, f1, ap_class = gm_cfg.evaluate_net(gm_net)
                evaluation_metrics = [
                    ("val_precision", precision.mean()),
                    ("val_recall", recall.mean()),
                    ("val_mAP", AP.mean()),
                    ("val_f1", f1.mean()),
                ]
                for metric_i in evaluation_metrics:
                    gm_cfg.log_in_board("vocimg416_vali_" + metric_i[0], {metric_i[0]: metric_i[1]}, epoch_i)
            
                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, gm_cfg.class_Lst[c], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")

                if (epoch_i >gm_cfg.save_epoch_begin and epoch_i %gm_cfg.save_epoch_interval == 1):
                    # save weight at regular interval
                    torch.save(obj = gm_netD.state_dict(), 
                        f = gm_cfg.name_save_model("processing_netD", epoch_i))
                    torch.save(obj = gm_netG.state_dict(), 
                        f = gm_cfg.name_save_model("processing_netG", epoch_i))
            

                gm_cfg.log_epoch_txt.flush()
        
        # end the train process(training_epoch_amount times to reuse the data)
        torch.save(obj = gm_netD.state_dict(),  f = gm_cfg.name_save_model("ending_netD"))
        torch.save(obj = gm_netG.state_dict(),  f = gm_cfg.name_save_model("ending_netG"))
        gm_cfg.log_epoch_txt.close()
        gm_cfg.writer.close()

    except KeyboardInterrupt:
        print("Save the Inter.pth".center(60, "*"))
        torch.save(obj = gm_netD.state_dict(), f = gm_cfg.name_save_model("interrupt_netD"))
        torch.save(obj = gm_netG.state_dict(), f = gm_cfg.name_save_model("interrupt_netG"))




                













