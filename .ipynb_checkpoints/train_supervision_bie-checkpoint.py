import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
from torch.cuda import empty_cache

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.automatic_optimization = False

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x, hw_range=None):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)

        if isinstance(prediction, (tuple, list)):
            if self.config.use_aux_loss:
                main_pred = prediction[0]
                if isinstance(main_pred, (tuple, list)):
                    main_pred = main_pred[0]  # 取元组的第一个元素
            else:
                main_pred = prediction[0]
        else:
            main_pred = prediction
        
        pre_mask = nn.Softmax(dim=1)(main_pred)
        pre_mask = pre_mask.argmax(dim=1)

        loss = self.loss(prediction, mask)
        
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # supervision stage
        for name, param in self.net.named_parameters():
            if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                print(f"参数 {name} 类型错误：{param.dtype}")
                param.data = param.data.float()  # 强制转为float32

        opt = self.optimizers(use_pl_optimizer=False)
        self.manual_backward(loss)
        if (batch_idx + 1) % self.config.accumulate_n == 0:
            opt.step()
            opt.zero_grad()

        sch = self.lr_schedulers()
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            sch.step()

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whu' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'mass' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'inria' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU':np.round(mIoU,6),
                      'F1': np.round(F1,6),
                      'OA': np.round(OA,6)}
        print(" ")
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = np.round(iou,6)
        print(iou_value)
        print("======================")
        print(" ")

        self.metrics_train.reset()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log_dict = {"train_loss": loss, 'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)
        empty_cache()  # 清理显存缓存，释放预留的未使用内存

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}

    def validation_epoch_end(self, outputs):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whu' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'mass' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'inria' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU':np.around(mIoU,6) ,
                      'F1': np.around(F1,6),
                      'OA': np.around(OA,6)}

        print(" ")
        print('val:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = np.around(iou,6)
        print(iou_value)
        print("======================")
        
        self.metrics_val.reset()
        loss = torch.stack([x["loss_val"] for x in outputs]).mean()
        log_dict = {"val_loss": loss, 'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)
        empty_cache()


    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='gpu',  # gpu
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy=config.strategy,
                         resume_from_checkpoint=config.resume_ckpt_path, logger=logger,
                         #precision=32,sync_batchnorm=True,
                         )

    trainer.fit(model=model)
    
    # checkpoint_callback.best_model_path
    # # trainer.save_checkpoint('model_weights/whubuilding/buildformer_large_edge_all/buildformer_large_edge_all.ckpt')
    # trainer.save_checkpoint('/root/BuildingExtraction/model_weights/Massa/dsatnet.ckpt')


if __name__ == "__main__":
    main()

# python train_supervision.py -c ./config/mass/tdfnet.py
# python train_supervision.py -c ./config/mass/afaMamba.py
# python train_supervision.py -c ./config/mass/afeNet.py


# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# import warnings
# warnings.filterwarnings("ignore")

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from tools.cfg import py2cfg
# import os
# import torch
# from torch import nn
# import cv2
# import numpy as np
# import argparse
# from pathlib import Path
# from tools.metric import Evaluator
# from pytorch_lightning.loggers import CSVLogger
# import random
# import torch.nn.functional as F

# torch.autograd.set_detect_anomaly(True)

# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def get_args():
#     parser = argparse.ArgumentParser()
#     arg = parser.add_argument
#     arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
#     return parser.parse_args()


# class Supervision_Train(pl.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.net = config.net  # 这应该是输出(seg_logits, rec_logits)的模型
#         self.automatic_optimization = False

#         # 分割损失（配置中可能已定义，这里做兼容处理）
#         self.loss_seg = config.loss
#         self.loss_rec = nn.L1Loss()  # 重建损失固定为L1损失
        
#         self.ignore_index = config.ignore_index  # 从配置获取忽略索引

#         self.metrics_train = Evaluator(num_class=config.num_classes)
#         self.metrics_val = Evaluator(num_class=config.num_classes)
        
#         # MLP优化重建结果的特征
#         self.conv_mlp = ConvModule(    
#                     in_channels=self.out_channels + 1,
#                     out_channels=self.out_channels + 1,
#                     kernel_size=1,
#                     stride=1,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg)
        
#         # 用于保存训练批次输出
#         self.train_outputs = []

#     def forward(self, x):
#         # 前向传播返回分割和重建结果
#         seg_logits, rec_logits = self.net(x)
#         return seg_logits, rec_logits

#     def training_step(self, batch, batch_idx):
#         # 解包数据（包含图像、分割标签、增强图像用于重建）
#         img, seg_label, img_aug = batch['img'], batch['gt_semantic_seg'], batch['img_aug']

#         # 模型输出：分割logits和重建logits
#         seg_logits, rec_logits = self.net(img)


#         if seg_label.dim() == 4:  # 标签形状为 (B, 1, H, W)（带通道维度）
#             target_size = seg_label.shape[2:]  # 取 [H, W]，如 (512, 512)
#         else:  
#             target_size = seg_label.shape[1:]
#         seg_logits = F.interpolate(seg_logits,size=target_size,mode='bilinear')

#         if rec_logits.dim() == 4:  # 重建输出为 (B, C, H, W)
#             rec_target_size = img_aug.shape[2:]  # 增强图的 (H, W)
#         else:
#             rec_target_size = img_aug.shape[1:]
#         rec_logits = F.interpolate(rec_logits,size=rec_target_size,mode='bilinear')
        
#         #残差连接
#         # 重建结果与原始图像融合，经卷积优化后计算损失
#         rec_logits = rec_logits + img
#         rec_logits = self.conv_mlp(rec_logits)
        

#         # 计算分割损失
#         seg_label_squeezed = seg_label.squeeze(1)  # 移除通道维度
#         loss_seg = self.loss_seg(seg_logits, seg_label_squeezed)
#         loss_rec = self.loss_rec(rec_logits, img_aug)
#         total_loss = loss_seg + loss_rec

#         # 计算分割指标
#         pre_mask = F.softmax(seg_logits, dim=1).argmax(dim=1)
#         for i in range(seg_label.shape[0]):
#             self.metrics_train.add_batch(
#                 seg_label[i].cpu().numpy(), 
#                 pre_mask[i].cpu().numpy()
#             )

#         # 优化器步骤
#         opt = self.optimizers(use_pl_optimizer=False)
#         self.manual_backward(total_loss)
        
#         if (batch_idx + 1) % self.config.accumulate_n == 0:
#             opt.step()
#             opt.zero_grad()

#         # 学习率调度
#         sch = self.lr_schedulers()
#         if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
#             sch.step()

#         # 保存当前批次的损失
#         self.train_outputs.append({
#             "total_loss": total_loss,
#             "loss_seg": loss_seg,
#             "loss_rec": loss_rec
#         })
#         return total_loss

#     def on_train_epoch_end(self):
#         outputs = self.train_outputs
        
#         # 计算分割指标
#         if 'vaihingen' in self.config.log_name:
#             mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
#             F1 = np.nanmean(self.metrics_train.F1()[:-1])
#         elif 'potsdam' in self.config.log_name:
#             mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
#             F1 = np.nanmean(self.metrics_train.F1()[:-1])
#         elif 'whu' in self.config.log_name:
#             mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
#             F1 = np.nanmean(self.metrics_train.F1()[:-1])
#         elif 'mass' in self.config.log_name:
#             mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
#             F1 = np.nanmean(self.metrics_train.F1()[:-1])
#         elif 'inria' in self.config.log_name:
#             mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
#             F1 = np.nanmean(self.metrics_train.F1()[:-1])
#         elif 'china' in self.config.log_name:
#             mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
#             F1 = np.nanmean(self.metrics_train.F1()[:-1])
#         else:
#             mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
#             F1 = np.nanmean(self.metrics_train.F1()[:-1])

#         OA = np.nanmean(self.metrics_train.OA())
#         iou_per_class = self.metrics_train.Intersection_over_Union()
        
#         # 打印分割指标
#         eval_value = {
#             'mIoU': np.round(mIoU, 6),
#             'F1': np.round(F1, 6),
#             'OA': np.round(OA, 6)
#         }
#         print("\ntrain metrics:", eval_value)
#         iou_value = {cls: np.round(iou, 6) for cls, iou in zip(self.config.classes, iou_per_class)}
#         print("per class IoU:", iou_value)
#         print("======================")

#         # 计算并记录损失
#         total_loss = torch.stack([x["total_loss"] for x in outputs]).mean()
#         loss_seg = torch.stack([x["loss_seg"] for x in outputs]).mean()
#         loss_rec = torch.stack([x["loss_rec"] for x in outputs]).mean()
        
#         log_dict = {
#             "train_total_loss": total_loss,
#             "train_loss_seg": loss_seg,
#             "train_loss_rec": loss_rec,
#             'train_mIoU': mIoU,
#             'train_F1': F1,
#             'train_OA': OA
#         }
#         self.log_dict(log_dict, prog_bar=True)
        
#         # 重置指标和输出列表
#         self.metrics_train.reset()
#         self.train_outputs.clear()

#     def validation_step(self, batch, batch_idx):
#         img = batch['img']
#         seg_label = batch['gt_semantic_seg']  # 假设形状为 (B, 1, H, W) 或 (B, H, W)
        

#         seg_logits, rec_logits = self.forward(img)
        
#         if seg_label.dim() == 4:  # 带通道维度 (B, C, H, W)
#             target_size = seg_label.shape[2:]  # 结果为 (H, W)
#         else:  # 不带通道维度 (B, H, W)
#             target_size = seg_label.shape[1:]  # 结果为 (H, W)
        
#         seg_logits = F.interpolate(
#             seg_logits,
#             size=target_size,  # 此时为 (512, 512)，二维
#             mode='bilinear',
#             align_corners=self.net.align_corners
#         )
        
#         pre_mask = F.softmax(seg_logits, dim=1).argmax(dim=1)
#         for i in range(seg_label.shape[0]):
#             pred = pre_mask[i].cpu().numpy()
#             label = seg_label[i].squeeze().cpu().numpy()  # 移除通道维度（若有）
#             self.metrics_val.add_batch(label, pred)
        
#         # 计算损失
#         seg_label_squeezed = seg_label.squeeze(1) if seg_label.dim() == 4 else seg_label
#         loss_seg = self.loss_seg(seg_logits, seg_label_squeezed)
#         total_loss = loss_seg
        
#         return {"val_total_loss": total_loss, "val_loss_seg": loss_seg}
    

#     def on_validation_epoch_end(self):
#         # 获取验证输出
#         outputs = []
#         if hasattr(self.trainer, 'callback_metrics'):
#             outputs = self.trainer.callback_metrics.get('val_total_loss', [])
#         if not outputs and hasattr(self.trainer, '_results'):
#             outputs = self.trainer._results.get('validation_step', [])

#         # 计算分割指标
#         # 先获取每类IoU，修复未定义问题
#         iou_per_class = self.metrics_val.Intersection_over_Union()
        
#         if 'vaihingen' in self.config.log_name:
#             mIoU = np.nanmean(iou_per_class[:-1])
#             F1 = np.nanmean(self.metrics_val.F1()[:-1])
#         elif 'potsdam' in self.config.log_name:
#             mIoU = np.nanmean(iou_per_class[:-1])
#             F1 = np.nanmean(self.metrics_val.F1()[:-1])
#         elif 'whu' in self.config.log_name:
#             mIoU = np.nanmean(iou_per_class[:-1])
#             F1 = np.nanmean(self.metrics_val.F1()[:-1])
#         elif 'mass' in self.config.log_name:
#             mIoU = np.nanmean(iou_per_class[:-1])
#             F1 = np.nanmean(self.metrics_val.F1()[:-1])
#         elif 'inria' in self.config.log_name:
#             mIoU = np.nanmean(iou_per_class[:-1])
#             F1 = np.nanmean(self.metrics_val.F1()[:-1])
#         elif 'jm' in self.config.log_name:
#             mIoU = np.nanmean(iou_per_class[:-1])
#             F1 = np.nanmean(self.metrics_val.F1()[:-1])
#         else:
#             mIoU = np.nanmean(iou_per_class[:-1])
#             F1 = np.nanmean(self.metrics_val.F1()[:-1])

#         OA = np.nanmean(self.metrics_val.OA())
        
#         # 打印验证指标
#         eval_value = {
#             'mIoU': np.around(mIoU, 6),
#             'F1': np.around(F1, 6),
#             'OA': np.around(OA, 6)
#         }
#         print("\nval metrics:", eval_value)
#         iou_value = {cls: np.around(iou, 6) for cls, iou in zip(self.config.classes, iou_per_class)}
#         print("per class IoU:", iou_value)
#         print("======================")

#         # 构建日志字典 - 确保val_mIoU被记录
#         log_dict = {
#             'val_mIoU': mIoU,
#             'val_F1': F1,
#             'val_OA': OA
#         }

#         # 处理损失记录（如果有输出）
#         if outputs:
#             try:
#                 total_loss = torch.stack([x["val_total_loss"] for x in outputs]).mean()
#                 loss_seg = torch.stack([x["val_loss_seg"] for x in outputs]).mean()
#                 log_dict["val_total_loss"] = total_loss
#                 log_dict["val_loss_seg"] = loss_seg
#             except (KeyError, TypeError, ValueError):
#                 pass

#         # 强制日志记录，确保指标被记录
#         self.log_dict(log_dict, prog_bar=True, sync_dist=True)

#         # 重置验证指标
#         self.metrics_val.reset()

#     def configure_optimizers(self):
#         optimizer = self.config.optimizer
#         lr_scheduler = self.config.lr_scheduler
#         return [optimizer], [lr_scheduler]

#     def train_dataloader(self):
#         return self.config.train_loader

#     def val_dataloader(self):
#         return self.config.val_loader


# def main():
#     args = get_args()
#     config = py2cfg(args.config_path)
#     seed_everything(42)

#     # 检查点回调（监控总损失或mIoU，根据配置决定）
#     checkpoint_callback = ModelCheckpoint(
#         save_top_k=config.save_top_k,
#         monitor=config.monitor,
#         save_last=config.save_last,
#         mode=config.monitor_mode,
#         dirpath=config.weights_path,
#         filename=config.weights_name
#     )
#     logger = CSVLogger('lightning_logs', name=config.log_name)

#     model = Supervision_Train(config)
#     if config.pretrained_ckpt_path:
#         model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

#     # 训练器配置
#     trainer = pl.Trainer(
#         devices=config.gpus,
#         max_epochs=config.max_epoch,
#         accelerator='gpu',
#         check_val_every_n_epoch=1,
#         callbacks=[checkpoint_callback],
#         strategy=None,
#         logger=logger,
#     )

#     trainer.fit(model=model)


# if __name__ == "__main__":
#     main()


# # python train_supervision.py -c ./config/mass/tdfnet.py
# # python train_supervision_bie.py -c ./config/mass/bienet.py
# # python train_supervision.py -c ./config/mass/afeNet.py