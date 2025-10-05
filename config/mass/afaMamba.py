from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.mass import *
# from catalyst.contrib.nn import Lookahead
from pytorch_optimizer import Lookahead
from catalyst import utils
from geoseg.models.AfaMamba import AFA

# training hparam
max_epoch = 1     # 105
ignore_index = 255
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

output_mask_dir, output_mask_rgb_dir = None, None
weights_name = "mass-new-afamamba-pvtv2"
weights_path = "/root/EB-TDFNet/log/mass/{}".format(weights_name)
test_weights_name = "last"
log_name = "/root/EB-TDFNet/log/mass/{}".format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
# pretrained_ckpt_path = "/root/EB-TDFNet/pre_trained_weights/whu-TDFNet-vgg16-v9192.ckpt"
pretrained_ckpt_path = None
# load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/pvt_v2_b2.pth',
# load_ckpt_path='/root/EB-TDFNet/pre_trained_weights/semi_weakly_supervised_resnet18.pth',
load_ckpt_path = None
# resume_ckpt_path = "/root/EB-TDFNet/log/mass/mass-new-afamamba-pvtv2/mass-new-afamamba-pvtv2-v4-Copy.ckpt"
resume_ckpt_path = None

#  define the network
net = AFA()

# define the loss
loss = EdgeLoss(ignore_index=255)
use_aux_loss = False

# define the dataloader


train_dataset = MassBuildDataset(data_root="/root/autodl-tmp/Massa_512/train/", mode='train', mosaic_ratio=0.25, transform=get_training_transform())
val_dataset = MassBuildDataset(data_root="/root/autodl-tmp/Massa_512/val/", mode='val', transform=get_validation_transform())
test_dataset = MassBuildDataset(data_root="/root/autodl-tmp/Massa_512/test/", mode='val', transform=get_validation_transform())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)

optimizer = Lookahead(base_optimizer,k=5,alpha=0.5)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)