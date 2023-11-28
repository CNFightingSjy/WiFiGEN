import os
import matplotlib
import matplotlib.pyplot as plt
import sys

# matplotlib.use('Agg')
sys.path.append(".")
sys.path.append("..")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from models.stylegan2.model import Discriminator

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss, loss
from configs import data_configs, transforms_config
from datasets.images_dataset import ImagesDataset
from datasets.mat_dataset import MatsDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger
from options.train_options import TrainOptions


def parse_and_log_images(y, y_hat, name, path):
    im_data = common.tensor2im(y_hat)
    # fig = plt.add_gridspec(figsize=(1, 1))
    # fig.add_subplot(im_data)
    path = os.path.join(dir_path, name)
    im_data.save(path)


def test(net, dataloader, files, device, dir_path):
    net.eval()
    for batch_idx, batch in enumerate(dataloader):
        x, y = batch

        with torch.no_grad():
            x, y = x.to(device).float(), y.to(device).float()
            y_hat, latent = net.forward(x, return_latents=True)
            parse_and_log_images(y ,y_hat, os.path.basename(files[batch_idx]).replace('mat', 'jpg'), dir_path)

if __name__ == '__main__':
    opts = TrainOptions().parse()
    test_source = '/data/shijianyang/data/wifi/data4psp/test/ptot'
    test_target = '/data/shijianyang/data/wifi/data4psp/test/rgb_epsono'
    dir_path = './test_all/'
    os.makedirs(dir_path)
    print(transforms_config.MatTransforms(opts).get_transforms())
    transforms_dict = transforms_config.MatTransforms(opts).get_transforms()
    source_transform = transforms_dict['transform_source']
    target_transform = transforms_dict['transform_gt']
    device = 'cuda:1'
    net = pSp(opts).to(device)
    # test_dataset = MatsDataset(source_root=test_source, target_root=test_target, source_transform=source_transform, target_transform=target_transform, opts=opts)
    # test_dataloader = DataLoader(test_dataset, batch_size=opts.test_batch_size, shuffle=False, num_workers=int(opts.test_workers), drop_last=True)
    # # print(test_dataloader)
    # file_paths = test_dataset.source_paths
    # for path in file_paths:
    #     print(os.path.basename(path).replace('mat', 'jpg'))
    # for batch_idx, batch in enumerate(test_dataloader):
    #     print(batch_idx)
    # test(net, test_dataloader, file_paths, device)

    # ckpt = torch.load('path/to/wifi_plus/checkpoints/iteration_500000.pt')
    # print(ckpt['state_dict'].keys())