import os
import math
import argparse
import random
import logging
import numpy as np
import cv2
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
import torch.nn as nn
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
# from hiding_models.utils.dataset import get_test_dataloader

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def cal_pnsr(sr_img, gt_img):
    # calculate PSNR
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.

    psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)

    return psnr
def calculate_rmse(img1, img2):
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    rmse = np.sqrt(mse)

    return np.mean(rmse)


def calculate_mae(img1, img2):

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    apd = np.mean(np.abs(img1 - img2))
    if apd == 0:
        return float('inf')

    return np.mean(apd)



def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = img1.transpose((1, 2, 0))
    img2 = img2.transpose((1, 2, 0))

    # print(img1.shape)
    # print(img2.shape)
    # bk
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')  # config 文件
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--ckpt', type=str, default=r'', help='Path to pre-trained model.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    # val_loader = get_test_dataloader('DAVIS', r'E:\dataset\DAVIS\JPEGImages\Full-Resolution', None, None, 7)
    # create model
    model = create_model(opt)
    model.load_test(args.ckpt)
            
    # validation
    avg_psnr, avg_ssim, avg_rmse, avg_mae = 0.0, 0.0, 0.0, 0.0
    avg_psnr_h, avg_ssim_h, avg_rmse_h, avg_mae_h = 0.0, 0.0, 0.0, 0.0
    avg_psnr_lr, avg_ssim_lr, avg_rmse_lr, avg_mae_lr = 0.0, 0.0, 0.0, 0.0
    idx = 0
    test_dataset_name = 'vimeo90k'
    for video_id, val_data in enumerate(val_loader):
        img_dir = os.path.join(opt['path']['val_images'], test_dataset_name)
        util.mkdir(img_dir)

        model.feed_data(val_data)
        model.test()

        visuals = model.get_current_visuals()

        t_step = visuals['GT'].shape[0]
        idx += t_step

        for i in range(t_step):

            sr_img = util.tensor2img(visuals['SR'][i])
            sr_img_h = util.tensor2img(visuals['SR_h'][i])
            gt_img = util.tensor2img(visuals['GT'][i])
            lr_img = util.tensor2img(visuals['LR'][i])
            lrgt_img = util.tensor2img(visuals['LR_ref'][i])

            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(video_id, i, 'SR'))
            util.save_img(sr_img, save_img_path)

            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(video_id, i, 'SR_h'))
            util.save_img(sr_img_h, save_img_path)

            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(video_id, i, 'GT'))
            util.save_img(gt_img, save_img_path)

            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(video_id, i, 'LR'))
            util.save_img(lr_img, save_img_path)

            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(video_id, i, 'LRGT'))
            util.save_img(lrgt_img, save_img_path)

            # --- 计算指标 ---
            # PSNR
            avg_psnr += cal_pnsr(sr_img, gt_img)
            avg_psnr_h += cal_pnsr(sr_img_h, lrgt_img)
            avg_psnr_lr += cal_pnsr(lr_img, gt_img)

            # SSIM (注意：内部有transpose，传入[C,H,W])
            avg_ssim += calculate_ssim(sr_img.transpose(2,0,1), gt_img.transpose(2,0,1))
            avg_ssim_h += calculate_ssim(sr_img_h.transpose(2,0,1), lrgt_img.transpose(2,0,1))
            avg_ssim_lr += calculate_ssim(lr_img.transpose(2,0,1), gt_img.transpose(2,0,1))

            # RMSE
            avg_rmse += calculate_rmse(sr_img, gt_img)
            avg_rmse_h += calculate_rmse(sr_img_h, lrgt_img)
            avg_rmse_lr += calculate_rmse(lr_img, gt_img)

            # MAE
            avg_mae += calculate_mae(sr_img, gt_img)
            avg_mae_h += calculate_mae(sr_img_h, lrgt_img)
            avg_mae_lr += calculate_mae(lr_img, gt_img)

    # 计算均值
    m_list = [avg_psnr, avg_psnr_h, avg_psnr_lr, avg_ssim, avg_ssim_h, avg_ssim_lr, 
              avg_rmse, avg_rmse_h, avg_rmse_lr, avg_mae, avg_mae_h, avg_mae_lr]
    res = [m / idx for m in m_list]

    print('#' * 20 + ' Validation Results ' + '#' * 20)
    print('Cover : PSNR: {:.4f}, SSIM: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}'.format(res[0], res[3], res[6], res[9]))
    print('Secret: PSNR: {:.4f}, SSIM: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}'.format(res[1], res[4], res[7], res[10]))
    print('Stego : PSNR: {:.4f}, SSIM: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}'.format(res[2], res[5], res[8], res[11]))


if __name__ == '__main__':
    main()