import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from thop import profile

import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization
from .modules.common import DWT,IWT
from models.modules.mutal import Mutual_info_reg
logger = logging.getLogger('base')
dwt=DWT()
iwt=IWT()


class Model_VSN(BaseModel):
    def __init__(self, opt):
        super(Model_VSN, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.gop = opt['gop']
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.opt_net = opt['network_G']
        self.center = self.gop // 2
        self.num_video = opt['num_video']
        self.idxx = 0

        self.netG = networks.define_G_v2(opt).to(self.device)
        self.mutual_info_reg = Mutual_info_reg(input_channels=36, channels=36).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.mutual_info_reg = DistributedDataParallel(self.mutual_info_reg, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
            self.mutual_info_reg = DataParallel(self.mutual_info_reg)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
            self.Reconstruction_center = ReconstructionLoss(losstype="center")

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            for k, v in self.mutual_info_reg.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        """加载输入数据到模型"""
        ref_L = data['LQ'].to(self.device)  # 低质量图像(秘密图像)
        # 如果ref_L有额外的维度1（形状为b,1,t,c,h,w），去掉它
        if len(ref_L.shape) == 6 and ref_L.shape[1] == 1:
            self.ref_L = ref_L.squeeze(1)  # (b, 1, t, c, h, w) → (b, t, c, h, w)
        else:
            self.ref_L = ref_L
        self.real_H = data['GT'].to(self.device)  # 高质量图像(载体图像)

    def init_hidden_state(self, z):
        b, c, h, w = z.shape
        h_t = []
        c_t = []
        for _ in range(self.opt_net['block_num_rbm']):
            h_t.append(torch.zeros([b, c, h, w]).cuda())
            c_t.append(torch.zeros([b, c, h, w]).cuda())
        memory = torch.zeros([b, c, h, w]).cuda()

        return h_t, c_t, memory

    def loss_forward(self, out, y):
        if self.opt['model'] == 'LSTM-VRN':
            l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
            return l_forw_fit
        elif self.opt['model'] == 'MIMO-VRN-h':
            l_forw_fit = 0
            for i in range(out.shape[1]):
                l_forw_fit += self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out[:, i], y[:, i])
            return l_forw_fit

    def loss_back_rec(self, out, x):
        if self.opt['model'] == 'LSTM-VRN':
            l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
            return l_back_rec
        elif self.opt['model'] == 'MIMO-VRN-h':
            l_back_rec = 0
            for i in range(x.shape[1]):
                l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out[:, i], x[:, i])
            return l_back_rec
    
    def loss_back_rec_mul(self, out, x):
        out = torch.chunk(out,self.num_video,dim=1)
        out = [outi.squeeze(1) for outi in out]
        x = torch.chunk(x,self.num_video,dim=1)
        x = [xi.squeeze(1) for xi in x]
        l_back_rec = 0
        for i in range(len(x)):
            for j in range(x[i].shape[1]):
                l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out[i][:, j], x[i][:, j])
        return l_back_rec

    def loss_center(self, out, x):
        # x.shape: (b, t, c, h, w)
        b, t = x.shape[:2]
        l_center = 0
        for i in range(b):
            mse_s = self.Reconstruction_center(out[i], x[i])
            mse_mean = torch.mean(mse_s)
            for j in range(t):
                l_center += torch.sqrt((mse_s[j] - mse_mean.detach()) ** 2 + 1e-18)
        l_center = self.train_opt['lambda_center'] * l_center / b

        return l_center

    def optimize_parameters(self, current_step):
        self.optimizer_G.zero_grad()
        
        b, t, c, h, w = self.ref_L.shape
        center = t // 2
        intval = self.gop // 2

        self.host = self.real_H[:, center - intval:center + intval + 1]
        self.secret = self.ref_L[:, center - intval:center + intval + 1]
        self.output, out_h = self.netG(x=dwt(self.host.reshape(b, -1, h, w)), x_h=dwt(self.secret.reshape(b, -1, h, w)))
        self.output = iwt(self.output)
        l_mi = self.mutual_info_reg(dwt(self.secret.reshape(b, -1, h, w)), out_h)
        l_mi = torch.clip(l_mi, -1, 1)
        Gt_ref = self.real_H[:, center - intval:center + intval + 1].detach()
        container = self.output[:, :3 * self.gop, :, :].reshape(-1, self.gop, 3, h, w)
        # Gt_ref形状: (b, self.gop, 3, h, w)，与container形状匹配
        l_forw_fit = self.loss_forward(container, Gt_ref)

        # 对所有帧进行量化，而不是只对中心帧
        output_reshaped = self.output[:, :3 * self.gop, :, :].view(-1, self.gop, 3, h, w)
        y = self.Quantization(output_reshaped.reshape(b, -1, h, w))


        out_x, out_x_h, out_z = self.netG(x=dwt(y), rev=True)
        out_x = iwt(out_x)
        out_x_h = iwt(out_x_h)

        l_back_rec = self.loss_back_rec(out_x.reshape(-1, self.gop, 3, h, w), self.host)



        l_center_x = self.loss_back_rec(out_x_h.reshape(-1, self.gop, 3, h, w), self.ref_L[:, center - intval:center + intval + 1])

        loss = l_forw_fit*2 + l_back_rec + l_center_x*4 + l_mi*8
        loss.backward()

        if self.train_opt['lambda_center'] != 0:
            self.log_dict['l_center_x'] = l_center_x.item()

        # set log
        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_mi'] = l_mi.item()
        self.log_dict['l_h'] = (l_center_x*10).item()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
            nn.utils.clip_grad_norm_(self.mutual_info_reg.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

    # def test(self):
    #     self.netG.eval()
    #     with torch.no_grad():
    #         forw_L = []
    #         forw_L_h = []
    #         fake_H = []
    #         fake_H_h = []
    #         pred_z = []
    #         # 开始计时
    #         # start_time = time.time()
    #         b, t, c, h, w = self.real_H.shape
    #         center = t // 2
    #         intval = self.gop // 2
    #         ids=[-1,0,1]
    #         b, t, c, h, w = self.ref_L.shape
    #         for j in range(3):
    #             id=ids[j]
    #             self.host = self.real_H[:, center - intval + id :center + intval + 1 + id].reshape(b, -1, h, w)
    #             self.secret = self.ref_L[:, center - intval + id :center + intval + 1 + id].reshape(b, -1, h, w)

    #             self.output, out_h = self.netG(x=dwt(self.host), x_h=dwt(self.secret))
    #             self.output = iwt(self.output)
    #             out_lrs = self.output
    #             y = self.Quantization(self.output)
    #             out_x, out_x_h, out_z = self.netG(x=dwt(y), rev=True)
    #             out_x = iwt(out_x)
    #             out_x_h = iwt(out_x_h)
    #             # 结束计时并打印
    #             # elapsed_time = time.time() - start_time
    #             # print('运行时间: {:.3f} 秒'.format( elapsed_time))

    #             forw_L.append(out_lrs.reshape(b,-1,c,h,w))
    #             fake_H.append(out_x.reshape(b,-1,c,h,w))
    #             fake_H_h.append(out_x_h.reshape(b,-1,c,h,w))


    #     self.forw_L = torch.clamp(torch.stack(forw_L, dim=1), 0, 1)
    #     self.fake_H = torch.clamp(torch.stack(fake_H, dim=1), 0, 1)
    #     self.fake_H_h = torch.clamp(torch.stack(fake_H_h, dim=1), 0, 1)
    #     self.netG.train()
    def test(self):
        self.netG.eval()
        self.mutual_info_reg.eval()
        with torch.no_grad():
            # forw_L = []
            # forw_L_h = []
            # fake_H = []
            # fake_H_h = []
            # pred_z = []
            # 开始计时
            # start_time = time.time()
            b, t, c, h, w = self.real_H.shape
            center = t // 2
            intval = self.gop // 2

            b, t, c, h, w = self.ref_L.shape

            self.host = self.real_H[:, center - intval:center + intval + 1].reshape(b, -1, h, w)
            self.secret = self.ref_L[:, center - intval:center + intval + 1].reshape(b, -1, h, w)
            # # --- 替换计时逻辑开始 ---
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            self.output, out_h = self.netG(x=dwt(self.host), x_h=dwt(self.secret))
            self.output = iwt(self.output)
            # end_event.record()
            # torch.cuda.synchronize() # 确保 GPU 任务完成
            # elapsed_time = start_event.elapsed_time(end_event) / 1000.0 # 毫秒转秒
            # print('运行时间: {:.3f} 秒'.format(elapsed_time))
            out_lrs = self.output
            y = self.Quantization(self.output)
            out_x, out_x_h, out_z = self.netG(x=dwt(y), rev=True)
            out_x = iwt(out_x)
            out_x_h = iwt(out_x_h)
            # 结束计时并打印
            # elapsed_time = time.time() - start_time
            # print('运行时间: {:.3f} 秒'.format( elapsed_time))

            forw_L=out_lrs.reshape(b,-1,c,h,w)
            fake_H=out_x.reshape(b,-1,c,h,w)
            fake_H_h=out_x_h.reshape(b,-1,c,h,w)

        self.forw_L = torch.clamp(forw_L, 0, 1)
        self.fake_H = torch.clamp(fake_H, 0, 1)
        self.fake_H_h = torch.clamp(fake_H_h, 0, 1)
        self.netG.train()
        self.mutual_info_reg.train()

    # def test(self):
    #     self.netG.eval()
    #     self.mutual_info_reg.eval()

    #     with torch.no_grad():

    #         b, t, c, h, w = self.real_H.shape
    #         center = t // 2
    #         intval = self.gop // 2

    #         # ============================================================
    #         # Prepare GoF input
    #         # ============================================================
    #         b, t, c, h, w = self.ref_L.shape

    #         self.host = self.real_H[
    #             :,
    #             center - intval:center + intval + 1
    #         ].reshape(
    #             b, -1, h, w
    #         )

    #         self.secret = self.ref_L[
    #             :,
    #             center - intval:center + intval + 1
    #         ].reshape(
    #             b, -1, h, w
    #         )

    #         # ============================================================
    #         # DWT
    #         # DWT is NOT included in FLOPs measurement
    #         # ============================================================
    #         host_dwt = dwt(self.host)
    #         secret_dwt = dwt(self.secret)

    #         # ============================================================
    #         # FLOPs measurement
    #         # Only measure ONE forward embedding pass
    #         #
    #         # Included:
    #         #     netG forward
    #         #
    #         # Excluded:
    #         #     DWT
    #         #     IWT
    #         #     Quantization
    #         #     reverse netG
    #         # ============================================================

    #         if isinstance(self.netG, torch.nn.DataParallel):
    #             netG_profile = self.netG.module
    #         else:
    #             netG_profile = self.netG

    #         device = host_dwt.device

    #         netG_profile = netG_profile.to(device)
    #         netG_profile.eval()

    #         secret_profile = secret_dwt.to(device)

    #         flops, params = profile(
    #             netG_profile,
    #             inputs=(
    #                 host_dwt,
    #                 secret_profile
    #             ),
    #             verbose=False
    #         )

    #         # ============================================================
    #         # Print FLOPs
    #         # ============================================================
    #         print('========================================')
    #         print(
    #             'Forward FLOPs (GoF): {:.2f} G'.format(
    #                 flops / 1e9
    #             )
    #         )
    #         print(
    #             'FLOPs / frame: {:.2f} G'.format(
    #                 flops / 1e9 / self.gop
    #             )
    #         )
    #         print(
    #             'Parameters: {:.2f} M'.format(
    #                 params / 1e6
    #             )
    #         )
    #         print('========================================')

    #         # ============================================================
    #         # Warm-up
    #         # Avoid CUDA initialization affecting inference time
    #         # ============================================================
    #         for _ in range(10):
    #             _output, _out_h = self.netG(
    #                 x=host_dwt,
    #                 x_h=secret_dwt
    #             )

    #         torch.cuda.synchronize()

    #         # ============================================================
    #         # Forward inference time
    #         #
    #         # Only measure ONE embedding forward pass
    #         # ============================================================
    #         start_event = torch.cuda.Event(
    #             enable_timing=True
    #         )

    #         end_event = torch.cuda.Event(
    #             enable_timing=True
    #         )

    #         start_event.record()

    #         self.output, out_h = self.netG(
    #             x=host_dwt,
    #             x_h=secret_dwt
    #         )

    #         end_event.record()

    #         torch.cuda.synchronize()

    #         elapsed_time = start_event.elapsed_time(
    #             end_event
    #         )

    #         # ============================================================
    #         # Print inference time
    #         # ============================================================
    #         print(
    #             '========================================'
    #         )

    #         print(
    #             'Forward inference time (GoF): {:.3f} ms'.format(
    #                 elapsed_time
    #             )
    #         )

    #         print(
    #             'Inference time / frame: {:.3f} ms'.format(
    #                 elapsed_time / self.gop
    #             )
    #         )

    #         print(
    #             '========================================'
    #         )

    #         # ============================================================
    #         # IWT
    #         # Not included in the above timing
    #         # ============================================================
    #         self.output = iwt(
    #             self.output
    #         )

    #         # ============================================================
    #         # Forward stego output
    #         # ============================================================
    #         out_lrs = self.output

    #         # ============================================================
    #         # Reverse process
    #         # NOT included in FLOPs or forward inference time
    #         # ============================================================
    #         y = self.Quantization(
    #             self.output
    #         )

    #         out_x, out_x_h, out_z = self.netG(
    #             x=dwt(y),
    #             rev=True
    #         )

    #         out_x = iwt(out_x)
    #         out_x_h = iwt(out_x_h)

    #         # ============================================================
    #         # Reshape results
    #         # ============================================================
    #         forw_L = out_lrs.reshape(
    #             b,
    #             -1,
    #             c,
    #             h,
    #             w
    #         )

    #         fake_H = out_x.reshape(
    #             b,
    #             -1,
    #             c,
    #             h,
    #             w
    #         )

    #         fake_H_h = out_x_h.reshape(
    #             b,
    #             -1,
    #             c,
    #             h,
    #             w
    #         )

    #     # ================================================================
    #     # Save outputs
    #     # ================================================================
    #     self.forw_L = torch.clamp(
    #         forw_L,
    #         0,
    #         1
    #     )

    #     self.fake_H = torch.clamp(
    #         fake_H,
    #         0,
    #         1
    #     )

    #     self.fake_H_h = torch.clamp(
    #         fake_H_h,
    #         0,
    #         1
    #     )

    #     self.netG.train()
    #     self.mutual_info_reg.train()

        
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        b, t, c, h, w = self.ref_L.shape
        center = t // 2
        intval = 3 // 2
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L[:, center - intval:center + intval + 1].detach()[0].float().cpu()
        out_dict['SR_h'] = self.fake_H_h.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H[:, center - intval:center + intval + 1].detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    
    def load_test(self,load_path_G):
        self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
