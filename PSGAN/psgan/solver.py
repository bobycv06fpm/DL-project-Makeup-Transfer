#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import os.path as osp
pwd = osp.split(osp.realpath(__file__))[0]

import time
import datetime

import torch
from torch import nn
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import torch.nn.init as init
from torch.autograd import Variable
from torchgpipe import GPipe

from ops.loss_added import GANLoss
from ops.histogram_loss import HistogramLoss
import tools.plot as plot_fig
from . import net
from .preprocess import PreProcess
from concern.track import Track


class Solver(Track):
    def __init__(self, config, device="cpu", data_loader=None, inference=False):
        self.G = net.Generator()
        if inference:
            self.G.load_state_dict(torch.load(inference, map_location=torch.device(device)))
            self.G = self.G.to(device).eval()
            return

        self.start_time = time.time()
        self.checkpoint = config.MODEL.WEIGHTS
        self.log_path = config.LOG.LOG_PATH
        self.result_path = os.path.join(self.log_path, config.LOG.VIS_PATH)
        self.snapshot_path = os.path.join(self.log_path, config.LOG.SNAPSHOT_PATH)
        self.log_step = config.LOG.LOG_STEP
        self.vis_step = config.LOG.VIS_STEP
        self.snapshot_step = config.LOG.SNAPSHOT_STEP // torch.cuda.device_count()

        # Data loader
        self.data_loader_train = data_loader
        self.img_size = config.DATA.IMG_SIZE

        self.num_epochs = config.TRAINING.NUM_EPOCHS
        self.num_epochs_decay = config.TRAINING.NUM_EPOCHS_DECAY
        self.g_lr = config.TRAINING.G_LR
        self.d_lr = config.TRAINING.D_LR
        self.g_step = config.TRAINING.G_STEP
        self.beta1 = config.TRAINING.BETA1
        self.beta2 = config.TRAINING.BETA2

        self.lambda_idt      = config.LOSS.LAMBDA_IDT
        self.lambda_A        = config.LOSS.LAMBDA_A
        self.lambda_B        = config.LOSS.LAMBDA_B
        self.lambda_his_lip  = config.LOSS.LAMBDA_HIS_LIP
        self.lambda_his_skin = config.LOSS.LAMBDA_HIS_SKIN
        self.lambda_his_eye  = config.LOSS.LAMBDA_HIS_EYE
        self.lambda_vgg      = config.LOSS.LAMBDA_VGG


        # Hyper-parameteres
        self.d_conv_dim = config.MODEL.D_CONV_DIM
        self.d_repeat_num = config.MODEL.D_REPEAT_NUM
        self.norm = config.MODEL.NORM

        self.device = device
        
        self.D_A = net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm)
        self.D_B = net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm)

        self.G.apply(self.weights_init_xavier)
        self.D_A.apply(self.weights_init_xavier)
        self.D_B.apply(self.weights_init_xavier)


        # self.load_checkpoint()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)

        self.vgg = net.vgg16(pretrained=True)
        self.criterionHis = HistogramLoss()

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), self.d_lr, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.device = "cuda"
            if torch.cuda.device_count() > 1:
                self.G = nn.DataParallel(self.G)
                self.D_A = nn.DataParallel(self.D_A)
                self.D_B = nn.DataParallel(self.D_B)
                self.vgg = nn.DataParallel(self.vgg)
                self.criterionHis = nn.DataParallel(self.criterionHis)
                self.criterionGAN = nn.DataParallel(self.criterionGAN)
                self.criterionL1 = nn.DataParallel(self.criterionL1)
                self.criterionL2 = nn.DataParallel(self.criterionL2)
                self.criterionGAN = nn.DataParallel(self.criterionGAN)

            self.G.cuda()
            self.vgg.cuda()
            self.criterionHis.cuda()
            self.criterionGAN.cuda()
            self.criterionL1.cuda()
            self.criterionL2.cuda()
            self.D_A.cuda()
            self.D_B.cuda()

        super(Solver, self).__init__()

    # For generator
    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)


    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)
    

    def generate(self, org_A, ref_B, lms_A=None, lms_B=None, mask_A=None, mask_B=None, 
                 diff_A=None, diff_B=None, gamma=None, beta=None, ret=False):
        """org_A is content, ref_B is style"""
        res = self.G(org_A, ref_B, mask_A, mask_B, diff_A, diff_B, gamma, beta, ret)
        return res

    # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose
    # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck

    def test(self, real_A, mask_A, diff_A, real_B, mask_B, diff_B):
        cur_prama = None
        with torch.no_grad():
            cur_prama = self.generate(real_A, real_B, None, None, mask_A, mask_B, 
                                      diff_A, diff_B, ret=True)
            fake_A = self.generate(real_A, real_B, None, None, mask_A, mask_B, 
                                   diff_A, diff_B, gamma=cur_prama[0], beta=cur_prama[1])
        fake_A = fake_A.squeeze(0)

        # normalize
        min_, max_ = fake_A.min(), fake_A.max()
        fake_A.add_(-min_).div_(max_ - min_ + 1e-5)

        return ToPILImage()(fake_A.cpu())

    def train(self):
        # The number of iterations per epoch
        self.iters_per_epoch = len(self.data_loader_train)
        # Start with trained model if exists
        g_lr = self.g_lr
        d_lr = self.d_lr
        start = 0

        for self.e in range(start, self.num_epochs):
            for self.i, (source_input, reference_input) in enumerate(self.data_loader_train):
                # image, mask, dist
                image_s, image_r = source_input[0].to(self.device), reference_input[0].to(self.device)
                mask_s, mask_r = source_input[1].to(self.device), reference_input[1].to(self.device) 
                dist_s, dist_r = source_input[2].to(self.device), reference_input[2].to(self.device)
                self.track("data")

                # ================== Train D ================== #
                # training D_A, D_A aims to distinguish class B
                # Real
                out = self.D_A(image_r)
                self.track("D_A")
                d_loss_real = self.criterionGAN(out, True)
                self.track("D_A_loss")
                # Fake
                fake_A = self.G(image_s, image_r, mask_s, mask_r, dist_s, dist_r)
                self.track("G")
                fake_A = Variable(fake_A.data).detach()
                out = self.D_A(fake_A)
                self.track("D_A_2")
                d_loss_fake =  self.criterionGAN(out, False)
                self.track("D_A_loss_2")

                # Backward + Optimize
                d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
                self.d_A_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                self.d_A_optimizer.step()

                # Logging
                self.loss = {}
                self.loss['D_A_loss'] = d_loss.item()


                # training D_B, D_B aims to distinguish class A
                # Real
                out = self.D_B(image_s)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                self.track("G-before")
                fake_B = self.G(image_r, image_s, mask_r, mask_s, dist_r, dist_s)
                self.track("G-2")
                fake_B = Variable(fake_B.data).detach()
                out = self.D_B(fake_B)
                d_loss_fake =  self.criterionGAN(out, False)

                # Backward + Optimize
                d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
                self.d_B_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                self.d_B_optimizer.step()

                # Logging
                self.loss['D_B_loss'] = d_loss.item()

               
                # ================== Train G ================== #
                if (self.i + 1) % self.g_step == 0:

                    # identity loss
                    assert self.lambda_idt > 0
                    
                    # G should be identity if ref_B or org_A is fed
                    idt_A = self.G(image_s, image_s, mask_s, mask_s, dist_s, dist_s)
                    idt_B = self.G(image_r, image_r, mask_r, mask_r, dist_r, dist_r)
                    loss_idt_A = self.criterionL1(idt_A, image_s) * self.lambda_A * self.lambda_idt
                    loss_idt_B = self.criterionL1(idt_B, image_r) * self.lambda_B * self.lambda_idt
                    # loss_idt
                    loss_idt = (loss_idt_A + loss_idt_B) * 0.5
           

                    # Adversarial loss D_A(G_A(A))
                    # fake_A in class B, 
                    fake_A = self.G(image_s, image_r, mask_s, mask_r, dist_s, dist_r)
                    pred_fake = self.D_A(fake_A)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)

                    # Adversarial loss D_B(G_B(B))
                    fake_B = self.G(image_r, image_s, mask_r, mask_s, dist_r, dist_s)
                    pred_fake = self.D_B(fake_B)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)

                    loss_adv = g_A_loss_adv + g_B_loss_adv

           
                    # makeup loss
                    g_A_loss_his = 0
                    g_B_loss_his = 0
                    g_A_lip_loss_his = self.criterionHis(
                        fake_A, image_r, mask_s[:, 0], mask_r[:, 0]
                    ) * self.lambda_his_lip
                    g_B_lip_loss_his = self.criterionHis(
                        fake_B, image_s, mask_r[:, 0], mask_s[:, 0]
                    ) * self.lambda_his_lip
                    g_A_loss_his += g_A_lip_loss_his
                    g_B_loss_his += g_B_lip_loss_his

                    g_A_skin_loss_his = self.criterionHis(
                        fake_A, image_r, mask_s[:, 1], mask_r[:, 1]
                    ) * self.lambda_his_skin
                    g_B_skin_loss_his = self.criterionHis(
                        fake_B, image_s, mask_r[:, 1], mask_s[:, 1]
                    ) * self.lambda_his_skin
                    g_A_loss_his += g_A_skin_loss_his
                    g_B_loss_his += g_B_skin_loss_his

                    g_A_eye_loss_his = self.criterionHis(
                        fake_A, image_r, mask_s[:, 2], mask_r[:, 2]
                    ) * self.lambda_his_eye
                    g_B_eye_loss_his = self.criterionHis(
                        fake_B, image_s, mask_r[:, 2], mask_s[:, 2]
                    ) * self.lambda_his_eye
                    g_A_loss_his += g_A_eye_loss_his
                    g_B_loss_his += g_B_eye_loss_his

                    loss_makeup = g_A_loss_his + g_B_loss_his

                    # cycle loss
                    rec_A = self.G(fake_A, image_s, mask_s, mask_s, dist_s, dist_s)
                    g_loss_rec_A = self.criterionL1(rec_A, image_s) * self.lambda_A


                    rec_B = self.G(fake_B, image_r, mask_r, mask_r, dist_r, dist_r)
                    g_loss_rec_B = self.criterionL1(rec_B, image_r) * self.lambda_B

                    loss_cycle = (g_loss_rec_A + g_loss_rec_B) * 0.5


                    # perceptual loss
                    vgg_s = self.vgg(image_s)
                    vgg_s = Variable(vgg_s.data).detach()
                    vgg_fake_A = self.vgg(fake_A)
                    g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_s) * self.lambda_A * self.lambda_vgg

                    vgg_r = self.vgg(image_r)
                    vgg_r = Variable(vgg_r.data).detach()
                    vgg_fake_B = self.vgg(fake_B)
                    g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_r) * self.lambda_B * self.lambda_vgg

                    loss_percept = (g_loss_A_vgg + g_loss_B_vgg) * 0.5


                    # Combined loss
                    g_loss = (loss_idt + loss_adv + loss_cycle + loss_percept + loss_makeup).mean()

                    self.g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=False)
                    self.g_optimizer.step()

                    # Logging
                    self.loss['G_loss'] = g_loss.item()
                    self.loss['G_adv_loss'] = loss_adv.mean().item()
                    self.loss['G_cycle_loss'] = loss_cycle.mean().item()
                    self.loss['G_percept_loss'] = loss_percept.mean().item()
                    self.loss['G_makeup_loss'] = loss_makeup.mean().item()
                    self.loss['G_idt_loss'] = loss_idt.mean().item()


                # Print out log info
                if (self.i + 1) % self.log_step == 0:
                    self.log_terminal()

                #plot the figures
                for key_now in self.loss.keys():
                    plot_fig.plot(key_now, self.loss[key_now])

                #save the images
                if (self.i) % self.vis_step == 0:
                    print("Saving middle output...")
                    self.vis_train([image_s, image_r, fake_A, rec_A, mask_s[:, :, 0], mask_r[:, :, 0]])

                # Save model checkpoints
                if (self.i) % self.snapshot_step == 0:
                    self.save_models()

                if (self.i % 100 == 99):
                    plot_fig.flush(self.log_path)

                plot_fig.tick()

            # Decay learning rate
            if (self.e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_A_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.d_B_optimizer.param_groups:
            param_group['lr'] = d_lr

    def save_models(self):
        if not osp.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)
        torch.save(
            self.G.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.D_A.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_D_A.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.D_B.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_D_B.pth'.format(self.e + 1, self.i + 1)))

    def vis_train(self, img_train_list):
        # saving training results
        mode = "train_vis"
        img_train_list = torch.cat(img_train_list, dim=3)
        result_path_train = osp.join(self.result_path, mode)
        if not osp.exists(result_path_train):
            os.makedirs(result_path_train)
        save_path = os.path.join(result_path_train, '{}_{}_fake.jpg'.format(self.e, self.i))
        save_image(self.de_norm(img_train_list.data), save_path, normalize=True)

    def log_terminal(self):
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
            elapsed, self.e+1, self.num_epochs, self.i+1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)
