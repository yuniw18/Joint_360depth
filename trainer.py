import math
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch.nn as nn
import scipy.misc
import matplotlib.pyplot as plt
import cv2
from bilinear import *
from torch import optim
import ThreeD60
from torch.autograd import Variable
import OpenEXR
import Imath
import array
import matplotlib as mpl
import matplotlib.cm as cm
from posenet import PoseCNN
import argparse
import random
from imageio import imread
import skimage
import skimage.transform
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from losses import ScaleAndShiftInvariantLoss, SSIM, get_disparity_smoothness

class Train(object):
    def __init__(self,config,supervised_loader,equi_loader):
        self.posenet = None
        
        self.model_name = config.model_name
        self.model_path = os.path.join(config.model_name,config.model_path)
        
        self.sample_path = os.path.join(self.model_name,config.sample_path)
        self.log_path = os.path.join(self.model_name,'log.txt')
        self.eval_path = os.path.join(self.model_name, config.eval_path)
        
        self.supervised_loader = supervised_loader
        self.equi_loader = equi_loader
        
        self.config = config

        self.depthnet = None
        self.posenet = None
        
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.sample_path):
            os.mkdir(self.sample_path)
        if not os.path.exists(self.eval_path):
            os.mkdir(self.eval_path)

        self.build_model()

    def build_model(self):

        self.depthnet = DPTDepthModel(
                    path = None,
                    backbone="vitb_rn50_384",
                    non_negative=True,
                    enable_attention_hooks=False,
                        )

        self.posenet = PoseCNN(2)
        self.gT_optimizer = optim.Adam([{"params": list(self.depthnet.parameters())}],
                                        self.config.lr,[self.config.beta1,self.config.beta2])

        self.g_optimizer = optim.Adam([
                {"params": list(self.depthnet.parameters()),"lr": self.config.lr / self.config.self_lr_ratio }],
                                        self.config.lr,[self.config.beta1,self.config.beta2])

        if self.config.pose_only:
            self.g_optimizer = optim.Adam([{"params":list(self.posenet.parameters())}], self.config.lr,[self.config.beta1,self.config.beta2])
   
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, 0.95)
  
        if torch.cuda.is_available():
            self.depthnet.cuda()
            self.posenet.cuda()                 

    def to_variable(self,x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
 
    def parse_data(self, data):
        rgb = data[0].to(self.device)
        gt_depth_1x = data[1].to(self.device)
        gt_depth_2x = F.interpolate(gt_depth_1x, scale_factor=0.5)
        gt_depth_4x = F.interpolate(gt_depth_1x, scale_factor=0.25)
        mask_1x = data[2].to(self.device)
        mask_2x = F.interpolate(mask_1x, scale_factor=0.5)
        mask_4x = F.interpolate(mask_1x, scale_factor=0.25)

        inputs = [rgb]
        gt = [gt_depth_1x, mask_1x, gt_depth_2x, mask_2x, gt_depth_4x, mask_4x]

        return inputs, gt

    def reset_grad(self):
        self.depthnet.zero_grad()
        self.posenet.zero_grad()
        
    def post_process_disparity(self,disp):
        disp = disp.cpu().detach().numpy() 
        _, h, w = disp.shape
        l_disp = disp[0,:,:]
        
        return l_disp

    def generate_image_left_equi(self, img, disp, move_ratio, depth_sample = True):

        fov_y = 360 * (img.size(2) / img.size(3))
        
        output = bilinear_self_equi(img, disp, move_ratio,depth_sample = depth_sample,FOV_y = fov_y)
        return output
 
    def generate_image_right_equi(self, img, disp, move_ratio, depth_sample = True):

        fov_y = 360 * (img.size(2) / img.size(3))
        ## Instead of using inverse camera movements, using posenet output on inverse input directly may produce better results
        # move_r = move_ratio 

        ######## Check this part######### 
        # According to the camera movements in the video data, change this part 
        move_r = []
        move_r.append( - move_ratio[0])
        move_r.append( - move_ratio[1])
        ################################
        
        output = bilinear_self_equi(img, disp, move_r, depth_sample = depth_sample,FOV_y = fov_y)
        
        return output

    def scale_pyramid(self,img,num_scales):
        scaled_imgs = [img]
        height = img.size(2)
        width = img.size(3) 
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            scaled_imgs.append(F.interpolate(img,size=(height//ratio,width//ratio)))
        return scaled_imgs

    def Random_Data(self):
        if (not self.config.Video):
            random_data = random.choice(['Supervised'])
        else:
            random_data = random.choice(['Supervised','Video'])

        return random_data

    def post_crop_data(self,left_est,right_est,disp_left_est,disp_right_est,left_pyramid,right_pyramid,lr_est,rl_est):
        height = left_est[0].size(2)
        fovy_ratio = self.config.self_crop_ratio
        
        # crop top/bottom part of the image/depth
        crop_ratio = int(height * (fovy_ratio) //2)
        
        for i in range(self.config.num_scales):
            left_est[i] = left_est[i][:,:,crop_ratio:height - crop_ratio,:]
            right_est[i] = right_est[i][:,:,crop_ratio:height - crop_ratio,:]
            disp_left_est[i] = disp_left_est[i][:,:,crop_ratio:height - crop_ratio,:]
            disp_right_est[i] = disp_right_est[i][:,:,crop_ratio:height - crop_ratio,:]
            left_pyramid[i] = left_pyramid[i][:,:,crop_ratio: height - crop_ratio,:]
            right_pyramid[i] = right_pyramid[i][:,:,crop_ratio:height - crop_ratio,:]
            lr_est[i] = lr_est[i][:,:,crop_ratio: height - crop_ratio,:]
            rl_est[i] = rl_est[i][:,:,crop_ratio:height - crop_ratio,:]
            
        return left_est,right_est,disp_left_est,disp_right_est,left_pyramid,right_pyramid,lr_est,rl_est 

    def train(self):
        if os.path.isfile(self.log_path):
            os.remove(self.log_path)  
        
        self.depthnet.load_state_dict(torch.load(self.config.pretrained_gen_path))
        self.posenet.load_state_dict(torch.load(self.config.pretrained_posenet_path))

        self.max_depth = 255.0
        max_batch_num = len(self.supervised_loader) - 1

        self.scale_loss = ScaleAndShiftInvariantLoss(Image_align=self.config.Image_align).cuda()

        with torch.no_grad(): 
            eval_name = 'SAMPLE_%d' %(0)
            self.sample(self.config.val_path,'test',eval_name,self.config.eval_crop_ratio)

        for epoch in range(self.config.num_epochs):
            for batch_num, data in enumerate(self.supervised_loader):

                with torch.autograd.set_detect_anomaly(True):
                    
                    random_data = self.Random_Data()
                    
                    if random_data == 'Video':
                        ## modify the code below as to call iter() only once per each epoch for faster data loading
                        data_kitti = next(iter(self.equi_loader))
                        
                        if self.config.pre_crop:
                            crop_height = int(self.config.input_height * (self.config.self_crop_ratio) // 2)
                        else:
                            crop_height=0

                        left = self.to_variable(data_kitti[0][:,:,crop_height:self.config.input_height - crop_height,:])
                        right = self.to_variable(data_kitti[1][:,:,crop_height:self.config.input_height - crop_height,:])

                        left_pyramid = self.scale_pyramid(left,self.config.num_scales)
                        right_pyramid = self.scale_pyramid(right,self.config.num_scales)
                        
                        move_ratio = self.posenet(left,right)
                        move_ratio_r = self.posenet(right,left)

                        pred_depth = self.depthnet(left).unsqueeze(1)
                        pred_depth_right = self.depthnet(right).unsqueeze(1)

                        disp_left_est = [pred_depth]
                        disp_right_est = [pred_depth_right]

                        # Synthesizing images on different viewpoint
                        left_est = [self.generate_image_left_equi(left_pyramid[i],disp_left_est[i],move_ratio, depth_sample = False) for i in range(self.config.num_scales)] 
                        right_est = [self.generate_image_right_equi(right_pyramid[i],disp_right_est[i],move_ratio, depth_sample = False) for i in range(self.config.num_scales)]
                        
                        # Synthesizing depths on different viewpoint
                        right_to_left_disp = [self.generate_image_left_equi(disp_left_est[i],disp_left_est[i],move_ratio, depth_sample = True) for i in range(self.config.num_scales)] 
                        left_to_right_disp = [self.generate_image_right_equi(disp_right_est[i],disp_right_est[i],move_ratio, depth_sample = True) for i in range(self.config.num_scales)] 
 
                        # disparity smoothness -> not used
#                        disp_left_smoothness = get_disparity_smoothness(disp_left_est,left_pyramid)
#                        disp_right_smoothness = get_disparity_smoothness(disp_right_est,right_pyramid)

                        # Crop Top-down part of estimated depth when training -> due to captured undesirable object 
                        if self.config.post_crop:
                            left_est,right_est,disp_left_est,disp_right_est,left_pyramid,right_pyramid,left_to_right_disp,right_to_left_disp = self.post_crop_data(left_est,right_est,disp_left_est,disp_right_est,left_pyramid,right_pyramid,left_to_right_disp,right_to_left_disp)

                ########## buliding losses #########
                        l1_reconstruction_loss_left = [torch.mean(l) for l in [torch.abs(left_est[i] - right_pyramid[i]) for i in range(self.config.num_scales)]]
                        l1_reconstruction_loss_right = [torch.mean(l) for l in [torch.abs(right_est[i] - left_pyramid[i]) for i in range(self.config.num_scales)]]
    
                        ssim_loss_left = [torch.mean(s) for s in [SSIM(left_est[i], right_pyramid[i]) for i in range(self.config.num_scales)]]
                        ssim_loss_right = [torch.mean(s) for s in [SSIM(right_est[i], left_pyramid[i]) for i in range(self.config.num_scales)]]

                        # Image Consistency loss
                        image_loss_right = [self.config.alpha_image_loss * ssim_loss_right[i] + (1 - self.config.alpha_image_loss) * l1_reconstruction_loss_right[i] for i in range(self.config.num_scales)]
                        image_loss_left  = [self.config.alpha_image_loss * ssim_loss_left[i]  + (1 - self.config.alpha_image_loss) * l1_reconstruction_loss_left[i]  for i in range(self.config.num_scales)]

                        image_loss = (image_loss_left + image_loss_right)

                        # Depth consistency loss
                        lr_loss = [torch.mean(torch.abs(right_to_left_disp[i] - disp_right_est[i]))  for i in range(self.config.num_scales)] + [torch.mean(torch.abs(left_to_right_disp[i] - disp_left_est[i])) for i in range(self.config.num_scales)]

                        # DISPARITY SMOOTHNESS -> not used 
#                        disp_left_loss  = [torch.mean(torch.abs(disp_left_smoothness[i]))  / 2 ** i for i in range(self.config.num_scales)]
#                        disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(self.config.num_scales)]
#                        disp_gradient_loss = disp_left_loss + disp_right_loss

                        # Pose Consistency loss                                                    
                        pose_loss = torch.mean(torch.abs(move_ratio[0] + move_ratio_r[0]))

                    if random_data == 'Supervised':
                        if self.config.train_data == 'Structure3D':
                            data = next(iter(self.supervised_loader))
                            if self.config.pre_crop:
                               crop_height = int(self.config.input_height * (self.config.super_crop_ratio) // 2)
                            else:
                                crop_height=0

                            inputs = self.to_variable(data[0][:,crop_height:self.config.input_height - crop_height,:])

                            gt = self.to_variable(data[1][:,crop_height:self.config.input_height - crop_height,:])
                            mask = self.to_variable(torch.ones_like(gt))
                            gt = gt / 32768.

                            pred_depth = self.depthnet(inputs).unsqueeze(1)
                            image_loss = self.scale_loss(pred_depth,gt,mask)

                        elif self.config.train_data == '3D60':
                            data = next(iter(self.supervised_loader))

                            if self.config.pre_crop:
                                crop_height = int(self.config.input_height * (self.config.super_crop_ratio) // 2)
                            else:
                                crop_height = 0

                            inputs = ThreeD60.extract_image(data,ThreeD60.Placements.CENTER,ThreeD60.ImageTypes.COLOR)[:,:,crop_height:self.config.input_height - crop_height,:]
                            gt = ThreeD60.extract_image(data,ThreeD60.Placements.CENTER,ThreeD60.ImageTypes.DEPTH)[:,:,crop_height:self.config.input_height - crop_height,:]

                            if self.config.super_resize_ratio == 1:
                                inputs = self.to_variable(inputs)
                                gt = self.to_variable(gt)

                            else:
                                inputs = self.to_variable(F.interpolate(inputs,scale_factor=self.config.super_resize_ratio))
                                gt = self.to_variable(F.interpolate(gt,scale_factor=self.config.super_resize_ratio))
                            
                            mask = self.to_variable(((gt <= self.max_depth) & (gt > 0.)).to(torch.float32))
                            gt = gt / 255. 

                            pred_depth = self.depthnet(inputs).unsqueeze(1)
                            image_loss = self.scale_loss(pred_depth,gt,mask)

                        elif self.config.train_data == 'Stanford':
                            data = next(iter(self.supervised_loader))
          
                            inputs = data[0].float().cuda()
                            
                            # To reduce data-loading time, resize Stanford dataset (512x1024) in advance before running the codes.
                            inputs = F.interpolate(inputs,scale_factor=0.25)
                            inputs = self.to_variable(inputs)

                            ## scales of ground truth is modified when loading data ##
                            gt = self.to_variable(data[1].float().cuda())

                            mask = self.to_variable(((gt <= self.max_depth) & (gt > 0.)).to(torch.float32))

                            gt = gt / 255. 
                            
                            pred_depth = self.depthnet(inputs).unsqueeze(1)
                            image_loss = self.scale_loss(pred_depth,gt,mask)

                    
                    ### Back propagate ###
                    total_loss = 0
                    self.reset_grad()                   

                    if random_data == 'Video':
                        for i in range(self.config.num_scales):
                            total_loss += image_loss[i] + self.config.lr_loss_weight * lr_loss[i] 
                        total_loss = total_loss * self.config.self_loss_weight        
                        
                        try: 
                            total_loss.backward()
                            self.g_optimizer.step()
                        except:
                            print('skip')
                            self.g_optimizer.zero_grad()                           
                        
                    else:
                        total_loss = image_loss
                        total_loss.backward()
                        self.gT_optimizer.step()

                if (batch_num) % self.config.log_step == 0:
                    if random_data == 'Video': 
                        if self.config.DEBUG:
                            torchvision.utils.save_image(left_pyramid[0],os.path.join(self.sample_path,'left_pyramid-%d.png' %(epoch)))
                            torchvision.utils.save_image(left_est[0].data,os.path.join(self.sample_path,'left_est_samples-%d.png' %(epoch)))
                            torchvision.utils.save_image(right_pyramid[0].data,os.path.join(self.sample_path,'right_pyramid-%d.png' %(epoch)))
                            torchvision.utils.save_image(disp_left_est[0].data * 20  ,os.path.join(self.sample_path,'disp_left_est-%d.png' %(epoch)))

                        print('Epoch [%d/%d], Step[%d/%d], image_loss: %.4f, lr_loss: %.4f, pose_loss: %.4f' 
                          %(epoch, self.config.num_epochs, batch_num, max_batch_num, 
                            image_loss[0].item(),lr_loss[0].item(), pose_loss.item()))
                    else:
                        if self.config.DEBUG:
                            torchvision.utils.save_image(inputs.data,os.path.join(self.sample_path,'inputs-%d.png' %(epoch)))
                            torchvision.utils.save_image(kitti_output.data * 300,os.path.join(self.sample_path,'output-%d.png' %(epoch)))
                            torchvision.utils.save_image(gt.data , os.path.join(self.sample_path,'GT-%d.png' %(epoch)))
                        
                        print('Epoch [%d/%d], Step[%d/%d], image_loss: %.7f' 
                          %(epoch, self.config.num_epochs, batch_num, max_batch_num, 
                            image_loss.item()))
 
                if (batch_num) % self.config.sample_step == 0:
                    g_latest_path = os.path.join(self.model_path,'generator_latest.pkl')
                    g_path = os.path.join(self.model_path,'generator-%d.pkl' % (epoch + 1))
                    
                    eval_name = 'SAMPLE_%d' %(epoch)

                    with torch.no_grad():
                        self.sample(self.config.val_path,g_path,eval_name,self.config.eval_crop_ratio)
            
            g_path = os.path.join(self.model_path,'generator-%d.pkl' % (epoch + 1))
            p_path =  os.path.join(self.model_path,'pose-%d.pkl' % (epoch + 1))
                         
            torch.save(self.depthnet.state_dict(),g_path)
            if self.config.pose_only:
                torch.save(self.posenet.state_dict(),p_path)
            
            with torch.no_grad():
#                self.lr_scheduler.step()
                eval_name = 'SAMPLE_%d' %(epoch)
                self.sample(self.config.val_path,g_path,eval_name,self.config.eval_crop_ratio)

        
    ### Estimate the depths of the samples ###
    
    def sample(self,root,checkpoint_path,eval_name,eval_crop_ratio):
        image_list = os.listdir(root)
        eval_image = []
        for image_name in image_list:
            eval_image.append(os.path.join(root,image_name))
        
        index = 0  
        for image_path in eval_image:
            index = index + 1
 
            input_image = (imread(image_path).astype("float32")/255.0)
            original_height, original_width, num_channels = input_image.shape
        
            input_height = 512
            input_width = 1024
            
            crop_ratio = int(input_height * eval_crop_ratio // 2)

            input_image = skimage.transform.resize(input_image, [input_height, input_width])

            input_image = input_image.astype(np.float32)
            input_image = torch.from_numpy(input_image).unsqueeze(0).float().permute(0,3,1,2).cuda()

            input_image = input_image[:,:,crop_ratio:input_height - crop_ratio,:]     

            disp = self.depthnet(input_image)

            # disp to depth

            if True:
                disp_comp = disp.unsqueeze(0)
                disp_comp = disp_comp[:,0,:,:]

                max_value = torch.tensor([0.000005]).cuda()
                disp_comp =1. / torch.max(disp_comp,max_value)

            disp_pp = self.post_process_disparity(disp_comp).astype(np.float32)

            pred_width = disp_pp.shape[1]
            original_crop_ratio = int(original_height * eval_crop_ratio // 2)
            disp_pp = cv2.resize(disp_pp.squeeze(), (original_width, original_height - original_crop_ratio * 2))

            disp_pp = disp_pp.squeeze()

            vmax = np.percentile(disp_pp, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_pp.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            disp_pp = (mapper.to_rgba(disp_pp)[:, :, :3] * 255).astype(np.uint8)

            save_name = eval_name + '_'+str(index)+'.png'        
    
            plt.imsave(os.path.join(self.eval_path,save_name ), disp_pp, cmap='magma')
