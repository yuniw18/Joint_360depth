import torch
import torch.nn.functional as F
import time
import os
import math
import shutil
import os.path as osp
import matplotlib.pyplot as plt
import util
import torchvision
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
import matplotlib as mpl
import matplotlib.cm as cm
import argparse
import importlib
import numpy as np

## HoHoNet ##
from previous_works.HoHoNet.lib.misc.pano_lsd_align import rotatePanorama, panoEdgeDetection
from previous_works.HoHoNet.lib.config import config2, update_config, infer_exp_id
from previous_works.HoHoNet.lib import dataset

## Bifuse ##
from previous_works.BiFuse.models.FCRN import MyModel as ResNet
import previous_works.BiFuse.Utils 

## SvSyn ##
import previous_works.svsyn.models
import previous_works.svsyn.utils

## EBS ##
#from VGG import VGG_128

## Omnidepth ##
from previous_works.network import *

class BadPixelMetric:
    def __init__(self, threshold=1.25, depth_cap=10,data_type='None'):
        self.__threshold = threshold
        self.__depth_cap = depth_cap
        self.data_type = data_type

        ##### Column-wise scale-and-shift alignment ####    

    # From https://github.com/isl-org/MiDaS
    def compute_scale_and_shift(self, prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)
        
        det = a_00 * a_11 - a_01 * a_01
        
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def __call__(self, prediction, target, mask):

        ##### Column-wise scale-and-shift alignment ####    

        scale, shift = self.compute_scale_and_shift(prediction, target, mask)

        scale = scale.unsqueeze(0).unsqueeze(0)
        shift = shift.unsqueeze(0).unsqueeze(0)

        prediction_aligned = scale * prediction + shift

        depth_cap = self.__depth_cap
        
        prediction_aligned[prediction_aligned > depth_cap] = depth_cap
        gt = target
        pred = prediction_aligned

        ### refer to HoHoNet evaluation metric ###
        if self.data_type == 'Stanford':
            gt[gt<0.01] = 0.01
            pred[pred<0.01] = 0.01
        
        abs_rel_error = ((pred[mask>0] - gt[mask>0]).abs() / gt[mask>0]).mean()
        sq_rel_error = (((pred[mask>0] - gt[mask>0]) ** 2) / gt[mask>0]).mean()
        lin_rms_sq_error = ((pred[mask>0] - gt[mask>0]) ** 2).mean()
        mask_log = (mask > 0) & (pred > 1e-7) & (gt > 1e-7) # Compute a mask of valid values
        log_rms_sq_error = ((pred[mask_log].log() - gt[mask_log].log()) ** 2).mean()
        d1_ratio = (torch.max(pred[mask>0] / gt[mask>0], gt[mask>0] / pred[mask>0]) < (1.25 ** 1)).float().mean()
        d2_ratio = (torch.max(pred[mask>0] / gt[mask>0], gt[mask>0] / pred[mask>0]) < (1.25 ** 2)).float().mean()
        d3_ratio = (torch.max(pred[mask>0] / gt[mask>0], gt[mask>0] / pred[mask>0]) < (1.25 ** 3)).float().mean()

        err = torch.zeros_like(pred, dtype=torch.float)

        err[mask == 1] = torch.max(
            pred[mask == 1] / target[mask == 1],
            target[mask == 1] / pred[mask == 1],
        )

        err[mask == 1] = (err[mask == 1] > self.__threshold).float()

        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        return 100 * torch.mean(p), abs_rel_error, sq_rel_error,lin_rms_sq_error,log_rms_sq_error,d1_ratio,d2_ratio,d3_ratio

# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']

class Evaluation(object):

    def __init__(self,
                 config,
                 val_dataloader,
                 device):

        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        # Some timers
        self.batch_time_meter = AverageMeter()
        # Some trackers
        self.epoch = 0

        # Accuracy metric trackers
        self.abs_rel_error_meter = AverageMeter()
        self.sq_rel_error_meter = AverageMeter()
        self.lin_rms_sq_error_meter = AverageMeter()
        self.log_rms_sq_error_meter = AverageMeter()
        self.d1_inlier_meter = AverageMeter()
        self.d2_inlier_meter = AverageMeter()
        self.d3_inlier_meter = AverageMeter()

        # List of length 2 [Visdom instance, env]
        
        # Loss trackers
        self.loss = AverageMeter()
    def post_process_disparity(self,disp):
        disp = disp.cpu().detach().numpy()
        return disp   


    def evaluate_omnidepth(self):

        print('Evaluating Omnidepth')

        model = RectNet()
        alpha_list = [0.535, 0.272]
        beta_list = [0.134, 0.068,]

        device_ids = [0]
        device = self.device

        self.net = nn.DataParallel(
    	    model.float(),
	        device_ids=device_ids).to(device)

        self.load_checkpoint(self.net,self.config.checkpoint_path, True, True)
 
        self.net = self.net.eval()

        # Reset meter
        self.reset_eval_metrics()

        # Load data

        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(

                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')

                # Parse the data
                if self.config.eval_data == 'Structure':

                    inputs = data[0].float().cuda()
                    gt = data[1][:,:,128:384,:].float().cuda()
                    gt = gt / gt.max()
                
                elif self.config.eval_data == 'Stanford':
                    inputs = data[0].float().cuda()
                    inputs = F.interpolate(inputs,scale_factor=0.25)
                    gt = data[1][:,:,128:384,:].float().cuda()

                elif self.config.eval_data == '3D60':
                    inputs, gt, other = self.parse_data(data)


                # Run a forward pass
                output = self.net(*inputs)
                output[0] = output[0][:,:,128:384,:]
                # Compute the evaluation metrics
                self.compute_eval_metrics(output[0], gt)

        # Print a report on the validation results
        print('Evaluation finished in {} seconds'.format(time.time() - s))
        self.print_validation_report()


    def evaluate_hohonet(self):
        print('Evaluating Hohonet')

        # Load the checkpoint to evaluate
        
        update_config(config2, self.config)
        model_file = importlib.import_module(config2.model.file)
        model_class = getattr(model_file, config2.model.modelclass)
        self.net = model_class(**config2.model.kwargs).cuda()
        self.net.load_state_dict(torch.load(self.config.checkpoint_path))
        self.net.eval()
 
        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(
                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')

                # Parse the data
        
                if self.config.eval_data == 'Structure3D':
                    inputs = data[0].float().cuda()
                    gt = data[1][:,:,128:384,:].float().cuda()
                    gt = gt / gt.max()
                
                elif self.config.eval_data == 'Stanford':
                    inputs = data[0].float().cuda()
                    inputs = F.interpolate(inputs,scale_factor=0.25)
                    gt = data[1][:,:,128:384,:].float().cuda()
                
                elif self.config.eval_data == '3D60':    
                    inputs, gt, other = self.parse_data(data)

                # Run a forward pass
        
                output = self.net(*inputs)
                output = output.pop('depth')
                output = output[:,:,128:384,:]
                
                self.compute_eval_metrics(output, gt)

        # Print a report on the validation results
        print('Evaluation finished in {} seconds'.format(time.time() - s))
        self.print_validation_report()
 
    def evaluate_bifuse(self):
        print('Evaluating Bifuse')


        self.net = ResNet(
        		layers=50,
        		decoder="upproj",
    	    	output_size=None,
    		    in_channels=3,
        		pretrained=True
        		).cuda()
        
        self.net.load_state_dict(torch.load(self.config.checkpoint_path))
        self.net.cuda().eval()

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(
                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')

                # Parse the data
                if self.config.eval_data == 'Structure3D':
                    inputs = data[0].float().cuda()
                    gt = data[1][:,:,128:384,:].float().cuda()
                    gt = gt / gt.max()
                
                elif self.config.eval_data == 'Stanford':
                    inputs = data[0].float().cuda()
                    inputs = F.interpolate(inputs,scale_factor=0.25)
                    gt = data[1][:,:,128:384,:].float().cuda()
 
                elif self.config.eval_data == '3D60' :    
                    inputs, gt, other = self.parse_data(data)
      

                # Run a forward pass
                _,_,output = self.net(*inputs)
                output = output[:,:,128:384,:]
 
                output = torch.clamp(output,0,10)
                output = output / 10.
                # Compute the evaluation metrics
                self.compute_eval_metrics(output, gt)

        # Print a report on the validation results
        print('Evaluation finished in {} seconds'.format(time.time() - s))
        self.print_validation_report()

    def evaluate_svsyn(self):
        print('Evaluating SVSYN')

        model_params = { 'width': 512, 'configuration': 'mono'}
        self.net = previous_works.svsyn.models.get_model("resnet_coord", model_params)
        previous_works.svsyn.utils.init.initialize_weights(self.net,self.config.checkpoint_path, pred_bias=None)
 
        # Put the model in eval mode
        self.net.cuda().eval()

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(
                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')
                ## Due to resolution constraint, we upsample the predicted depths 

                if self.config.eval_data == 'Structure3D':
                    inputs = F.interpolate(data[0].float().cuda(),scale_factor=0.5)
                    gt = data[1][:,:,128:384,:].float().cuda()
                    gt = gt / gt.max()
                    output = F.interpolate(self.net(inputs),scale_factor=2)[:,:,128:384,:]

                elif self.config.eval_data == 'Stanford':
                    inputs = data[0].float().cuda()
                    inputs = F.interpolate(inputs,scale_factor=0.125).unsqueeze(0)
                    gt = data[1][:,:,128:384,:].float().cuda()
                    output = F.interpolate(self.net(*inputs),scale_factor=2)[:,:,128:384,:]

                elif self.config.eval_data == '3D60':    
                    inputs, gt, other = self.parse_data(data,model='svsyn')
                    output = F.interpolate(self.net(*inputs),scale_factor=2)[:,:,128:384,:]

                # Compute the evaluation metrics
                self.compute_eval_metrics(output, gt)

        # Print a report on the validation results
        print('Evaluation finished in {} seconds'.format(time.time() - s))
        self.print_validation_report()

    def evaluate_EBS(self):
        print('Evaluating EBS')

        # Put the model in eval mode
        self.net = VGG_128()
        self.net.load_weights('./numpy_weight')
 
        self.net.cuda()
        self.net.eval()

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(
                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')
      
                if self.config.eval_data == 'Structure3D':
                    inputs = data[0][:,:,128:384,:].float().cuda()
                    gt = data[1][:,:,128:384,:].float().cuda()
                    gt = gt / gt.max()
                    output = self.net(inputs,ring_pad=True,kitti=True)[0]
                    output = output[:,0,:,:].unsqueeze(0)
                    output = output               
                
                elif self.config.eval_data == 'Stanford':
                    inputs = data[0].float().cuda()
                    inputs = F.interpolate(inputs,scale_factor=0.25)[:,:,128:384,:]
                    gt = data[1][:,:,128:384,:].float().cuda()

                elif self.config.eval_data == '3D60':    
                    inputs, gt, other = self.parse_data(data,model='SelfEqui')
                    output = self.net(*inputs)[0].unsqueeze(0)
                    output = output[:,0,:,:]
 
                # Compute the evaluation metrics
                self.compute_eval_metrics(output, gt)


        # Print a report on the validation results
        print('Evaluation finished in {} seconds'.format(time.time() - s))
        self.print_validation_report()

    def evaluate_jointdepth(self):

        print('Evaluating JointDepth')

        # Put the model in eval mode
        self.net = DPTDepthModel(
                    path=None,
                    backbone="vitb_rn50_384",
                    non_negative=True,
                    enable_attention_hooks=False,
                        )

        self.net.load_state_dict(torch.load(self.config.checkpoint_path))
        self.net.cuda().eval()

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print(
                    'Evaluating {}/{}'.format(batch_num, len(
                        self.val_dataloader)),
                    end='\r')
      
                if self.config.eval_data == 'Structure3D':
                    if self.config.pre_crop:
                        inputs = data[0][:,:,128:384,:].float().cuda()
                    else:
                        inputs = data[0].float().cuda()
 
                    gt = data[1][:,:,128:384,:].float().cuda()
                    gt = gt / gt.max()

                elif self.config.eval_data == 'Stanford':
                    inputs = data[0].float().cuda()
                    inputs = F.interpolate(inputs,scale_factor=0.25)
                    if self.config.pre_crop:
                        inputs = inputs[:,:,128:384,:]
                    gt = data[1][:,:,128:384,:].float().cuda()

                elif self.config.eval_data == '3D60':    
                    inputs, gt, other = self.parse_data(data,model='SelfEqui')
                
                output = self.net(*inputs).unsqueeze(0)

                if not self.config.pre_crop:
                    output = output[:,:,128:384,:]

                self.compute_eval_metrics(output, gt)

        # Print a report on the validation results
        print('Evaluation finished in {} seconds'.format(time.time() - s))
        self.print_validation_report()


    def parse_data(self, data, model='default'):
        '''
        Returns a list of the inputs as first output, a list of the GT as a second output, and a list of the remaining info as a third output. Must be implemented.
        '''
        if model == 'SelfEqui':
            if self.config.pre_crop:
                data[0] = F.interpolate(data[0],scale_factor=2)[:,:,128:384,:]
            else:
                data[0] = F.interpolate(data[0],scale_factor=2)
        elif model == 'svsyn': 
            data[0] = data[0]
        
        else:
            data[0] = F.interpolate(data[0],scale_factor=2)
     
        data[1] = F.interpolate(data[1],scale_factor=2)[:,:,128:384,:]
        data[2] = F.interpolate(data[2],scale_factor=2)[:,:,128:384,:]
           
        rgb = data[0].to(self.device)
        gt_depth_1x = data[1].to(self.device)
        gt_depth_2x = F.interpolate(gt_depth_1x, scale_factor=0.5)
        gt_depth_4x = F.interpolate(gt_depth_1x, scale_factor=0.25)
        mask_1x = data[2].to(self.device)
        mask_2x = F.interpolate(mask_1x, scale_factor=0.5)
        mask_4x = F.interpolate(mask_1x, scale_factor=0.25)

        inputs = [rgb]
        gt = [gt_depth_1x, mask_1x, gt_depth_2x, mask_2x, gt_depth_4x, mask_4x]
        other = data[3]

        return inputs, gt, other

    def reset_eval_metrics(self):
        '''
        Resets metrics used to evaluate the model
        '''
        self.abs_rel_error_meter.reset()
        self.sq_rel_error_meter.reset()
        self.lin_rms_sq_error_meter.reset()
        self.log_rms_sq_error_meter.reset()
        self.d1_inlier_meter.reset()
        self.d2_inlier_meter.reset()
        self.d3_inlier_meter.reset()
        
        self.is_best = False

    def compute_eval_metrics(self, output, gt):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output

        if self.config.eval_data == 'Structure3D':
            gt_depth = gt
            depth_mask = (gt>0).cuda()
        
        elif self.config.eval_data == 'Stanford':
            gt_depth = gt
            depth_mask = (gt>0).cuda()
        
        else:      
            gt_depth = gt[0]
            depth_mask = gt[1]

      
       
        Bmetric = BadPixelMetric(depth_cap=100,data_type=self.config.eval_data)
        Bloss = Bmetric(depth_pred,gt_depth,depth_mask)
        
        N = depth_mask.sum()

        abs_rel = Bloss[1]
        sq_rel = Bloss[2]
        rms_sq_lin = Bloss[3]
        rms_sq_log = Bloss[4]
        d1 = Bloss[5]
        d2 = Bloss[6]
        d3 = Bloss[7]

        self.abs_rel_error_meter.update(abs_rel, N)
        self.sq_rel_error_meter.update(sq_rel, N)
        self.lin_rms_sq_error_meter.update(rms_sq_lin, N)
        self.log_rms_sq_error_meter.update(rms_sq_log, N)
        self.d1_inlier_meter.update(d1, N)
        self.d2_inlier_meter.update(d2, N)
        self.d3_inlier_meter.update(d3, N)


    ## used for Omnidepth
    def load_checkpoint(self,
                        model,
                        checkpoint_path=None,
                        weights_only=False,
                        eval_mode=False):
        '''
        Initializes network with pretrained parameters
        '''
        if checkpoint_path is not None:
            print('Loading checkpoint \'{}\''.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

            # If we want to continue training where we left off, load entire training state
            if not weights_only:
                self.epoch = checkpoint['epoch']
                experiment_name = checkpoint['experiment']
                self.vis[1] = experiment_name
                self.best_d1_inlier = checkpoint['best_d1_inlier']
                self.loss.from_dict(checkpoint['loss_meter'])
            else:
                print('NOTE: Loading weights only')

            # Load the optimizer and model state
            if not eval_mode:
                util.load_optimizer(self.optimizer, checkpoint['optimizer'],
                                    self.device)
            util.load_partial_model(model, checkpoint['state_dict'])

            print('Loaded checkpoint \'{}\' (epoch {})'.format(
                checkpoint_path, checkpoint['epoch']))
        else:
            print('WARNING: No checkpoint found')


    def print_validation_report(self):
        '''
        Prints a report of the validation results
        '''
        print('Epoch: {}\n'
              '  Avg. Abs. Rel. Error: {:.4f}\n'
              '  Avg. Sq. Rel. Error: {:.4f}\n'
              '  Avg. Lin. RMS Error: {:.4f}\n'
              '  Avg. Log RMS Error: {:.4f}\n'
              '  Inlier D1: {:.4f}\n'
              '  Inlier D2: {:.4f}\n'
              '  Inlier D3: {:.4f}\n\n'.format(
                  self.epoch + 1, self.abs_rel_error_meter.avg,
                  self.sq_rel_error_meter.avg,
                  math.sqrt(self.lin_rms_sq_error_meter.avg),
                  math.sqrt(self.log_rms_sq_error_meter.avg),
                  self.d1_inlier_meter.avg, self.d2_inlier_meter.avg,
                  self.d3_inlier_meter.avg))


