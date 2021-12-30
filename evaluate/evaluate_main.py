import torch
import torch.nn as nn
import argparse
from evaluate import Evaluation
from network import *
from util import mkdirs, set_caffe_param_mult
from data_loader import S3D_loader,OmniDepthDataset,Stanford_loader
import os.path as osp
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms


def evaluate_main(config):
    
    transform = transforms.Compose([
            transforms.ToTensor()])
 
    
    input_dir = ""
    device_ids = [0]


    if config.eval_data == "3D60":
        val_loader = OmniDepthDataset(root_path=input_dir, path_to_img_list=config.data_path)

    elif config.eval_data == 'Structure3D':
        val_loader = S3D_loader(config.data_path,transform = transform,transform_t = transform)

    elif config.eval_data == 'Stanford':
        val_loader = Stanford_loader(config.data_path,transform = transform,transform_t = transform)
    else:
        print("Check the command option")
# -------------------------------------------------------
    device = torch.device('cuda', device_ids[0])

    val_dataloader = torch.utils.data.DataLoader(
    	val_loader,
        batch_size=1,
	    shuffle=False,
    	num_workers=config.num_workers,
	    drop_last=False)

    evaluation = Evaluation(
    	config,
        val_dataloader, 
	    device)
    
    if config.method == "EBS":
        evaluation.evaluate_EBS()
    elif config.method == "SvSyn":
        evaluation.evaluate_svsyn()
    elif config.method == "JointDepth":
        evaluation.evaluate_jointdepth()
    elif config.method == "Bifuse":
        evaluation.evaluate_bifuse()
    elif config.method == "Hohonet":
        evaluation.evaluate_hohonet()
    elif config.method == "Omnidepth":
        evaluation.evaluate_omnidepth()
    else:
        print("Check Command options")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## JointDepth -> Proposed model 
    parser.add_argument("--method",
                                 type=str,
                                 help="method to be evaluated",
                                 choices=["JointDepth", "Bifuse", "Hohonet", "Omnidepth", "SvSyn", "EBS"],
                                 default="JointDepth")
    
    parser.add_argument("--eval_data",
                                 type=str,
                                 help="data category to be evaluated",
                                 choices=["3D60", "Structure3D", "Stanford"],
                                 default="3D60")

    
    # For 3D60 testset -> data_path should be text file containing image list
    # For Structure3D and Stanford testset -> data_path should be folder containing image/video frames
    parser.add_argument('--data_path', help = 'Data_path' , type=str, default='')

    parser.add_argument('--num_workers' , type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str, default='rectnet.pth')
    parser.add_argument('--save_sample', help= 'save sampled results', action='store_true')
    parser.add_argument('--output_path', help = 'path where inferenced samples saved' , type=str, default='output')
    parser.add_argument('--pre_crop', help= 'crop image before input to the network ', action='store_true') # Used for 'Hres' pre-trained model 
    
    #### Used for HoHoNet evaluatoin ####
    parser.add_argument('--cfg',default='./HoHoNet/config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1.yaml')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
  
    config = parser.parse_args()
    evaluate_main(config)


