import torch
import torch.nn as nn
import argparse
from inference import Inference
import os.path as osp
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms

def inference_main(config):
    
    inference = Inference(
    	config)
    
    inference.inference()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', help = 'Data_path' , type=str, default='./sample')
    parser.add_argument('--num_workers' , type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/Joint_S3D_Fres.pth')

    parser.add_argument('--Input_Full', help='Use input of full angular resolution', action='store_true')

    parser.add_argument('--pred_height', type=int, default=512)
    parser.add_argument('--pred_width', type=int, default=1024)

    parser.add_argument('--output_path', help = 'path where inferenced samples saved' , type=str, default='./output')
   
    config = parser.parse_args()
    inference_main(config)


