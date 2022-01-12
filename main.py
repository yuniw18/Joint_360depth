import argparse
import os
from trainer import Train
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from data_load import EQUI_loader, S3D_loader, Stanford_loader
from torchvision import transforms
import torch
import ThreeD60

def main(config):
    cudnn.benchmark = True
    
    transform = transforms.Compose([
                    transforms.Resize((config.input_height,config.input_width)),
                    transforms.ToTensor()
                    ])

    transform_s3d = transforms.Compose([
                    transforms.ToTensor()
                    ])
     
    if config.train_data == '3D60':
        ThreeD_loader = ThreeD60.get_datasets(config.ThreeD_path, \
                datasets=["suncg","m3d", "s2d3d"],
                placements=[ThreeD60.Placements.CENTER,ThreeD60.Placements.RIGHT,ThreeD60.Placements.UP],
                image_types=[ThreeD60.ImageTypes.COLOR, ThreeD60.ImageTypes.DEPTH, ThreeD60.ImageTypes.NORMAL], longitudinal_rotation=True)
        supervised_dataloader = DataLoader(ThreeD_loader,batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers,pin_memory=True)

    elif config.train_data == 'Structure3D':
        S3D_data = S3D_loader(config.S3D_path,transform = transform_s3d,transform_t = transform_s3d)
        supervised_dataloader = DataLoader(S3D_data,batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers,pin_memory=True)
    
    
    elif config.train_data == 'Stanford':
        S_loader = Stanford_loader(config.Stanford_path,transform = transform_s3d,transform_t = transform_s3d)
        supervised_dataloader = DataLoader(S_loader,batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers,pin_memory=True,drop_last=True)

       

    if config.Video:
        data_path = []
        equi_loader = [] 
        len_data = []
        weights = []
        data_path = list(map(lambda x:os.path.join(config.Video_path,x),os.listdir(config.Video_path)))
        inx = 0
        for i,sub_data_path in enumerate(data_path):
            if config.WILD:
                equi_loader.append(EQUI_loader(sub_data_path,transform,small_step =(sub_data_path.split('_')[-1] == 'wild')))
                for index in range(len(equi_loader[i])):
                    value = 1. / len(equi_loader[i])
                    weights.append(value)
            else:
                if sub_data_path.split('_')[-1] != 'wild':
                    equi_loader.append(EQUI_loader(sub_data_path,transform,small_step =(sub_data_path.split('_')[-1] == 'wild')))
                    for index in range(len(equi_loader[inx])):
                        value = 1. / len(equi_loader[inx])
                        weights.append(value)
                    inx = inx + 1

        Full_dataset = ConcatDataset(equi_loader)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(Full_dataset),replacement=False)

        Equi_dataloader = DataLoader(Full_dataset,batch_size=config.batch_size,num_workers=config.num_workers,sampler = sampler,pin_memory=True)
    
    if config.mode == 'train':
        if not config.Video:
            Equi_dataloader = None
        train = Train(config,supervised_dataloader,Equi_dataloader)
        train.train()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--ThreeD_path',help='Text file containing the image list', type=str,default='../3D60/splits/filtered/final_train.txt') # train image list file for supervised learning.
    parser.add_argument('--Video_path',help='Folder containing video frames',type=str,default='../Video_folder') # train image list file for supervised learning
    parser.add_argument('--S3D_path',help='Folder containing Structure3D dataset', type=str,default='../Structure3D/Structured3D') # train image list file for supervised learning.
    parser.add_argument('--Stanford_path',help='Folder containing Stanford dataset', type=str,default='../Structure3D/Structured3D') # train image list file for supervised learning.
 
    parser.add_argument('--val_path',type=str,default='./inference/sample') # file path which contains images to be sampled

    parser.add_argument('--pretrained_gen_path',type=str,default='./pretrained/generator.pkl') # detphnet checkpoint path
    parser.add_argument('--pretrained_posenet_path',type=str,default='./pretrained/posenet.pkl') # posenet checkpoint path
 
    ##### hyper-parameters #####
    parser.add_argument('--lr_loss_weight',            type=float, help='LR Depth consistency weight', default=0.5)
    parser.add_argument('--num_scales',type=int, help='number of scales', default=1)
    parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
    parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0) # not used in the experiments. 
    parser.add_argument('--self_loss_weight', type=float, help='self-supervised loss weight', default=0.03) 
    
    # Use either pre_crop or post_crop
    parser.add_argument('--pre_crop', help='Crop Top-bottom part of an image before input to the depth network ', action='store_true')
    parser.add_argument('--post_crop', help='Crop Top-bottom part of depth & image after the depthnet prediction', action='store_true')

    parser.add_argument('--super_crop_ratio',help='Ratio to be cropped for supervised learning', type=float, default=0)
    parser.add_argument('--self_crop_ratio',help='Ratio to be cropped for self-supervised learning', type=float, default=0.5)
 
    parser.add_argument('--self_lr_ratio',help='learning ratio of self-supervised learning (compared to the supervised learning)', type=float, default=3)

    # Not implemented  
    parser.add_argument('--WILD', help='USE Wild video for self-supervised learning', action='store_true')


    parser.add_argument('--pose_only', help='Train pose network only', action='store_true')
    parser.add_argument('--Image_align', help='Do Image-wise depth align instead of column-wise align', action='store_true')

    ## Input height/width of the video data
    parser.add_argument('--input_height', type=int, help='input height', default=256)
    parser.add_argument('--input_width', type=int, help='input width', default=512)

    ## resize image data for supervised learning
    parser.add_argument('--super_resize_ratio', type=int , default=1)
   
    parser.add_argument("--train_data",
                                 type=str,
                                 help="data to be used for supervised learning",
                                 choices=["3D60","Stanford","Structure3D"],
                                 default="Structure3D")

    parser.add_argument('--Video', help='Use Video data for training (self_supervised learning)' , action='store_true')

    parser.add_argument('--DEBUG', help='Save estimated depths of training data for debugging' , action='store_true')


    ##### training environment #####
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    
    ############## Directory ############## 
    parser.add_argument('--model_name',help='path where models are to be saved' , type=str, default='./checkpoints/default') 
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--eval_path', type=str, default='evaluate')

    ############ Set log step ############
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=100)
    parser.add_argument('--checkpoint_step', type=int , default=10000)
    parser.add_argument('--eval_crop_ratio', type=float , default=0)
    
    config = parser.parse_args()
    
    config_path = os.path.join(config.model_name,'config.txt')
    f = open(config_path,'w')
    print(config,file=f)
    f.close()
 
    print(config)
    main(config)
