import os
from torch.utils import data
from torchvision import transforms
import math
from PIL import ImageEnhance
import random
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import RandomSampler
import torchvision.transforms.functional as F
#from torchsampler import ImbalancedDatasetSampler
from imageio import imread
import numpy as np
from skimage.transform import rescale
from skimage import io


class EQUI_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None,train = True,KITTI = True,small_step = False):
            self.small_step = small_step 
            "makes directory list which lies in the root directory"
            if KITTI:
                dir_path = []
                dir_path.append(list(map(lambda x:os.path.join(root,x),os.listdir(root))))

                dir_path_deep=[]
                left_path=[]
                right_path=[]
                self.left_image_paths = []
                self.right_image_paths = []

                dir_sub_list = []
                for sub_dir_path in dir_path:
                    dir_sub_list.append([file for file in sub_dir_path if file.split('/')[5].startswith('Col')]) # Redesignate 'n' in file.split('/')[n] according to your folder location
                    dir_sub_list.sort()  
    
                # Until Column 
                for dir_list in dir_sub_list:
                    for dir_sub_sub in dir_list:
                        left_name = os.listdir(dir_sub_sub)
                        left_name.sort()
                    
                        for left_image_name in left_name:
                           self.left_image_paths.append(os.path.join(dir_sub_sub,left_image_name))         
                self.transform = transform
                self.transform_t = transform_t
                self.train = train
                self.KITTI = KITTI

    def data_aug(self,left,right):
        cont_rand = random.uniform(0.8,1.2)
        br_rand = random.uniform(0.8,1.2)        
        sat_rand = random.uniform(0.8,1.2)        
        hue_rand = random.uniform(-0.1,0.1)        

        left = F.adjust_brightness(left, br_rand)
        right = F.adjust_brightness(right, br_rand)

        left = F.adjust_contrast(left, cont_rand)
        right = F.adjust_contrast(right, cont_rand)

        left = F.adjust_saturation(left, sat_rand)
        right = F.adjust_saturation(right, sat_rand)

        left = F.adjust_hue(left, hue_rand)
        right = F.adjust_hue(right, hue_rand)

        if random.random() > 0.5:
            left = F.hflip(left)
            right = F.hflip(right)

        return left,right


    def __getitem__(self,index):
           
        if self.KITTI == True:
            if self.small_step:
                diff = random.randint(1,3)
            else:     
                diff = random.randint(10,15)

            sign = random.choice([1,-1])
            diff = diff * sign

            try:
                left_path = self.left_image_paths[index]
                right_path = self.left_image_paths[index + diff]
                if left_path.split('/')[5] != right_path.split('/')[5]: # Redesignate 'n' in file.split('/')[n] according to your folder location
                    left_path = self.left_image_paths[index]
                    right_path = self.left_image_paths[index - diff]
            except:
                left_path = self.left_image_paths[index]
                right_path = self.left_image_paths[index - diff]
 
                
            right = Image.open(right_path).convert('RGB')
            left = Image.open(left_path).convert('RGB')
            
            # do data augmentation
            if False:            
                left ,right = self.data_aug(left,right)


            data=[]
   
        if self.transform is not None:
            data.append(self.transform(left))
            data.append(self.transform(right))
   
        return data
    def __len__(self):
        
        return len(self.left_image_paths)

class S3D_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None, train = True,KITTI = True):
            "makes directory list which lies in the root directory"
            if KITTI:
                dir_path = list(map(lambda x:os.path.join(root,x),os.listdir(root)))
                dir_path_deep=[]
                left_path=[]
                right_path=[]
                self.image_paths = []
                self.depth_paths = []
                self.semantic_paths = []
                dir_sub_dir=[]
#                self.color_palette = create_color_palette() 

                for dir_sub in dir_path:
                    
                    sub_path = os.path.join(dir_sub,'2D_rendering')
                    sub_path_list = os.listdir(sub_path)
                    

                    for path in sub_path_list:
                        dir_sub_dir.append(os.path.join(sub_path,path,'panorama/full'))
                

                for final_path in dir_sub_dir:
                    self.image_paths.append(os.path.join(final_path,'rgb_rawlight.png'))                    
                    self.depth_paths.append(os.path.join(final_path,'depth.png'))                    
                    self.semantic_paths.append(os.path.join(final_path,'semantic.png'))                    
                    
                self.transform = transform
                self.transform_t = transform_t
                self.train = train
                self.KITTI = KITTI

#    def load_semantic(self,semantic_paths):
#        semantic = Image.open(semantic_paths)
#        na = np.array(semantic.convert('RGB'))
#        label = np.zeros([512,1024,3])
        
#        for idx,color in enumerate(self.color_palette):
#            label[na == color] = idx
        
#        return label[:,:,0]

    def __getitem__(self,index):
           
        if self.KITTI == True:
            
            image_path = self.image_paths[index]
            depth_path = self.depth_paths[index]
#            semantic_path = self.semantic_paths[index]
                
            image = Image.open(image_path).convert('RGB')
            depth = io.imread(depth_path,as_gray=True).astype(np.float)
#            semantic = self.load_semantic(semantic_path)
#            semantic_rgb = Image.open(semantic_path).convert('RGB')

            data=[]

        if self.transform is not None:
            data.append(self.transform(image))
            data.append(self.transform_t(depth))
#            data.append(self.transform(semantic))
#            data.append(self.transform(semantic_rgb))


        return data

    def __len__(self):
        
        return len(self.image_paths)


class Stanford_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None, train = True,KITTI = True):
            "makes directory list which lies in the root directory"
            if KITTI:
                dir_path = list(map(lambda x:os.path.join(root,x),os.listdir(root)))
                dir_path_deep=[]
                left_path=[]
                right_path=[]
                self.image_paths = []
                self.depth_paths = []
                dir_sub_dir=[]
 

                for dir_sub in dir_path:
                    
                    sub_path = os.path.join(dir_sub,'rgb')
                    subd_path = os.path.join(dir_sub,'depth')

                    sub_path_list = os.listdir(sub_path)
                    sub_path_list.sort()
                    for name in sub_path_list:
                        depth_name = name.split('rgb')[0] + 'depth.png'
                        self.image_paths.append(os.path.join(sub_path,name))                    
                        self.depth_paths.append(os.path.join(subd_path,depth_name))                    

                self.transform = transform
                self.transform_t = transform_t
                self.train = train
                self.KITTI = KITTI

    def __getitem__(self,index):
           
        if self.KITTI == True:
            
            image_path = self.image_paths[index]
            depth_path = self.depth_paths[index]
                
            image = Image.open(image_path).convert('RGB')
#            image = rescale(image,0.25)
            depth = imread(depth_path,as_gray=True).astype(np.int16)
            depth = rescale(depth,0.25)
            data=[]

        if self.transform is not None:
            data.append(self.transform(image))
            data.append(self.transform_t(depth))
        return data

    def __len__(self):
        
        return len(self.image_paths)


