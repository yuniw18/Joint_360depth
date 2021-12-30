import os
from torch.utils import data
from torchvision import transforms
from PIL import ImageEnhance
import random
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import RandomSampler
import torchvision.transforms.functional as F
from imageio import imread
import numpy as np
from skimage import io
import OpenEXR, Imath, array
import math
import os.path as osp
import torch.utils.data
from skimage.transform import rescale,resize

# From https://github.com/meder411/OmniDepth-PyTorch
class OmniDepthDataset(torch.utils.data.Dataset):
	def __init__(self, 
		root_path, 
		path_to_img_list):

		# Set up a reader to load the panos
		self.root_path = root_path

		# Create tuples of inputs/GT
		self.image_list = np.loadtxt(path_to_img_list, dtype=str)

		# Max depth for GT
		self.max_depth = 8.0


	def __getitem__(self, idx):
		'''Load the data'''

		# Select the panos to load
		relative_paths = self.image_list[idx]

		# Load the panos
		relative_basename = osp.splitext((relative_paths[0]))[0]
		basename = osp.splitext(osp.basename(relative_paths[0]))[0]
		rgb = self.readRGBPano(osp.join(self.root_path, relative_paths[0]))
		depth = self.readDepthPano(osp.join(self.root_path, relative_paths[1]))
		depth_mask = ((depth <= self.max_depth) & (depth > 0.)).astype(np.uint8)

		# Threshold depths
		depth *= depth_mask

		# Make a list of loaded data
		pano_data = [rgb, depth, depth_mask, basename]

		# Convert to torch format
		pano_data[0] = torch.from_numpy(pano_data[0].transpose(2,0,1)).float()
		pano_data[1] = torch.from_numpy(pano_data[1][None,...]).float()
		pano_data[2] = torch.from_numpy(pano_data[2][None,...]).float()

		# Return the set of pano data
		return pano_data
		
	def __len__(self):
		'''Return the size of this dataset'''
		return len(self.image_list)

	def readRGBPano(self, path):
		'''Read RGB and normalize to [0,1].'''
		rgb = io.imread(path).astype(np.float32) / 255.
		return rgb


	def readDepthPano(self, path):
		return self.read_exr(path)[...,0].astype(np.float32)


	def read_exr(self, image_fpath):
		f = OpenEXR.InputFile( image_fpath )
		dw = f.header()['dataWindow']
		w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)    
		im = np.empty( (h, w, 3) )

		# Read in the EXR
		FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
		channels = f.channels( ["R", "G", "B"], FLOAT )
		for i, channel in enumerate( channels ):
			im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
		return im

class S3D_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None, train = True):
            "makes directory list which lies in the root directory"
            if True:
                dir_path = list(map(lambda x:os.path.join(root,x),os.listdir(root)))
                dir_path_deep=[]
                left_path=[]
                right_path=[]
                self.image_paths = []
                self.depth_paths = []
                dir_sub_dir=[]
 

                for dir_sub in dir_path:
                    
                    sub_path = os.path.join(dir_sub,'2D_rendering')
                    sub_path_list = os.listdir(sub_path)
                    

                    for path in sub_path_list:
                        dir_sub_dir.append(os.path.join(sub_path,path,'panorama/full'))
                

                for final_path in dir_sub_dir:
                    self.image_paths.append(os.path.join(final_path,'rgb_rawlight.png'))                    
                    self.depth_paths.append(os.path.join(final_path,'depth.png'))                    



                self.transform = transform
                self.transform_t = transform_t
                self.train = train

    def __getitem__(self,index):
           
        if True:
            
            image_path = self.image_paths[index]
            depth_path = self.depth_paths[index]
                
            image = Image.open(image_path).convert('RGB')
            depth = imread(depth_path,as_gray=True).astype(np.float)

            data=[]

        if self.transform is not None:
            data.append(self.transform(image))
            data.append(self.transform_t(depth))
        return data

    def __len__(self):
        
        return len(self.image_paths)

class Stanford_loader(data.Dataset):
    def __init__(self,root,transform = None,transform_t = None, train = True):
            "makes directory list which lies in the root directory"
            if True:
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

    def __getitem__(self,index):
           
        image_path = self.image_paths[index]
        depth_path = self.depth_paths[index]
                
        image = Image.open(image_path).convert('RGB')
        depth = imread(depth_path,as_gray=True).astype(np.int16)
        
        depth = rescale(depth,0.25)
        data=[]
        
        if self.transform is not None:
            data.append(self.transform(image))
            data.append(self.transform_t(depth))
        return data

    def __len__(self):
        
        return len(self.image_paths)


