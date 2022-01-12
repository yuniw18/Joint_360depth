import math
import torch
import torch.nn.functional as F


def bilinear_self_equi(input_image, depth, move_ratio, depth_sample = False, FOV_y = 90):
    batch = input_image.size(0)
    width = input_image.size(3)
    height = input_image.size(2)
    pi = torch.acos(torch.zeros(1)).item() * 2

    def read_depth(disp,disp_rescale=3.):
        return depth

    def offset(base, move_ratio):

        height = base.size(2)
        width = base.size(3)
        ###### spherical coordinate #####
        
        spherical_base = torch.zeros_like(base)
        spherical_base_shift = torch.zeros_like(base)

        ## restrict fov_y smaller than 180
        fov_y = FOV_y - 1 # 0 ~ 179

        spherical_base[:,2,:,:] = base[:,2,:,:] # rho = abs(depth)
        spherical_base[:,0,:,:] = ((base[:,0,:,:]/width) * 359 - 359/2 + 180) * pi / 180 # theta
        spherical_base[:,1,:,:] = ((base[:,1,:,:] / height) * fov_y - fov_y/2 + 90 ) * pi/ 180 # phi

        spherical_base_shift = spherical_base.clone()
        move_ratio_x = move_ratio[0]

############ According to the video data, change the options ###########       
        # we use video in which the camera moves in x-axis only without rotation

        move_ratio_y = 0
#        move_ratio_y = move_ratio[1]
        move_ratio_z = 0
#        move_ratio_z = move_ratio[2]
#######################################################################

        # Some optimization may require to reduce computational cost
        # Changes in roation is not reflected below. Modify them if needed.

        spherical_base_shift[:,0,:,:] = torch.atan2(torch.sin(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_y / spherical_base[:,2,:,:], torch.cos(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_x / spherical_base[:,2,:,:])
        spherical_base_shift[:,0,:,:] = torch.where(0> spherical_base_shift[:,0,:,:], spherical_base_shift[:,0, :, :] + 2 * pi, spherical_base_shift[:,0, :, :])

        theta_p = torch.atan2(torch.sin(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_y / spherical_base[:,2,:,:], torch.cos(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_x / spherical_base[:,2,:,:])
        theta_p = torch.where(0> theta_p, theta_p + 2 * pi, theta_p)
         
        spherical_base_shift[:,1,:,:] = torch.atan2(torch.sin(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_y/spherical_base[:,2,:,:] ,(torch.cos(spherical_base[:,1,:,:]) - move_ratio_z/spherical_base[:,2,:,:]) * torch.sin(theta_p))
        spherical_base_shift[:,1,:,:] = torch.where(spherical_base_shift[:,1, :, :] < 0, spherical_base_shift[:,1, :, :] + pi, spherical_base_shift[:,1, :, :])

        phi_p = torch.atan2(torch.sin(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_y/spherical_base[:,2,:,:] ,(torch.cos(spherical_base[:,1,:,:]) - move_ratio_z/spherical_base[:,2,:,:]) * torch.sin(theta_p))
        phi_p = torch.where(phi_p < 0, phi_p + pi, phi_p)

        ## Modify depth values when synthesizing depth from another viewpoint
        depth_synth =  (torch.cos(spherical_base[:,1,:,:]) * spherical_base[:,2,:,:] - move_ratio_z ) / torch.cos(phi_p) 
        
        # spherical 2 cartesian
        spherical_base_shift[:,0,:,:] = spherical_base_shift[:,0,:,:] * width / (pi * 2)
        spherical_base_shift[:,1,:,:] = (spherical_base_shift[:,1,:,:] * 180/pi + fov_y/2 -90) * height / fov_y
        

        return spherical_base_shift[:,0:2,:,:], depth_synth


    x_base = torch.linspace(0, width, width).repeat( height, 1).cuda().unsqueeze(0).unsqueeze(0).float().repeat(batch,1,1,1)
    y_base = torch.linspace(0, height, height).repeat(width, 1).transpose(0, 1).cuda().unsqueeze(0).unsqueeze(0).float().repeat(batch,1,1,1)
    depth = read_depth(depth)
    
    base = torch.cat((x_base,y_base,depth),dim = 1)

    base_shift, depth_synth = offset(base,move_ratio)
    
    base_shift[:,0,:,:] = base_shift[:,0,:,:] / width
    base_shift[:,1,:,:] = base_shift[:,1,:,:] / height
    base_shift = base_shift.permute(0,2,3,1)

    if depth_sample == True:
        input_image = depth_synth.unsqueeze(1)
    
    output = F.grid_sample(input_image, 2 * base_shift - 1  , mode='bilinear',
                               padding_mode='zeros')
    
    return output

