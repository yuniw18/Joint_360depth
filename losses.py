import torch
import torch.nn as nn
import torch.nn.functional as F

# Code below from https://github.com/isl-org/MiDaS
def compute_scale_and_shift(prediction, target, mask):
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
    valid = det.nonzero(as_tuple=True)

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero(as_tuple=True)

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_image_based):
    M = torch.sum(mask, (1, 2))
    res = torch.abs(prediction - target)
    image_loss = torch.sum(mask * res   , (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_image_based):

    M = torch.sum(mask, (1, 2))

 
    diff = prediction - target
    
    
    diff = torch.mul(mask, diff)
    
    grad_x = torch.abs(diff[:,:, :, 1:] - diff[:,:, :, :-1])
    mask_x = torch.mul(mask[:,:, :, 1:], mask[:,:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:,:, 1:, :] - diff[:,:, :-1, :])
    mask_y = torch.mul(mask[:,:, 1:, :], mask[:,:, :-1, :])

    grad_y = torch.mul(mask_y, grad_y)
    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))[:,1:]

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='image-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='image-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based',Image_align=False):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha
        self.Image_align = Image_align
        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        
        if self.Image_align:
            prediction = prediction.squeeze(1)
            target = target.squeeze(1)
            mask = mask.squeeze(1)
        
        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scale = scale.unsqueeze(1).unsqueeze(1)
        shift = shift.unsqueeze(1).unsqueeze(1)

        self.__prediction_ssi = scale * prediction + shift
        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)

# Code below from https://github.com/gdlg/panoramic-depth-estimation & https://github.com/mrharicot/monodepth

def SSIM(x, y):
    
    refl = nn.ReflectionPad2d(1)
    
    x = refl(x)
    y = refl(y)
        
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)

    sigma_x  = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)

def gradient_x(img):
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:,:,:,:-1] - img[:,:,:,1:]
    return gx

def gradient_y(img):
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:,:,:-1,:] - img[:,:,1:,:]
    return gy

def get_disparity_smoothness(disp, pyramid):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.config.num_scales)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.config.num_scales)]
    
    return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(1)]
    
    
