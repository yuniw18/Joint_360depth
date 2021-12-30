import os
from yacs.config import CfgNode as CN

config2 = CN()

config2.ckpt_root = 'ckpt'
config2.cuda = True
config2.cuda_benchmark = True
config2.num_workers = 8

config2.dataset = CN()
config2.dataset.name = 'PanoCorBonDataset'
config2.dataset.common_kwargs = CN(new_allowed=True)
config2.dataset.train_kwargs = CN(new_allowed=True)
config2.dataset.valid_kwargs = CN(new_allowed=True)

config2.training = CN()
config2.training.epoch = 300
config2.training.batch_size = 4
config2.training.save_every = 100
config2.training.optim = 'Adam'
config2.training.optim_lr = 0.0001
config2.training.optim_betas = (0.9, 0.999)
config2.training.weight_decay = 0.0
config2.training.wd_group_mode = 'bn and bias'
config2.training.optim_milestons = [0.5, 0.9]
config2.training.optim_gamma = 0.2
config2.training.optim_poly_gamma = -1.0
config2.training.fix_encoder_bn = False

config2.model = CN()
config2.model.file = 'previous_works.HoHoNet.lib.model.HorizonNet'
config2.model.modelclass = 'HorizonNet'
config2.model.kwargs = CN(new_allowed=True)


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

def infer_exp_id(cfg_path):
    cfg_path = cfg_path.split('config/')[-1]
    if cfg_path.endswith('.yaml'):
        cfg_path = cfg_path[:-len('.yaml')]
    return '_'.join(cfg_path.split('/'))

