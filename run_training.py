# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
 
import argparse
import copy
import os
import sys
 
import dnnlib
from dnnlib import EasyDict
 
from metrics.metric_defaults import metric_defaults
 
#----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument("--dataset", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--num_gpus", type=int)
parser.add_argument("--total_kimg", type=int)
parser.add_argument("--mirror_augment", type=bool)
parser.add_argument("--minibatch_size_base", type=int)
parser.add_argument("--resolution", type=int)
parser.add_argument("--config_id", type=str)
parser.add_argument("--gamma")
parser.add_argument("--resume_pkl", type=str)
parser.add_argument("--result_dir", type=str)

args = parser.parse_args()

_valid_configs = [
    # Table 1
    'config-a', # Baseline StyleGAN
    'config-b', # + Weight demodulation
    'config-c', # + Lazy regularization
    'config-d', # + Path length regularization
    'config-e', # + No growing, new G & D arch.
    'config-f', # + Large networks (default)
 
    # Table 2
    'config-e-Gorig-Dorig',   'config-e-Gorig-Dresnet',   'config-e-Gorig-Dskip',
    'config-e-Gresnet-Dorig', 'config-e-Gresnet-Dresnet', 'config-e-Gresnet-Dskip',
    'config-e-Gskip-Dorig',   'config-e-Gskip-Dresnet',   'config-e-Gskip-Dskip',
]
 
#----------------------------------------------------------------------------
 
train     = EasyDict(run_func_name='training.training_loop.training_loop') # Options for training loop.
G         = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
D         = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
G_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
G_loss    = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')      # Options for generator loss.
D_loss    = EasyDict(func_name='training.loss.D_logistic_r1')              # Options for discriminator loss.
sched     = EasyDict()                                                     # Options for TrainingSchedule.
grid      = EasyDict(size='8k', layout='random')                           # Options for setup_snapshot_image_grid().
sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().
 
train.data_dir = args.data_dir
train.total_kimg = args.total_kimg
train.mirror_augment = args.mirror_augment
train.image_snapshot_ticks = train.network_snapshot_ticks = 1
sched.G_lrate_base = sched.D_lrate_base = 0.002
sched.minibatch_size_base = args.minibatch_size_base
sched.minibatch_gpu_base = 4
D_loss.gamma = 10
metrics = []
metrics = [metric_defaults[x] for x in metrics]
desc = 'stylegan2'
 
desc += '-' + args.dataset
dataset_args = EasyDict(tfrecord_dir=args.dataset, resolution=args.resolution)
 
assert args.num_gpus in [1, 2, 4, 8]
sc.num_gpus = args.num_gpus
desc += '-%dgpu' % args.num_gpus
 
config_id = args.config_id

assert config_id in _valid_configs
desc += '-' + config_id
 
# Configs A-E: Shrink networks to match original StyleGAN.
if config_id != 'config-f':
    G.fmap_base = D.fmap_base = 8 << 10
 
# Config E: Set gamma to 100 and override G & D architecture.
if config_id.startswith('config-e'):
    D_loss.gamma = 100
    if 'Gorig'   in config_id: G.architecture = 'orig'
    if 'Gskip'   in config_id: G.architecture = 'skip' # (default)
    if 'Gresnet' in config_id: G.architecture = 'resnet'
    if 'Dorig'   in config_id: D.architecture = 'orig'
    if 'Dskip'   in config_id: D.architecture = 'skip'
    if 'Dresnet' in config_id: D.architecture = 'resnet' # (default)
 
# Configs A-D: Enable progressive growing and switch to networks that support it.
if config_id in ['config-a', 'config-b', 'config-c', 'config-d']:
    sched.lod_initial_resolution = 8
    sched.G_lrate_base = sched.D_lrate_base = 0.001
    sched.G_lrate_dict = sched.D_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    sched.minibatch_size_base = 8 # (default)
    sched.minibatch_size_dict = {8: 256, 16: 128, 32: 64, 64: 32}
    sched.minibatch_gpu_base = 4 # (default)
    sched.minibatch_gpu_dict = {8: 32, 16: 16, 32: 8, 64: 4}
    G.synthesis_func = 'G_synthesis_stylegan_revised'
    D.func_name = 'training.networks_stylegan2.D_stylegan'
 
# Configs A-C: Disable path length regularization.
if config_id in ['config-a', 'config-b', 'config-c']:
    G_loss = EasyDict(func_name='training.loss.G_logistic_ns')
 
# Configs A-B: Disable lazy regularization.
if config_id in ['config-a', 'config-b']:
    train.lazy_regularization = False
 
# Config A: Switch to original StyleGAN networks.
if config_id == 'config-a':
    G = EasyDict(func_name='training.networks_stylegan.G_style')
    D = EasyDict(func_name='training.networks_stylegan.D_basic')
 
if args.gamma is not None:
    D_loss.gamma = args.gamma
 
sc.submit_target = dnnlib.SubmitTarget.LOCAL
sc.local.do_not_copy_source_files = True
 
if args.resume_pkl == ' ':
  resume_pkl = None
else:
  resume_pkl = args.resume_pkl
#----------------------------------------------------------------------------

def main():
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid,
                  metric_arg_list=metrics, tf_config=tf_config,resume_pkl=resume_pkl)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = args.result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)
 
#----------------------------------------------------------------------------
 
if __name__ == "__main__":
    main()
 
#----------------------------------------------------------------------------