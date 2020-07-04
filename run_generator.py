# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
from tqdm import tqdm_notebook as tqdm
import scipy.interpolate as interpolate
from opensimplex import OpenSimplex
import os

import pretrained_networks

# from https://colab.research.google.com/drive/1ShgW6wohEFQtqs_znMna3dzrcVoABKIH
def generate_zs_from_seeds(seeds,Gs):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        zs.append(z)
    return zs

def generate_images_from_seeds(seeds, truncation_psi):
    return generate_images(generate_zs_from_seeds(seeds), truncation_psi)

def convertZtoW(latent, truncation_psi=0.7, truncation_cutoff=9):
    dlatent = Gs.components.mapping.run(latent, None) # [seed, layer, component]
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]
    for i in range(truncation_cutoff):
        dlatent[0][i] = (dlatent[0][i]-dlatent_avg)*truncation_psi + dlatent_avg
    
    return dlatent

def generate_latent_images(zs, truncation_psi,save_npy,prefix):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)
    
    for z_idx, z in enumerate(zs):
        if isinstance(z,list):
          z = np.array(z).reshape(1,512)
        elif isinstance(z,np.ndarray):
          z.reshape(1,512)
        print('Generating image for step %d/%d ...' % (z_idx, len(zs)))
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('%s%05d.png' % (prefix,z_idx)))
        if save_npy:
          np.save(dnnlib.make_run_dir_path('%s%05d.npy' % (prefix,z_idx)), z)

def generate_images_in_w_space(dlatents, truncation_psi,save_npy,prefix):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]

    # temp_dir = 'frames%06d'%int(1000000*random.random())
    # os.system('mkdir %s'%temp_dir)

    for row, dlatent in enumerate(dlatents):
        print('Generating image for step %d/%d ...' % (row, len(dlatents)))
        #row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
        dl = (dlatent-dlatent_avg)*truncation_psi   + dlatent_avg
        row_images = Gs.components.synthesis.run(dlatent,  **Gs_kwargs)
        PIL.Image.fromarray(row_images[0], 'RGB').save(dnnlib.make_run_dir_path('frame%05d.png' % row))
        if save_npy:
            np.save(dnnlib.make_run_dir_path('%s%05d.npy' % (prefix,row)), dlatent)

def line_interpolate(zs, steps):
   out = []
   for i in range(len(zs)-1):
    for index in range(steps):
     fraction = index/float(steps) 
     out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
   return out

def truncation_traversal(network_pkl,npys,seed=[0],start=-1.0,stop=1.0,increment=0.1):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    count = 1
    trunc = start

    while trunc <= stop:
        Gs_kwargs.truncation_psi = trunc
        print('Generating truncation %0.2f' % trunc)

        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]        
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('frame%05d.png' % count))

        trunc+=increment
        count+=1



#----------------------------------------------------------------------------

def generate_images(network_pkl, seeds, npy_files, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    if seeds is not None:
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx+1, len(seeds)))
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
            images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))
        
    if npy_files is not None:
        npys = npy_files.split(',')
        dlatent_avg = Gs.get_var('dlatent_avg') # [component]
        
        for npy in range(len(npys)):
            print('Generating image from npy (%d/%d) ...' % (npy+1, len(npys)))
            w = np.load(npys[npy])
            print(w.shape)
            rnd = np.random.RandomState(1)
            dl = (w-dlatent_avg)*truncation_psi   + dlatent_avg
            images = Gs.components.synthesis.run(w,  **Gs_kwargs) # [minibatch, height, width, channel]
            name = os.path.basename(npys[npy])
            PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('%s.png' % name))
        
     
        
#----------------------------------------------------------------------------

def generate_neighbors(network_pkl, seeds, npys, diameter, truncation_psi, num_samples, save_vector):
    global _G, _D, Gs, noise_vars
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx+1, len(seeds)))
        rnd = np.random.RandomState(seed)
        
        og_z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(og_z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))
        
        zs = []
        z_prefix = 'seed%04d_neighbor' % seed

        for s in range(num_samples):
            random = np.random.uniform(-diameter,diameter,[1,512])
#             zs.append(np.clip((og_z+random),-1,1))
            new_z = np.clip(np.add(og_z,random),-1,1)
            images = Gs.run(new_z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('%s%04d.png' % (z_prefix,s)))
            # generate_latent_images(zs, truncation_psi, save_vector, z_prefix)
            if save_vector:
                np.save(dnnlib.make_run_dir_path('%s%05d.npy' % (z_prefix,s)), new_z)


#----------------------------------------------------------------------------

def valmap(value, istart, istop, ostart, ostop):
  return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))

class OSN():
  min=-1
  max= 1

  def __init__(self,seed,diameter):
    self.tmp = OpenSimplex(seed)
    self.d = diameter
    self.x = 0
    self.y = 0

  def get_val(self,angle):
    self.xoff = valmap(np.cos(angle), -1, 1, self.x, self.x + self.d);
    self.yoff = valmap(np.sin(angle), -1, 1, self.y, self.y + self.d);
    return self.tmp.noise2d(self.xoff,self.yoff)

def get_noiseloop(endpoints, nf, d, start_seed):
    features = []
    zs = []
    for i in range(512):
      features.append(OSN(i+start_seed,d))

    inc = (np.pi*2)/nf
    for f in range(nf):
      z = np.random.randn(1, 512)
      for i in range(512):
        z[0,i] = features[i].get_val(inc*f) 
      zs.append(z)

    return zs
        
def get_latent_interpolation_bspline(endpoints, nf, k, s, shuffle):
    if shuffle:
        random.shuffle(endpoints)
    x = np.array(endpoints)
    x = np.append(x, x[0,:].reshape(1, x.shape[1]), axis=0)
    
    nd = x.shape[1]
    latents = np.zeros((nd, nf))
    nss = list(range(1, 10)) + [10]*(nd-19) + list(range(10,0,-1))
    for i in tqdm(range(nd-9)):
        idx = list(range(i,i+10))
        tck, u = interpolate.splprep([x[:,j] for j in range(i,i+10)], k=k, s=s)
        out = interpolate.splev(np.linspace(0, 1, num=nf, endpoint=True), tck)
        latents[i:i+10,:] += np.array(out)
    latents = latents / np.array(nss).reshape((512,1))
    return latents.T

def generate_latent_walk(network_pkl, truncation_psi, walk_type, frames, seeds, npys, save_vector, diameter=2.0, start_seed=0 ):
    global _G, _D, Gs, noise_vars
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    zs = []
    
    if(len(seeds) > 0):
        zs = generate_zs_from_seeds(seeds,Gs)
    elif(len(npys) > 0):
        zs = npys
        
    if(len(zs) > 2 ):
        print('not enough values to generate walk')
#         return false;

    walk_type = walk_type.split('-')
    
    if walk_type[0] == 'line':
        number_of_steps = int(frames/(len(zs)-1))+1
    
        if (len(walk_type)>1 and walk_type[1] == 'w'):
          ws = []
          for i in range(len(zs)):
            ws.append(convertZtoW(zs[i]))
          points = line_interpolate(ws,number_of_steps)
          zpoints = line_interpolate(zs,number_of_steps)
        else:
          points = line_interpolate(zs,number_of_steps)

    # from Gene Kogan
    elif walk_type[0] == 'bspline':
        # bspline in w doesnt work yet
        # if (len(walk_type)>1 and walk_type[1] == 'w'):
        #   ws = []
        #   for i in range(len(zs)):
        #     ws.append(convertZtoW(zs[i]))

        #   print(ws[0].shape)
        #   w = []
        #   for i in range(len(ws)):
        #     w.append(np.asarray(ws[i]).reshape(512,18))
        #   points = get_latent_interpolation_bspline(ws,frames,3, 20, shuffle=False)
        # else:
          z = []
          for i in range(len(zs)):
            z.append(np.asarray(zs[i]).reshape(512))
          points = get_latent_interpolation_bspline(z,frames,3, 20, shuffle=False)

    # from Dan Shiffman: https://editor.p5js.org/dvs/sketches/Gb0xavYAR
    elif walk_type[0] == 'noiseloop':
        points = get_noiseloop(None,frames,diameter,start_seed)

    if (walk_type[0] == 'line' and len(walk_type)>1 and walk_type[1] == 'w'):
      # print(points[0][:,:,1])
      # print(zpoints[0][:,1])
      # ws = []
      # for i in enumerate(len(points)):
      #   ws.append(convertZtoW(points[i]))
      generate_images_in_w_space(points, truncation_psi,save_vector,'frame')
    elif (len(walk_type)>1 and walk_type[1] == 'w'):
      print('%s is not currently supported in w space, please change your interpolation type' % (walk_type[0]))
    else:
      generate_latent_images(points, truncation_psi,save_vector,'frame')

#----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d.png' % (row_seed, col_seed)))

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(dnnlib.make_run_dir_path('grid.png'))

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]


#----------------------------------------------------------------------------

def _parse_npy_files(files):
    '''Accept a comma separated list of npy files and return a list of z vectors.'''
    print(files)
    zs =[]
    
    for f in files:
        zs.append(np.load(files[f]))
        
    return zs
        
#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=66,230,389,1518 --truncation-psi=1.0

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=6000-6025 --truncation-psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0

  # Generate truncation animation from one seed
  python %(prog)s truncation_traversal --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seed=6600 --start=-1.0 --stop=1.0 --increment=0.1

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_truncation_traversal = subparsers.add_parser('truncation-traversal', help='Generate truncation walk')
    parser_truncation_traversal.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_truncation_traversal.add_argument('--seed', type=_parse_num_range, help='Singular seed value')
    parser_truncation_traversal.add_argument('--npys', type=_parse_npy_files, help='List of .npy files')
    parser_truncation_traversal.add_argument('--start', type=float, help='Starting value')
    parser_truncation_traversal.add_argument('--stop', type=float, help='Stopping value')
    parser_truncation_traversal.add_argument('--increment', type=float, help='Incrementing value')
    parser_truncation_traversal.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_generate_latent_walk = subparsers.add_parser('generate-latent-walk', help='Generate latent walk')
    parser_generate_latent_walk.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_latent_walk.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_latent_walk.add_argument('--walk-type', help='Type of walk (default: %(default)s)', default='line')
    parser_generate_latent_walk.add_argument('--frames', type=int, help='Frame count (default: %(default)s', default=240)
    parser_generate_latent_walk.add_argument('--seeds', type=_parse_num_range, help='List of random seeds')
    parser_generate_latent_walk.add_argument('--npys', type=_parse_npy_files, help='List of .npy files')
    parser_generate_latent_walk.add_argument('--save_vector', dest='save_vector', action='store_true', help='also save vector in .npy format')
    parser_generate_latent_walk.add_argument('--diameter', type=float, help='diameter of noise loop', default=2.0)
    parser_generate_latent_walk.add_argument('--start_seed', type=int, help='random seed to start noise loop from', default=0)
    parser_generate_latent_walk.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_generate_neighbors = subparsers.add_parser('generate-neighbors', help='Generate random neighbors of a seed')
    parser_generate_neighbors.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_neighbors.add_argument('--seeds', type=_parse_num_range, help='List of random seeds')
    parser_generate_neighbors.add_argument('--npys', type=_parse_npy_files, help='List of .npy files')
    parser_generate_neighbors.add_argument('--diameter', type=float, help='distance around seed to sample from', default=0.1)
    parser_generate_neighbors.add_argument('--save_vector', dest='save_vector', action='store_true', help='also save vector in .npy format')
    parser_generate_neighbors.add_argument('--num_samples', type=int, help='How many neighbors to generate (default: %(default)s', default=25)
    parser_generate_neighbors.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_neighbors.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    
    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--seeds', type=_parse_num_range, help='List of random seeds')
    parser_generate_images.add_argument('--npys', help='List of .npy files', dest='npy_files')
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mixing_example = subparsers.add_parser('style-mixing-example', help='Generate style mixing video')
    parser_style_mixing_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_example.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_example.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_style_mixing_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'truncation-traversal': 'run_generator.truncation_traversal',
        'generate-images': 'run_generator.generate_images',
        'generate-neighbors': 'run_generator.generate_neighbors',
        'generate-latent-walk': 'run_generator.generate_latent_walk',
        'style-mixing-example': 'run_generator.style_mixing_example'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
