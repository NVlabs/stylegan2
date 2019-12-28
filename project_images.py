import argparse
import os
import shutil
import numpy as np

import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import projector
import dataset_tool
from training import dataset
from training import misc


def project_image(proj, src_file, dst_dir, tmp_dir, video=False):

    data_dir = '%s/dataset' % tmp_dir
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    image_dir = '%s/images' % data_dir
    tfrecord_dir = '%s/tfrecords' % data_dir
    os.makedirs(image_dir, exist_ok=True)
    shutil.copy(src_file, image_dir + '/')
    dataset_tool.create_from_images(tfrecord_dir, image_dir, shuffle=0)
    dataset_obj = dataset.load_dataset(
        data_dir=data_dir, tfrecord_dir='tfrecords',
        max_label_size=0, repeat=False, shuffle_mb=0
    )

    print('Projecting image "%s"...' % os.path.basename(src_file))
    images, _labels = dataset_obj.get_minibatch_np(1)
    images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
    proj.start(images)
    if video:
        video_dir = '%s/video' % tmp_dir
        os.makedirs(video_dir, exist_ok=True)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if video:
            filename = '%s/%08d.png' % (video_dir, proj.get_cur_step())
            misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)

    os.makedirs(dst_dir, exist_ok=True)
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.png')
    misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.npy')
    np.save(filename, proj.get_dlatents()[0])


def render_video(src_file, dst_dir, tmp_dir, num_frames, mode, size, fps, codec, bitrate):

    import PIL.Image
    import moviepy.editor

    def render_frame(t):
        frame = np.clip(np.ceil(t * fps), 1, num_frames)
        image = PIL.Image.open('%s/video/%08d.png' % (tmp_dir, frame))
        if mode == 1:
            canvas = image
        else:
            canvas = PIL.Image.new('RGB', (2 * src_size, src_size))
            canvas.paste(src_image, (0, 0))
            canvas.paste(image, (src_size, 0))
        if size != src_size:
            canvas = canvas.resize((mode * size, size), PIL.Image.LANCZOS)
        return np.array(canvas)

    src_image = PIL.Image.open(src_file)
    src_size = src_image.size[1]
    duration = num_frames / fps
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.mp4')
    video_clip = moviepy.editor.VideoClip(render_frame, duration=duration)
    video_clip.write_videofile(filename, fps=fps, codec=codec, bitrate=bitrate)


def main():

    parser = argparse.ArgumentParser(description='Project real-world images into StyleGAN2 latent space')
    parser.add_argument('src_dir', help='Directory with aligned images for projection')
    parser.add_argument('dst_dir', help='Output directory')
    parser.add_argument('--tmp-dir', default='.stylegan2-tmp', help='Temporary directory for tfrecords and video frames')
    parser.add_argument('--network-pkl', default='gdrive:networks/stylegan2-ffhq-config-f.pkl', help='StyleGAN2 network pickle filename')
    parser.add_argument('--vgg16-pkl', default='https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2', help='VGG16 network pickle filename')
    parser.add_argument('--num-steps', type=int, default=1000, help='Number of optimization steps')
    parser.add_argument('--initial-learning-rate', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--initial-noise-factor', type=float, default=0.05, help='Initial noise factor')
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose output')
    parser.add_argument('--video', type=bool, default=False, help='Render video of the optimization process')
    parser.add_argument('--video-mode', type=int, default=1, help='Video mode: 1 for optimization only, 2 for source + optimization')
    parser.add_argument('--video-size', type=int, default=1024, help='Video size (height in px)')
    parser.add_argument('--video-fps', type=int, default=25, help='Video framerate')
    parser.add_argument('--video-codec', default='libx264', help='Video codec')
    parser.add_argument('--video-bitrate', default='5M', help='Video bitrate')
    args = parser.parse_args()

    print('Loading networks from "%s"...' % args.network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(args.network_pkl)
    proj = projector.Projector(
        vgg16_pkl             = args.vgg16_pkl,
        num_steps             = args.num_steps,
        initial_learning_rate = args.initial_learning_rate,
        initial_noise_factor  = args.initial_noise_factor,
        verbose               = args.verbose
    )
    proj.set_network(Gs)

    src_files = sorted([os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir) if f[0] not in '._'])
    for src_file in src_files:
        project_image(proj, src_file, args.dst_dir, args.tmp_dir, video=args.video)
        if args.video:
            render_video(
                src_file, args.dst_dir, args.tmp_dir, args.num_steps, args.video_mode,
                args.video_size, args.video_fps, args.video_codec, args.video_bitrate
            )
        shutil.rmtree(args.tmp_dir)


if __name__ == '__main__':
    main()
