import os
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
tflib.init_tf()

# path to stylegan2 tensorflow network
network_dir = '../'
network_file = 'stylegan2_256x256_balanced.pkl'
network_path = os.path.join(network_dir, network_file)

if os.path.isfile(network_path):
    print("Network found, calling load network...")
    print(network_path)
    _G, _D, Gs = pretrained_networks.load_networks(network_path)
else:
    print(network_file, " not found")
    print("In that directory there is: ")
    print(os.listdir(network_dir))
    # exit 1

print(">>> Printing Loaded Stylegan Network <<<")
print(Gs.print_layers())

synthesis_kwargs = dict(output_transform=dict(
    func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)


# def make_latents(Gs, seed=0, sigma=1.0, mu=0.0):
#     latents = sigma * \
#         np.random.RandomState(seed).randn(1, Gs.input_shape[1]) + mu
#     return latents

def generate_latent_points(latent_dim, n_samples, n_classes=10, seed=-1):
    if seed == -1:
        seed = np.random.randint(0, 100000)
    print("Using seed: ", seed)
    np.random.RandomState(seed)
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

def slerp_points(val, low, high):
    omega = np.arccos(
        np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val) * low + val * high
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def interpolate_points(p1, p2, n_steps=10, slerp=True):
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = list()
    for ratio in ratios:
        if slerp:
            # spherical linear interpolation (slerp)
            v = slerp_points(ratio, p1, p2)
        else:
            # uniform interpolation between two points in latent space
            v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    print(f"Created {len(vectors)} latent points.")
    return np.asarray(vectors)

def slerp_experiment_make_images(input_dimensionality=512, seed=-1, n_steps=12):
    '''
    Returns (slerp imgs, linear interp imgs), where imgs is array [ImageA, ...InterpolationFrames, ImageB]
    '''
    print(">>> Making images <<<")
    image_A, image_B = generate_latent_points(
        input_dimensionality, 2, seed=seed)
    interpolated_points_slerp = interpolate_points(image_A, image_B, n_steps, slerp=True)
    interpolated_points_linear = interpolate_points(image_A, image_B, n_steps, slerp=False)
    print("Calling generator inference.")
    images_slerp = Gs.run(interpolated_points_slerp, None, **synthesis_kwargs)
    images_linear = Gs.run(interpolated_points_linear, None, **synthesis_kwargs)
    print("Done.")
    return (images_slerp, images_linear)


def plot_comparison(imgs_a, imgs_b, label_a="slerp", label_b="lin"):
    assert len(imgs_a) == len(imgs_b)
    num = len(imgs_a)
    num_cols = num
    num_rows = 2
    print(f"Plotting {num} images.")
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(num_rows, num_cols),
                     axes_pad=0.25
                     )
    for idx, ax in enumerate(grid):
        ax.set_axis_off()
        if idx < num:
            ax.set_title(f"{label_a} {idx}")
            ax.imshow(imgs_a[idx])
        else:
            ax.set_title(f"{label_b} {idx - num}")
            ax.imshow(imgs_b[idx - num])
    plt.show()
    print("Done.")

def plot_grid(imgs):
    num = len(imgs)
    num_cols = 3
    num_rows = (num // num_cols) + (num % num_cols)
    fig = plt.figure(figsize=(30, 20))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(num_rows, num_cols),
                     axes_pad=0.25,
                     )
    for idx, (ax, im) in enumerate(zip(grid, imgs)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(idx)
    plt.show()


def plot_pair(one, two):
    plot_grid((one, two))
