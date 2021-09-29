import os
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
tflib.init_tf()

network = 'stylegan2_256x256_balanced.pkl'
network_dir = '/host/research/mcw/'
network = os.path.join(network_dir, network)

if os.path.isfile(network):
    print("Network found, calling load network...")
    _G, _D, Gs = pretrained_networks.load_networks(network)
else:
    print("network not found at: ", network)
    print("In that directory there is: ")
    print(os.listdir(network_dir))
    # exit 1

print(">>> Printing Loaded Stylegan Network <<<")
print(Gs.print_layers())

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

def make_latents(Gs, seed=0, sigma = 1.0, mu = 0.0):
    latents =  sigma * np.random.RandomState(seed).randn(1, Gs.input_shape[1]) + mu
    return latents

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input

# spherical linear interpolation (slerp)
def slerp(val, low, high):
	omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
	so = np.sin(omega)
	if so == 0:
		# L'Hopital's rule/LERP
		return (1.0-val) * low + val * high
	return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
 
 # uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10, slerp=True):
	ratios = np.linspace(0, 1, num=n_steps)
	vectors = list()
	for ratio in ratios:
		v = slerp(ratio, p1, p2) if slerp else (1.0 - ratio) * p1 + ratio * p2
		vectors.append(v)
	return np.asarray(vectors)

def make_images(slerp=True):
    image_A, image_B = generate_latent_points(input_dimensionality, 2)
    interpolated_points = interpolate_points(image_A, image_B)
    images = Gs.run(interpolated_points, None, **synthesis_kwargs)
    print("made images: ", len(images), images.shape)

def plot_grid(images_sg2_output):
    imgs = images_sg2_output
    num = len(imgs)
    num_cols = 3
    num_rows = (num // num_cols) + (num % num_cols)
    fig = plt.figure(figsize=(20,20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(num_rows, num_cols),  # creates 2x2 grid of axes
                 axes_pad=0.25,  # pad between axes in inch.
                 )
    for idx,(ax, im) in enumerate(zip(grid, imgs)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(idx)
    plt.show()

def plot_pair(one,two):
    plot_grid((one,two))

