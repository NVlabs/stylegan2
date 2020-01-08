# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Linear Separability (LS)."""

from collections import defaultdict
import numpy as np
import sklearn.svm
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

classifier_urls = [
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-00-male.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-01-smiling.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-02-attractive.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-03-wavy-hair.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-04-young.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-05-5-o-clock-shadow.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-06-arched-eyebrows.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-07-bags-under-eyes.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-08-bald.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-09-bangs.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-10-big-lips.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-11-big-nose.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-12-black-hair.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-13-blond-hair.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-14-blurry.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-15-brown-hair.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-16-bushy-eyebrows.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-17-chubby.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-18-double-chin.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-19-eyeglasses.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-20-goatee.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-21-gray-hair.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-22-heavy-makeup.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-23-high-cheekbones.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-24-mouth-slightly-open.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-25-mustache.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-26-narrow-eyes.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-27-no-beard.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-28-oval-face.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-29-pale-skin.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-30-pointy-nose.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-31-receding-hairline.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-32-rosy-cheeks.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-33-sideburns.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-34-straight-hair.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-35-wearing-earrings.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-36-wearing-hat.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-37-wearing-lipstick.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-38-wearing-necklace.pkl',
    'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/celebahq-classifier-39-wearing-necktie.pkl',
]

#----------------------------------------------------------------------------

def prob_normalize(p):
    p = np.asarray(p).astype(np.float32)
    assert len(p.shape) == 2
    return p / np.sum(p)

def mutual_information(p):
    p = prob_normalize(p)
    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)
    result = 0.0
    for x in range(p.shape[0]):
        p_x = px[x]
        for y in range(p.shape[1]):
            p_xy = p[x][y]
            p_y = py[y]
            if p_xy > 0.0:
                result += p_xy * np.log2(p_xy / (p_x * p_y)) # get bits as output
    return result

def entropy(p):
    p = prob_normalize(p)
    result = 0.0
    for x in range(p.shape[0]):
        for y in range(p.shape[1]):
            p_xy = p[x][y]
            if p_xy > 0.0:
                result -= p_xy * np.log2(p_xy)
    return result

def conditional_entropy(p):
    # H(Y|X) where X corresponds to axis 0, Y to axis 1
    # i.e., How many bits of additional information are needed to where we are on axis 1 if we know where we are on axis 0?
    p = prob_normalize(p)
    y = np.sum(p, axis=0, keepdims=True) # marginalize to calculate H(Y)
    return max(0.0, entropy(y) - mutual_information(p)) # can slip just below 0 due to FP inaccuracies, clean those up.

#----------------------------------------------------------------------------

class LS(metric_base.MetricBase):
    def __init__(self, num_samples, num_keep, attrib_indices, minibatch_per_gpu, **kwargs):
        assert num_keep <= num_samples
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.num_keep = num_keep
        self.attrib_indices = attrib_indices
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu

        # Construct TensorFlow graph for each GPU.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()

                # Generate images.
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                dlatents = Gs_clone.components.mapping.get_output_for(latents, labels, **Gs_kwargs)
                images = Gs_clone.get_output_for(latents, None, **Gs_kwargs)

                # Downsample to 256x256. The attribute classifiers were built for 256x256.
                if images.shape[2] > 256:
                    factor = images.shape[2] // 256
                    images = tf.reshape(images, [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
                    images = tf.reduce_mean(images, axis=[3, 5])

                # Run classifier for each attribute.
                result_dict = dict(latents=latents, dlatents=dlatents[:,-1])
                for attrib_idx in self.attrib_indices:
                    classifier = misc.load_pkl(classifier_urls[attrib_idx])
                    logits = classifier.get_output_for(images, None)
                    predictions = tf.nn.softmax(tf.concat([logits, -logits], axis=1))
                    result_dict[attrib_idx] = predictions
                result_expr.append(result_dict)

        # Sampling loop.
        results = []
        for begin in range(0, self.num_samples, minibatch_size):
            self._report_progress(begin, self.num_samples)
            results += tflib.run(result_expr)
        results = {key: np.concatenate([value[key] for value in results], axis=0) for key in results[0].keys()}

        # Calculate conditional entropy for each attribute.
        conditional_entropies = defaultdict(list)
        for attrib_idx in self.attrib_indices:
            # Prune the least confident samples.
            pruned_indices = list(range(self.num_samples))
            pruned_indices = sorted(pruned_indices, key=lambda i: -np.max(results[attrib_idx][i]))
            pruned_indices = pruned_indices[:self.num_keep]

            # Fit SVM to the remaining samples.
            svm_targets = np.argmax(results[attrib_idx][pruned_indices], axis=1)
            for space in ['latents', 'dlatents']:
                svm_inputs = results[space][pruned_indices]
                try:
                    svm = sklearn.svm.LinearSVC()
                    svm.fit(svm_inputs, svm_targets)
                    svm.score(svm_inputs, svm_targets)
                    svm_outputs = svm.predict(svm_inputs)
                except:
                    svm_outputs = svm_targets # assume perfect prediction

                # Calculate conditional entropy.
                p = [[np.mean([case == (row, col) for case in zip(svm_outputs, svm_targets)]) for col in (0, 1)] for row in (0, 1)]
                conditional_entropies[space].append(conditional_entropy(p))

        # Calculate separability scores.
        scores = {key: 2**np.sum(values) for key, values in conditional_entropies.items()}
        self._report_result(scores['latents'], suffix='_z')
        self._report_result(scores['dlatents'], suffix='_w')

#----------------------------------------------------------------------------
