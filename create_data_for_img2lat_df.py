"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
#sys.path.append("../stylegan")
import model.stylegan.dnnlib as dnnlib
from model.stylegan.dnnlib import tflib
#import config
import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')


def GAN_output_to_RGB(img, resize=True):

    # [-1,1] => [0,255]
    img = np.clip(np.rint((img + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
    if resize:
        img = img.transpose(0, 2, 3, 1)  # NCHW => NHWC

    #else:
    #    img = img.transpose(1,2,0)

    return img


def RGB_to_GAN_output(img, resize=True, batch_size=1):

    img = np.array(img)
    if resize and batch_size==1:
        img = img.transpose(2, 0, 1)

    if resize and batch_size>1:
        img = img.transpose(0, 3, 1, 2)

    img = img.astype(float)
    img = 2 * (img / 255.0) - 1

    if not resize:
        return img

    return np.tile(img, (batch_size, 1, 1, 1))

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
#synthesis_kwargs = dict(output_transform=dict(func=GAN_output_to_RGB, resize=True), minibatch_size=8)

class GeneratorInverse:
    def __init__(self, sess=None):
        # Initialize TensorFlow.
        #tflib.init_tf()

        if sess is None:
            config = tf.ConfigProto(allow_soft_placement=False)
            config.gpu_options.allow_growth = True
            sess = tf.InteractiveSession(config=config)

        # Load pre-trained network.
        #url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
        #with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        filename = '../stylegan/models/karras2019stylegan-ffhq-1024x1024.pkl'
        infile = open(filename,'rb')
        _G, _D, self.Gs = pickle.load(infile)
        infile.close()
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
        
        self.rnd = np.random.RandomState(1898)

        # Print network details.
        self.Gs.print_layers()
        logging.debug("inverter initialized")

    def make_image(self, iter_num=0):
        # Initialize TensorFlow.
        i=iter_num

        # Pick latent vector.
        #rnd = np.random.RandomState(5)
        #rnd = np.random.RandomState(1898)
        latents = self.rnd.randn(1, self.Gs.input_shape[1])
        #latents = np.random.randn(1, self.Gs.input_shape[1])

        #print('Latent Vector: \n', latents)

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        #fmt = dict(func=GAN_output_to_RGB, resize=True)
        #images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images = self.Gs.run(latents, None, randomize_noise=False, output_transform=fmt, dlatent_avg_beta=None, style_mixing_prob=None, truncation_psi= None,truncation_cutoff= None)

        #src_dlatents = Gs.components.mapping.run(latents, None) # [seed, layer, component]
        #images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)

        #img_name = 'img2vec/img_iters/g_out.%06d.png' % (i)
        #npy_name = 'results/img2vec/npy/g_out.%06d.npy' % (i)
        img_name = 'img/g_out.%06d.png' % (i)
        npy_name = 'results_test/npy/g_out.%06d.npy' % (i)
        logging.debug(f"img_name: {img_name}")

        os.makedirs('results_test', exist_ok=True)
        os.makedirs('results_test/img', exist_ok=True)
        os.makedirs('results_test/npy', exist_ok=True)

        np.save(npy_name, latents)

        # Save image.
        #os.makedirs('results_test', exist_ok=True)
        png_filename = os.path.join('results_test', img_name)
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

    def return_image_from_vec(self, latents):

        logging.debug(f"Latent Vector: {latents}\n")

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        #fmt = dict(func=GAN_output_to_RGB, resize=True)
        #images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images = self.Gs.run(latents, None, randomize_noise=False, output_transform=fmt, dlatent_avg_beta=None, style_mixing_prob=None, truncation_psi= None,truncation_cutoff= None)

        #src_dlatents = Gs.components.mapping.run(latents, None) # [seed, layer, component]
        #images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)

        #img_name = 'img2vec/img_iters/g_out.%06d.png' % (i)
        #npy_name = 'results/img2vec/npy/g_out.%06d.npy' % (i)
        #logging.debug(f"img_name: {img_name}")

        #np.save(npy_name, latents)

        # return image
        #os.makedirs(config.result_dir, exist_ok=True)
        #png_filename = os.path.join(config.result_dir, img_name)
        #PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        return images[0]

if __name__ == "__main__":
    #main()
    gx = GeneratorInverse()
    #max_iter = 10000
    max_iter = 10
    for i in range(0, max_iter):
        gx.make_image(iter_num=i)
