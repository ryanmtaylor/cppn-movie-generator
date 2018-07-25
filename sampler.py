'''
Implementation of Compositional Pattern Producing Networks in Tensorflow

https://en.wikipedia.org/wiki/Compositional_pattern-producing_network

@hardmaru, 2016

Sampler Class

This file is meant to be run inside an IPython session, as it is meant
to be used interacively for experimentation.

It shouldn't be that hard to take bits of this code into a normal
command line environment though if you want to use outside of IPython.

usage:

%run -i sampler.py

sampler = Sampler(z_dim = 4, c_dim = 1, scale = 8.0, net_size = 32)

'''

import os
import numpy as np
import tensorflow as tf
import math
import random
import PIL
from PIL import Image
import pylab
from model import CPPN
import matplotlib.pyplot as plt
import images2gif
from images2gif import writeGif

import pprint as pp
import math

mgc = get_ipython().magic
mgc(u'matplotlib osx')
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

class Sampler():
  def __init__(self, z_dim = 8, c_dim = 1, scale = 10.0, net_size = 32):
    self.cppn = CPPN(z_dim = z_dim, c_dim = c_dim, scale = scale, net_size = net_size)
    self.z = self.generate_z() # saves most recent z here, in case we find a nice image and want the z-vec
  def reinit(self):
    self.cppn.reinit()
  def generate_z(self):
    z = np.random.uniform(-1.0, 1.0, size=(1, self.cppn.z_dim)).astype(np.float32)
    return z
  def generate(self, z=None, x_dim=1080, y_dim=1060, scale = 10.0):
    if z is None:
      z = self.generate_z()
    else:
      z = np.reshape(z, (1, self.cppn.z_dim))
    self.z = z
    return self.cppn.generate(z, x_dim, y_dim, scale)[0]
  def show_image(self, image_data):
    '''
    image_data is a tensor, in [height width depth]
    image_data is NOT the PIL.Image class
    '''
    plt.subplot(1, 1, 1)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = self.cppn.c_dim
    if c_dim > 1:
      plt.imshow(image_data, interpolation='nearest')
    else:
      plt.imshow(image_data.reshape(y_dim, x_dim), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.show()
  def save_png(self, image_data, filename):
    img_data = np.array(1-image_data)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = self.cppn.c_dim
    if c_dim > 1:
      img_data = np.array(img_data.reshape((y_dim, x_dim, c_dim))*255.0, dtype=np.uint8)
    else:
      img_data = np.array(img_data.reshape((y_dim, x_dim))*255.0, dtype=np.uint8)
    im = Image.fromarray(img_data)
    im.save(filename)
  def to_image(self, image_data):
    # convert to PIL.Image format from np array (0, 1)
    img_data = np.array(1-image_data)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = self.cppn.c_dim
    if c_dim > 1:
      img_data = np.array(img_data.reshape((y_dim, x_dim, c_dim))*255.0, dtype=np.uint8)
    else:
      img_data = np.array(img_data.reshape((y_dim, x_dim))*255.0, dtype=np.uint8)
    im = Image.fromarray(img_data)
    return im
  def save_anim_gif(self, z1, z2, filename, n_frame = 240, duration1 = 0.5, \
                    duration2 = 1.0, duration = 0.1, x_dim = 1920, y_dim = 1080, scale = 10.0, reverse = True):
    '''
    this saves an animated gif from two latent states z1 and z2
    n_frame: number of states in between z1 and z2 morphing effect, exclusive of z1 and z2
    duration1, duration2, control how long z1 and z2 are shown.  duration controls frame speed, in seconds
    '''
    delta_z = (z2-z1) / (n_frame+1)
    total_frames = n_frame + 2
    images = []
    for i in range(total_frames):
      z = z1 + delta_z*float(i)
      images.append(self.to_image(self.generate(z, x_dim, y_dim, scale)))
      print "processing image ", i
    durations = [duration1]+[duration]*n_frame+[duration2]
    if reverse == True: # go backwards in time back to the first state
      revImages = list(images)
      revImages.reverse()
      revImages = revImages[1:]
      images = images+revImages
      durations = durations + [duration]*n_frame + [duration1]
    print "writing gif file..."
    writeGif(filename, images, duration = durations)

  def save_anim_mp4(self, filename, n_frame = 120, x_dim = 1920, y_dim = 1080, scale = 10.0, reverse = True):
    z1 = self.generate_z()
    z2 = self.generate_z()
    path_folder = 'output/%s' % filename
    if not os.path.exists(path_folder):
      print 'creating path: %s' % path_folder
      os.makedirs(path_folder)

    delta_z = (z2-z1) / (n_frame+1)
    total_frames = n_frame + 2
    for i in range(total_frames):
      z = z1 + delta_z*float(i)
      img = self.to_image(self.generate(z, x_dim, y_dim, scale))
      img.save('%s/%s-%04d.png' % (path_folder, filename, i))
      print "processing image %d/%d" % (i, n_frame)
    os.system('ffmpeg -i %s/%s-%%04d.png -c:v libx264 -crf 0 -preset veryslow -framerate 30 %s/%s.mp4' % (path_folder, filename, path_folder, filename))
    if(reverse):
        os.system('ffmpeg -i %s/%s.mp4 -filter_complex "[0:v]reverse,fifo[r];[0:v][r] concat=n=2:v=1 [v]" -map "[v]" %s/%s-looped.mp4' % (path_folder, filename, path_folder, filename))

  def save_anim_mp4_2(self, filename, zs = [], n_frame = 360, x_dim = 1920, y_dim = 1080, scale = 10.0, count = 0):
    path_folder = 'output/%s' % filename
    if not os.path.exists(path_folder):
      print 'creating path: %s' % path_folder
      os.makedirs(path_folder)

    if(count <= 0):
        zs.append(zs[0]) # make it a full loop, return to first frame!
        first_z = zs[0]
        formatted_zs = map((lambda x: "%.3f" % x[0,0]), zs)
        print "%d vectors: %s" % (len(zs), ", ".join(formatted_zs))
        print "%d images total" % (len(zs) * n_frame)
        print "---"

    if(len(zs) <= 1):
        z = zs[0]
        img = self.to_image(self.generate(z, x_dim, y_dim, scale))
        img.save('%s/%s-%04d.png' % (path_folder, filename, count))
        print ">> %d : %.3f" % (count, z[0,0])
        print "---"
        print "%d images rendered" % count
        print "---"
        os.system('ffmpeg -i %s/%s-%%04d.png -c:v libx264 -crf 0 -preset veryslow -framerate 30 %s/%s.mp4' % (path_folder, filename, path_folder, filename))
        return

    z1 = zs.pop(0)
    z2 = zs[0]

    print ">> (%.3f to %.3f) step #%d" % (z1[0,0], z2[0,0], len(zs))

    delta_z = (z2-z1) / (n_frame + 1)
    total_frames = n_frame + 1

    for i in range(total_frames):
      z = z1 + delta_z*float(i)
      image_number = i + count;
      img = self.to_image(self.generate(z, x_dim, y_dim, scale))
      img.save('%s/%s-%04d.png' % (path_folder, filename, image_number))

      z_output = ", ".join(str(x) for x in z[0].tolist())
      print ">> %d : %.3f" % (image_number, z[0,0])
    self.save_anim_mp4_2(filename, zs, n_frame, x_dim, y_dim, scale, image_number + 1)

  def save_anim_mp4_loop(self, filename, zs, n_frame = 360, x_dim = 1920, y_dim = 1080, scale = 10.0, count = 0):
    path_folder = 'output/%s' % filename
    if not os.path.exists(path_folder):
      print 'creating path: %s' % path_folder
      os.makedirs(path_folder)

    if(isinstance(zs, (int, long))):
        zs = self.generate_zs(zs)

    if(count <= 0):
        formatted_zs = map((lambda x: "%.3f" % x[0,0]), zs)
        print "%d vectors: %s" % (len(zs), ", ".join(formatted_zs))
        print "%d images total" % (len(zs) * n_frame)
        print "---"
        zs.append(zs[0]) # make it a full loop, return to first frame!
        zs.append(zs[1]) # make it smoothly loop (knows next frame is coming)

    if(len(zs) <= 2):
        # z = zs[0]
        # img = self.to_image(self.generate(z, x_dim, y_dim, scale))
        # img.save('%s/%s-%04d.png' % (path_folder, filename, count))
        # print ">> %d : %.3f" % (count, z[0,0])
        print "---"
        print "%d images rendered" % count
        print "---"
        os.system('ffmpeg -i %s/%s-%%04d.png -c:v libx264 -crf 0 -preset veryslow -framerate 30 %s/%s.mp4' % (path_folder, filename, path_folder, filename))
        # os.system('ffmpeg -i %s/%s.mp4 -filter "minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=90" %s/%s-interpolated-90fps.mp4')
        return

    z1 = zs.pop(0)
    z2 = zs[0]
    z3 = zs[1]

    print ">> (%.3f to %.3f) with %d steps left" % (z1[0,0], z2[0,0], len(zs) - 2)

    total_frames = n_frame + 1
    for i in range(total_frames):
      percent_complete = float(i) / total_frames
      p = (math.asin(percent_complete*2 - 1) + math.pi/2)/math.pi
      delta_z1 = (z2-z1) / (n_frame + 1)
      delta_z2 = (z3-z2) / (n_frame + 1)
      delta_z = (p * delta_z2) + ((1 - p) * delta_z1)

      z = z1 + delta_z1*float(i)
      image_number = i + count;
      img = self.to_image(self.generate(z, x_dim, y_dim, scale))
      img.save('%s/%s-%04d.png' % (path_folder, filename, image_number))

      z_output = ", ".join(str(x) for x in z[0].tolist())
      print ">> #%d \tz = %.3f \t%.1f%% \t%.4f delta \t%0.4f" % (image_number, z[0,0], percent_complete * 100, delta_z[0,0], p)
    self.save_anim_mp4_loop(filename, zs, n_frame, x_dim, y_dim, scale, image_number + 1)

  def generate_zs(self, num):
    return map(lambda x: self.generate_z(), range(num))
