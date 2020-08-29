import argparse
import torch
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import string

from utils import *


def jitter(X, ox, oy):    
  if ox != 0:
    left = X[:, :, :, :-ox]
    right = X[:, :, :, -ox:]
    X = torch.cat([right, left], dim=3)
  if oy != 0:
    top = X[:, :, :-oy]
    bottom = X[:, :, -oy:]
    X = torch.cat([bottom, top], dim=2)
  return X

def create_class_visualization(target_y, model, idx2label, dtype, **kwargs):
  model.type(dtype)
  l2_reg = kwargs.pop('l2_reg', 1e-3)
  learning_rate = kwargs.pop('learning_rate', 25)
  num_iterations = kwargs.pop('num_iterations', 500)
  blur_every = kwargs.pop('blur_every', 10)
  max_jitter = kwargs.pop('max_jitter', 16)
  show_every = kwargs.pop('show_every', 100)

  img = torch.randn(1, 3, 224, 224).mul_(1.0).type(dtype).requires_grad_()

  for t in range(num_iterations):
    ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
    img.data.copy_(jitter(img.data, ox, oy))

    scores = model(img)
    score = scores[:,target_y] - (l2_reg * torch.norm(img))
    score.backward()
    img.data += (learning_rate*img.grad.data/torch.norm(img.grad.data))
    img.grad.data.zero_()
    
    img.data.copy_(jitter(img.data, -ox, -oy))

    for c in range(3):
      lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
      hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
      img.data[:, c].clamp_(min=lo, max=hi)
    if t % blur_every == 0:
      blur_image(img.data, sigma=0.5)
    
    if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
      plt.imshow(deprocess(img.data.clone().cpu()))
      class_name = idx2label[target_y]
      plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
      plt.gcf().set_size_inches(4, 4)
      plt.axis('off')
      plt.show()

  return deprocess(img.data.cpu())

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Class visualization')
  parser.add_argument('--class_index', type = int, help='Class index of class to be visualized', default = random.randint(0,999))
  parser.add_argument('--output_dir', help='Directory to save output', default = 'saliency_outputs')

  args = parser.parse_args()
  X, y, idx2label = load_imagenet_val() 
  model = get_pretrained_squeezenet()
  dtype = torch.FloatTensor
  model.type(dtype)
  target_y = args.class_index
  out = create_class_visualization(target_y, model, idx2label, dtype)

  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

  random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
  out.save(os.path.join(args.output_dir, 'Class_viz_%s_%s.png' % (idx2label[target_y], random_str)))
