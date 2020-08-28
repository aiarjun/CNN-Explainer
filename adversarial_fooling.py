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


def make_fooling_image(X, target_y, model):
  X_fooling = X.clone()
  X_fooling = X_fooling.requires_grad_()
  learning_rate = 1
  iter = 0
  print_every = 10

  while True:
    scores = model(X_fooling)
    _, idx = torch.max(scores, 1)
    if (idx != target_y):
      scores[:,target_y].backward()
      dX = learning_rate*X_fooling.grad.data/torch.norm(X_fooling.grad.data)
      X_fooling.data += dX.data
      X_fooling.grad.data.zero_()
      if iter % print_every == 0:         
        print('Iteration %d, target indices\' scores: ' % (iter), scores[:,target_y].data)
      iter += 1
    else:
      break 
  return X_fooling

def get_image_fooling_grids(X, y, idx2label, idx, X_fooling, target_y):
  X_fooling_np = deprocess(X_fooling.clone())
  X_fooling_np = np.asarray(X_fooling_np).astype(np.uint8)
  fig = plt.figure()
  ax = fig.add_subplot(1, 4, 1)
  ax.imshow(X[idx])
  ax.set_title(idx2label[y[idx]])
  ax.axis('off')

  ax = fig.add_subplot(1, 4, 2)
  ax.imshow(X_fooling_np)
  ax.set_title(idx2label[target_y])
  ax.axis('off')

  fig.add_subplot(1, 4, 3)
  X_pre = preprocess(Image.fromarray(X[idx]))
  diff = np.asarray(deprocess(X_fooling - X_pre, should_rescale=False))
  ax.imshow(diff)
  ax.set_title('Difference')
  ax.axis('off')

  fig.add_subplot(1, 4, 4)
  diff = np.asarray(deprocess(10 * (X_fooling - X_pre), should_rescale=False))
  ax.imshow(diff)
  ax.set_title('Magnified difference (10x)')
  ax.axis('off')

  return fig

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Saliency')
  parser.add_argument('--num_images', type = int, help='Number of images', required=True)
  parser.add_argument('--image_index', type=int, help='Index of image whose gradient is to ascended', required=True)
  parser.add_argument('--class_index', type = int, help='Class index of class whose score is to be maximized', default = random.randint(0,999))
  parser.add_argument('--output_dir', help='Directory to save output', default = 'fooling_outputs')

  args = parser.parse_args()
  X, y, idx2label = load_imagenet_val(count=args.num_images)
  model = get_pretrained_squeezenet()
  X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
  idx = args.image_index
  target_y = args.class_index
  X_fooling = make_fooling_image(X_tensor[idx:idx+1], target_y, model)

  grid = get_image_fooling_grids(X, y, idx2label, saliency)

  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

  random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
  grid.savefig(os.path.join(args.output_dir, 'Fooling_%s_%s_%s.png' % (idx2label[y[idx]], idx2label[target_y], random_str)))
