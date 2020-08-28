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


def compute_saliency(model, X, y):
  model.eval()
  X.requires_grad_()

  scores = model(X)
  correct_label_scores = (scores.gather(1, y.view(-1, 1)).squeeze())
  correct_label_scores.backward(torch.FloatTensor([1.0]*correct_label_scores.shape[0]))

  saliency, _ = torch.max(X.grad.data.abs(), axis = 1)
  return saliency

def get_image_saliency_grids(X, y, idx2label, saliency):
  saliency = saliency.numpy()
  N = X.shape[0]
  figures = []
  for i in range(N):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(X[i])
    ax.axis('off')
    ax.set_title(idx2label[y[i]])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(saliency[i], cmap=plt.cm.hot)
    ax.axis('off')
    figures.append(fig)
  return figures 

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Saliency')
  parser.add_argument('--num_images', type = int, help='Number of images', required=True)
  parser.add_argument('--output_dir', help='Directory to save output', default = 'saliency_outputs')

  args = parser.parse_args()
  X, y, idx2label = load_imagenet_val(count=args.num_images)
  model = get_pretrained_squeezenet()
  X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
  y_tensor = torch.LongTensor(y)
  saliency = compute_saliency(model, X_tensor, y_tensor)
  grids = get_image_saliency_grids(X, y, idx2label, saliency)

  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

  for i, grid in enumerate(grids):
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    grid.savefig(os.path.join(args.output_dir, 'Saliency_%s_%s.png' % (idx2label[y[i]], random_str)))
