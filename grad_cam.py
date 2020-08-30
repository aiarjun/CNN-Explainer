import os
import argparse
from PIL import Image, ImageFilter
from utils import *
import random
import numpy as np
import torch
import copy
import matplotlib.cm as mpl_color_map

def apply_colormap_on_image(org_im, activation, colormap_name = 'hsv'):
  # get the direct heatmap(only CAM, no direct image)
  color_map = mpl_color_map.get_cmap(colormap_name)
  direct_heatmap = color_map(activation)

  # set alpha of copied heatmap to 0.4(increasing will make the orig img less visible)
  heatmap = copy.copy(direct_heatmap)
  heatmap[:, :, 3] = 0.4 # vary and see


  # convert to PIL Images 0-255
  direct_heatmap = Image.fromarray((direct_heatmap*255).astype(np.uint8))
  heatmap = Image.fromarray((heatmap*255).astype(np.uint8))

  # Apply heatmap on the image by alpha compositing heatmap and original image(alpha/opacity level of heatmap is 0.4, and image is 1)
  heatmap_on_image = Image.new("RGBA", org_im.size)
  heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
  heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
  return direct_heatmap, heatmap_on_image

class CAMExtractor():
  '''Performs a forward pass of the model and also hooks the required layer to store gradients in the backward pass'''
  def __init__(self, model, target_layer):
    self.model = model
    self.target_layer = target_layer
    self.gradients = None
  def save_gradients(self, gradients):
    self.gradients = gradients
  def features_forward_pass(self, x):
    CAM_feature_map = None
    for idx, module in self.model.features._modules.items():
      x = module(x)
      if int(idx) == self.target_layer:
        x.register_hook(self.save_gradients)
        CAM_feature_map = x
    return x, CAM_feature_map
  def full_forward_pass(self, x):
    x, CAM_feature_map = self.features_forward_pass(x)
    x = self.model.classifier(x).reshape(x.shape[0], -1)
    return x, CAM_feature_map

class GradCAM():
  def __init__(self, model, target_layer):
    self.model = model
    self.model.eval()
    self.extractor = CAMExtractor(self.model, target_layer)

  def generate_cam(self, input_image, target_class = None):
    # perform forward pass 
    softmax_outputs, CAM_feature_map = self.extractor.full_forward_pass(input_image)
    if not target_class:
      target_class = np.argmax(softmax_outputs.data.numpy())
    # convert probabilities to one-hot output
    one_hot_softmax_outputs = torch.FloatTensor(1, softmax_outputs.size()[-1]).zero_() 
    one_hot_softmax_outputs[0][target_class] = 1
    # perform backward pass
    self.model.features.zero_grad()
    self.model.classifier.zero_grad()
    softmax_outputs.backward(gradient = one_hot_softmax_outputs)
    # get required layer's activation map and its gradients
    target_layer_activation_map = CAM_feature_map.data.numpy()[0]
    target_layer_gradients = self.extractor.gradients.data.numpy()[0]
    # get mean over each channel of gradients as weights
    channel_wise_weights = target_layer_gradients.mean(axis = (1,2))
    # multiply and sum the weighted activation maps
    cam = np.ones(target_layer_activation_map.shape[1:], dtype=np.float32)
    for channel, weight in enumerate(channel_wise_weights):
      cam += weight * target_layer_activation_map[channel]
    cam = np.maximum(cam, 0) # like ReLU
    # normalize and bilinearly interpolate to image size
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) # normalize to 0-1
    cam = np.uint8(cam * 255)
    cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2], input_image.shape[3]), Image.ANTIALIAS))/255
    return cam

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'GradCAM')
  parser.add_argument('--target_layer', type = int, help='Target layer', default = 12)
  parser.add_argument('--output_dir', help='Directory to save output', default = 'outputs')
  args = parser.parse_args()

  X, y, idx2label = load_imagenet_val()
  model = get_pretrained_squeezenet()
  X_tensor = torch.autograd.Variable(torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0), requires_grad = True)
  y_tensor = torch.LongTensor(y)
  idx = random.choice(range(y.shape[0]))

  gradCAM = GradCAM(model, args.target_layer)

  cam = gradCAM.generate_cam(X_tensor[idx:idx+1], y[idx])

  direct_heatmap, heatmap_on_image = apply_colormap_on_image(deprocess(X_tensor[idx:idx+1]), cam)
  
  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
  
  direct_heatmap.save(os.path.join(args.output_dir, 'GradCAM_heatmap_%s.png' % (idx2label[y[idx]])))
  heatmap_on_image.save(os.path.join(args.output_dir, 'GradCAM_heatmap_on_image_%s.png' % (idx2label[y[idx]])))
  Image.fromarray((cam * 255).astype(np.uint8)).save(os.path.join(args.output_dir, 'GradCAM_CAM_%s.png' % (idx2label[y[idx]])))

