import os
import argparse
from PIL import Image
from utils import *
import random



import torch
from torch.nn import ReLU


class GuidedBackProp():
  def __init__(self, model):
    self.model = model
    self.gradients = None
    self.forward_relu_outputs = []
    self.model.eval()
    self.update_relus()
    self.hook_layers()

  def update_relus(self):
    '''The ReLU backward pass is modified to zero-out the incoming non-negative gradients also'''
    def relu_backward_hook_function(module, grad_in, grad_out):
      current_layer_forward_output = self.forward_relu_outputs[-1]; del self.forward_relu_outputs[-1]  
      current_layer_forward_output[current_layer_forward_output > 0] = 1
      modified_grad_out = current_layer_forward_output * torch.clamp(grad_in[0], min=0.0)
      return (modified_grad_out,)

    def relu_forward_hook_function(module, ten_in, ten_out):
      self.forward_relu_outputs.append(ten_out)

    for pos, module in self.model.features._modules.items():
      if isinstance(module, ReLU):
        module.register_backward_hook(relu_backward_hook_function)
        module.register_forward_hook(relu_forward_hook_function)

  def hook_layers(self):
    '''Storing the gradients wrt the image in self.gradients for easy access later'''
    def hook_function(module, grad_in, grad_out):
        self.gradients = grad_in[0]
    first_layer = list(self.model.features._modules.items())[0][1]
    first_layer.register_backward_hook(hook_function)


  def generate_gradients(self, input_image, target_class):
    model_output = self.model(input_image)
    self.model.zero_grad()

    one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
    one_hot_output[0][target_class] = 1

    model_output.backward(gradient=one_hot_output) 

    return np.transpose(self.gradients.data.numpy()[0], (1,2,0)) 


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description = 'Guided Backprop')
  parser.add_argument('--output_dir', help='Directory to save output', default = 'outputs')
  args = parser.parse_args()

  X, y, idx2label = load_imagenet_val()

  model = get_pretrained_squeezenet()
  X_tensor = torch.autograd.Variable(torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0), requires_grad = True)
  y_tensor = torch.LongTensor(y)
  idx = random.choice(range(y.shape[0]))
  GBP = GuidedBackProp(model)
  guided_grads = GBP.generate_gradients(X_tensor[idx:idx+1], y[idx])
  guided_grads = guided_grads - guided_grads.min()
  guided_grads /= guided_grads.max()
  guided_grads = Image.fromarray((guided_grads*255).astype(np.uint8))
  guided_grads.save(os.path.join(args.output_dir,'Guided_Backprop_%s.png' % (idx2label[idx])))