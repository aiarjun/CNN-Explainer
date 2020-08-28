import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from PIL import Image

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled
    
def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def get_pretrained_squeezenet():
  model = torchvision.models.squeezenet1_1(pretrained=True)
  for param in model.parameters():
      param.requires_grad = False
  return model

def load_imagenet_val(path = 'data/imagenet_val_25.npz', count=None):
    """
    Returns `count` images from the ImageNet validation set(present at data/) as
    - X: np array of shape [count, 224, 224, 3]
    - y: np array of shape [count] 
    - idx2label: dict that maps class indices to text labels
    """
    f = np.load(path, allow_pickle=True)
    X = f['X']
    y = f['y']
    idx2label = f['label_map'].item()
    if count is not None:
        X = X[:count]
        y = y[:count]
    return X, y, idx2label