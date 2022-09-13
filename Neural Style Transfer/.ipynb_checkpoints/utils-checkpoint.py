import torch
from torch import optim
from torchvision import models
from PIL import Image
from torchvision import transforms as T
import numpy as np

IMG_MEAN_VALUES = [0.485, 0.456, 0.406]
IMG_STD_VALUES = [0.229, 0.224, 0.225]

### MODEL

def build_model():
    model = models.vgg19(pretrained=True).features
    
    for parameters in model.parameters():
        parameters.requires_grad_(False)
        
    return model
    
### IMAGE PROCESSING

def preprocess(img_path, max_size=500):
    image = Image.open(img_path).convert('RGB')
    
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
        
    img_transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=IMG_MEAN_VALUES,
                   std=IMG_STD_VALUES)
    ])
    
    image = img_transforms(image)
    image = image.unsqueeze(0)
    
    return image


def deprocess(tensor):
    
    image = tensor.to('cpu').clone().numpy()
    image = image.squeeze(0).transpose(1,2,0)
    image = image * np.array(IMG_STD_VALUES) + np.array(IMG_STD_VALUES)
    image = image.clip(0, 1)
    
    return image


### FEATURES

def get_features(image, model):
    
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1',
    }
    
    x = image
    
    features = dict()
    
    for name, layer in model._modules.items():
        x = layer(x)
        
        if name in layers:
            features[layers[name]] = x
        
    return features


def gram_matrix(tensor):
    b,c,h,w = tensor.size()
    tensor = tensor.view(c, h*w)
    gram = tensor @ tensor.T
    
    return gram
    
### LOSS FUNCTION

def content_loss(target_conv4_2, content_conv4_2):
    loss = torch.mean((target_conv4_2-content_conv4_2)**2)
    return loss

def style_loss(style_weights, target_features, style_grams):
    
    loss = 0
    
    for layer in style_weights:
        target_f = target_features[layer]
        target_gram = gram_matrix(target_f)
        style_gram = style_grams[layer]
        
        b,c,h,w = target_f.shape
        
        layer_loss = style_weights[layer] * torch.mean((target_gram-style_gram)**2)
        loss += layer_loss/(c*h*w) #pixel wise?
        
    return loss


def total_loss(content_loss, style_loss, alpha, beta):
    return alpha*content_loss + beta*style_loss