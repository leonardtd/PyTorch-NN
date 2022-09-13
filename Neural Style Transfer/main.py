from PIL import Image
import torch
from torch import optim
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

import utils

### I/O
CONTENT_PATH = 'assets/leo2.png'
STYLE_PATH = 'assets/fractal4.jpg'
EXPORT_PATH = 'exports/nst_2.png'


device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

### Hyperparameters
EPOCHS = 4000
LR = 5e-3

ALPHA = 1
BETA = 4e5

SHOW_EVERY = 250

style_weights = {
    'conv1_1': 1,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2,
}



def neural_style_transfer():
    
    print("Loading Model")
    model = utils.build_model().to(device)
    
    ### Preprocess images
    content_p = utils.preprocess(CONTENT_PATH).to(device)
    style_p = utils.preprocess(STYLE_PATH).to(device)

    ### feature maps
    content_features = utils.get_features(content_p, model)
    style_features = utils.get_features(style_p, model)
    
    ### style grams
    style_grams = {layer: utils.gram_matrix(style_features[layer]) for layer in style_features}

    ### init target tensor
    target = content_p.clone().requires_grad_(True).to(device)
    target_features = utils.get_features(target, model)
    
    
    ### training
    optimizer = optim.Adam([target], lr=LR)
    
    print("Starting training loop")
    
    for i in range(EPOCHS):
        target_features = utils.get_features(target, model)

        c_loss = utils.content_loss(target_features['conv4_2'], content_features['conv4_2'])
        s_loss = utils.style_loss(style_weights, target_features, style_grams)
        t_loss = utils.total_loss(c_loss, s_loss, ALPHA, BETA)

        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
        
        if i% SHOW_EVERY == 0:
            print("Total Loss at epoch {} : {}".format(i, t_loss.item()))
        
        
    ### DEPROCESS
    target_copy = utils.deprocess(target.detach())
          
    ### save fig
    rescaled = (255.0 / target_copy.max() * (target_copy - target_copy.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(EXPORT_PATH)

if __name__ == "__main__":
    neural_style_transfer()