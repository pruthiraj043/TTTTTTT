import warnings
warnings.filterwarnings('ignore')

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
import numpy as np
from torch_lr_finder import LRFinder

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch.torch.set_default_dtype(torch.float64)



def get_optimizer(model_obj,scheduler = False,scheduler_type = 'steplr',lr = 0.015):
    parameters = model_obj.parameters()

    optimizer = SGD( params = parameters,lr = lr,momentum = 0.9 )
    
    if (scheduler == True) & (scheduler_type == 'steplr'):
        scheduler = StepLR(optimizer,step_size = 3,gamma=0.1,verbose=True)
        return optimizer,scheduler

    elif (scheduler == True) & (scheduler_type == 'reducelronplateau'):
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=0.5, verbose=True, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-8)
        return optimizer,scheduler
    else:
        return optimizer,"_"

def run_lrfinder(model_obj, device, train_loader, test_loader, start_lr, end_lr, loss_type=None, lr_iter=1000):
    lrs = []
    num_iter = lr_iter

    for i in range(0,len(start_lr)):
        opti = SGD( params = model_obj.parameters(),lr = start_lr[i],momentum = 0.9,nesterov=True, weight_decay=0) 
        criterion = nn.NLLLoss()
        lr_finder = LRFinder(model_obj,opti,criterion,device = device,)
        lr_finder.range_test(train_loader ,start_lr=start_lr[i] ,end_lr=end_lr[i], num_iter=num_iter, step_mode='exp')
        
        try:
            grapg,lr_rate = lr_finder.plot()
        except:
            lr_rate = 0.001 # default
        print(f"Loss: {lr_finder.best_loss} LR :{lr_rate}")
        lr_finder.reset()
        lrs.append(lr_rate)

        opti = SGD( params = model_obj.parameters(),lr = lr_rate,momentum = 0.9,nesterov=True, weight_decay=0)
    return lrs,opti
