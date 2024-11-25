import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch.nn as nn
from model import Net

from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch.torch.set_default_dtype(torch.float64)


model = Net()

def training(model,device,train_data,optimizer,epochs,scheduler = None):
    total_loss = 0
    correct = 0
    processed = 0

    model.train()
    pbar = tqdm(train_data,colour = 'YELLOW')
    
    for index_id,(data,target) in enumerate(pbar):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        correct += output.argmax(dim = 1).eq(target).sum().item()
        processed += len(data)
        pbar.set_description(f"Train ==> Epochs: {epochs} Batch:  {index_id} loss: {loss.item()} Accuracy: { correct/processed *100 :.2f}% ")
    
    if scheduler is not None:
        scheduler.step()

    acc = correct /processed
    total_loss /= processed
    return total_loss, acc

def testing(model,device,test_data,epochs):
    model.eval()
    test_loss = 0
    correct = 0
    processed = 0
    pbar= tqdm(test_data)
    with torch.no_grad():
        for id_x,(data,target) in enumerate(pbar):
            data,target = data.to(device),target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output,target,reduction='sum').item()
            pred  = output.argmax(dim =1,keepdim = True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            
            pbar.set_description(f"Test ==> Epochs: {epochs} Batch:  {id_x} loss: {test_loss/processed} Accuracy: { correct / processed *100 :.2f}% ")
        
    acc = correct / processed
    test_loss /= processed
    # test_accuracy.append(acc)
    # test_losses.append(test_loss)
   
    
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
    #     test_loss, correct, processed, 100. * correct / processed))
    
    return test_loss,acc
    