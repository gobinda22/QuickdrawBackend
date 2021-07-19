# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:16:47 2021

@author: Gobinda
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import getDataLoader
from model import Net
import torch.optim as optim
import torch.onnx
import config
import dataloader
import os
from tqdm import tqdm

def train(model, use_cuda, train_loader, optimizer, epoch):
    
    model.train()
    
    accuracy = []
    
    iterator = tqdm(train_loader)
    
    
    for batch_idx, (data, target, path) in enumerate(iterator):#get the batch
        
            if use_cuda:
                data,target = data.cuda(), target.cuda()
                
            optimizer.zero_grad()  
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            pred = output.data.cpu().numpy().argmax(axis=1)
            accuracy.append(np.mean(np.float32(pred == target.data.cpu().numpy())))
            
            iterator.set_description('Train Epoch: {} | Loss: {:.6f} | Accuracy: {:.6f}'.format(
                
                epoch, loss.item(), np.mean(accuracy)))
        
def qudar(model, use_cuda, test_loader, epoch=0):
     
    model.eval()
    
    test_loss = 0
    correct = 0
    
    os.makedirs('check/'+str(epoch), exist_ok = True)
   
    with torch.no_grad():#gradients not calculated
       for data, target, path in test_loader:
           # Converting the target to one-hot-encoding from categorical encoding
            # Converting the data to [batch_size, 784] from [batch_size, 1, 28, 28]

          

            if use_cuda:
                data, target = data.cuda(), target.cuda()  # Sending the data to the GPU
                # data, target, y_onehot = data.cuda(), target.cuda(), y_onehot.cuda()  # Sending the data to the GPU

           
            output = model(data)  # Forward pass
            test_loss += F.cross_entropy(output, target, reduction='sum')  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the maximum output
            correct += pred.eq(target.view_as(pred)).sum().item()  # Get total number of correct samples
            
            #index = (pred.squeeze()!= target).data.cpu().numpy()
            
    test_loss /= len(test_loader.dataset)  # Accuracy = Total Correct / Total Samples

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def seed(seed_value):
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
    
def main():
    
    print('Loading the Model')
    
    model = Net() 
     
    use_cuda = False  # Set it to False if you are using a CPU

    seed(config.seed)   

    dataLoader = getDataLoader(type_='train')
    testDataLoader = getDataLoader(type_='test')



   
    if use_cuda:
        model = model.cuda()  # Put the model weights on GPU

    # criterion = nn.CrossEntropyLoss()
    # same as categorical_crossentropy loss used in Keras models which runs on Tensorflow
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # fine tuned the lr
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.4, momentum=0.9)

    for epoch in range(1, config.numEpochs + 1):
        #print(f'epoch={epoch}')
        train(model, use_cuda, dataLoader, optimizer, epoch)# Train the network
        if epoch % config.testEveryEpochs == 0:
            qudar(model, use_cuda, testDataLoader, epoch)  # Test the network
            torch.save(model.state_dict(), f"{config.savePath}/mnist_cnn_epoch_{epoch}.pt")
      
    #saveModel(model)
    dummy_input = torch.zeros(1, 1, config.imageSize[0], config.imageSize[1])
    torch.onnx.export(model, dummy_input, "savedModel/model.onnx", verbose=True, input_names=['data'], output_names=['output'])
        
    # torch.save(model.state_dict(), "cifar.pt")
    # model.load_state_dict(torch.load('cifar.pt'))
    # Loading a saved model - model.load_state_dict(torch.load('mnist_cnn.pt'))


# =============================================================================
# def saveModel(model: Net):
#     torch.save(model.state_dict(), f'{config.savePath}/model_final_epochs_{config.numEpochs}.pt')
#     # save model in onnx format
#     inp = torch.randn(1, 1, 32, 32)
#     torch.onnx.export(model, inp, f'{config.savePath}/model_final_user1.onnx', verbose=True,
#                       input_names=['data'], output_names=['output'])
# 
# =============================================================================

if __name__ == '__main__':
    main()    
    
    