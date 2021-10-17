# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import os.path as osp


class Trainer():
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        epochs
    ):
        self.cptdir = 'models'

        # prepare dataloaders
        self.data_loader = data_loader

        # models and optimizer
        self.model = model
        self.optimizer = optimizer

        # Training parameters
        self.epochs = epochs

    def fit(self):
        self.model.train()
        for epoch in range(self.epochs + 1):
            for batch_idx, samples in enumerate(self.data_loader):
                x_train, y_train = samples

                y_train = y_train.reshape(len(y_train))
                prediction = self.model(x_train)
                cost = F.cross_entropy(prediction, y_train)
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()
            if epoch % 10 == 0:
                print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                    epoch, self.epochs, cost.item()
                ))
        
        self.save_model(loss=cost, epoch=self.epochs)
    
    def save_model(self, loss, epoch):
        save_dict = {
            'epoch': epoch,
            'gen_state_dict': self.model.state_dict(),
            'loss': loss,
            'optimizer': self.optimizer.state_dict(),
        }

        filename = osp.join(self.cptdir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)