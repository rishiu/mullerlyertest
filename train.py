import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from models import SimpleMullerLyerModel, ResnetMullerLyerModel, MullerLyerDataset
from symbol import test

def train(model_type, epochs, train_dir, test_dir, lr=1e-4, checkpoint=None):
    model = model_type().float()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()

    train_dataset = MullerLyerDataset(train_dir)
    test_dataset = MullerLyerDataset(test_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    os.makedirs("./chkpts/", exist_ok=True)

    for epoch in range(epochs):
        avg_loss = 0.0
        batch_count = 0
        for batch, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            output = model(X.float())
            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print("Batch: "+str(batch)+" Loss: " + str(float(loss)))
            avg_loss += float(loss)
            batch_count += 1
        avg_loss /= batch_count
        print("Epoch: "+str(epoch)+" Loss: " + str(avg_loss))
    
        with torch.no_grad():
            avg_loss = 0.0
            correct = 0
            for tbatch, (X, y) in enumerate(test_dataloader):
                output = model(X.float())
                loss = loss_fn(output, y)
                correct += (output.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            
            print("Test: ")
            print("Avg Loss: "+str(avg_loss)+" Accuracy: "+str(correct / len(test_dataset)))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "./chkpts/"+str(epoch)+".pt")

if __name__ == "__main__":
    train(SimpleMullerLyerModel, 100, "./output/cross_fin/", "./output/control/")

        
