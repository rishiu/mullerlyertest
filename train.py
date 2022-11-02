import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from models import SimpleMullerLyerModel, ResnetMullerLyerModel, MullerLyerDataset

def train(model_type, epochs, train_dir, val_dir, test_dir, lr=5e-4, checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = model_type().to(device).float()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()

    train_dataset = MullerLyerDataset(train_dir)
    val_dataset = MullerLyerDataset(val_dir)
    test_dataset = MullerLyerDataset(test_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    os.makedirs("./chkpts/", exist_ok=True)

    for epoch in range(epochs):
        avg_loss = 0.0
        batch_count = 0
        for batch, (X, y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            output = model(X.float())
            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                print("Batch: "+str(batch)+" Loss: " + str(float(loss)))
            avg_loss += float(loss)
            batch_count += 1
        avg_loss /= batch_count
        print("Epoch: "+str(epoch)+" Loss: " + str(avg_loss))
    
        with torch.no_grad():
            avg_loss = 0.0
            correct = 0
            total_long = 0
            total_short = 0
            long_correct = 0
            short_correct = 0
            for vbatch, (X, y) in enumerate(val_dataloader):
                X = X.to(device)
                output = model(X.float()).cpu()
                loss = loss_fn(output, y)
                long_idx = (y.argmax(1) == 0)
                short_idx = (y.argmax(1) == 1)
                total_long += long_idx.count_nonzero().item()
                total_short += short_idx.count_nonzero().item()
                long_correct += (output[long_idx].argmax(1) == y[long_idx].argmax(1)).type(torch.float).sum().item()
                short_correct += (output[short_idx].argmax(1) == y[short_idx].argmax(1)).type(torch.float).sum().item()
                print(long_correct)
                print(short_correct)
                print(total_long)
                print(total_short)
                correct += long_correct + short_correct
                avg_loss += float(loss)
            avg_loss /= len(val_dataloader)
            print("Validation: ")
            print("Avg Loss: "+str(avg_loss)+" Accuracy: "+str(correct / len(val_dataset)))
            print("Long Accuracy: "+str(long_correct / total_long)+" Short Accuracy:"+str(short_correct / total_short))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "./chkpts/"+str(epoch)+".pt")

    tavg_loss = 0.0
    tcorrect = 0
    ttotal_long = 0
    ttotal_short = 0
    tlong_correct = 0
    tshort_correct = 0
    for tbatch, (X, y) in enumerate(test_dataloader):
        X = X.to(device)
        output = model(X.float()).cpu()
        loss = loss_fn(output, y)
        long_idx = (y.argmax(1) == 0)
        short_idx = (y.argmax(1) == 1)
        ttotal_long += long_idx.count_nonzero().item()
        ttotal_short += short_idx.count_nonzero().item()
        tlong_correct += (output[long_idx].argmax(1) == y[long_idx].argmax(1)).type(torch.float).sum().item()
        tshort_correct += (output[short_idx].argmax(1) == y[short_idx].argmax(1)).type(torch.float).sum().item()
        tcorrect += (output.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        tavg_loss += float(loss)
    tavg_loss /= len(test_dataloader)
    print("Test: ")
    print("Avg Loss: "+str(tavg_loss)+" Accuracy: "+str(correct / len(test_dataset)))
    print("Long Accuracy: "+str(tlong_correct / ttotal_long)+" Short Accuracy:"+str(tshort_correct / ttotal_short))

if __name__ == "__main__":
    train(SimpleMullerLyerModel, 100, "./output/cross_fin/", "./output/control/", "./output/illusion")

        
