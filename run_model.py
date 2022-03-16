import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pdb
from data.my_dataset import *
from models import *


   
def _train_RNN(model, data_loader, valid_loader, optimizer, device=torch.device("cuda:0")):
    """
    This function will implement one epoch of training a given model

    Inputs:
    model: the neural network to be trained
    data_loader: for loading the network input and targets from the training dataset
    optimizer: the optimiztion method, e.g., SGD
    device: hanlon's GPU

    Outputs:
    model: the trained model
    
    """
    loss_func = nn.CrossEntropyLoss(reduction="sum").cuda(0)
    losses = torch.zeros(len(data_loader), dtype= torch.float64, device=device)

    torch.cuda.empty_cache()
    print("\nTRAINING MODEL\n")

    count = 0
    training_perp = torch.zeros(20, dtype = torch.float64, device = device)
    valid_perp = torch.zeros(20, dtype = torch.float64, device = device)

    while count != 20:
        hidden_initial = torch.zeros((1, 20, 100), device=device)
        context_tensor = torch.zeros((20, 30), device=device).long()
        final_word_tensor = torch.zeros((20, len(data_loader.vocab)), device=device).long()


        for i, data in enumerate(data_loader):

            context, final_word = data
            batch_index = i % 20

            for idx, word in enumerate(context):
                context_tensor[batch_index][idx] = data_loader.vocab2index[word]

            final_word_tensor[batch_index, data_loader.vocab2index[final_word]] = 1

            if i % 20 == 0 and i != 0:
                hidden_initial = hidden_initial.detach()
                predicted_final_word, hidden_initial = model(context_tensor, hidden_initial)

                loss = loss_func(predicted_final_word[0,:,:].float(), final_word_tensor.float()) # calculates loss between prediction and label
                losses[i] = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                hidden_initial = torch.zeros((1, 20, 100), device=device)
                context_tensor = torch.zeros((20, 30), device=device).long()
                final_word_tensor = torch.zeros((20, len(data_loader.vocab)), device=device).long()

            if i == len(data_loader) - 1:
                training_perp[count] = torch.exp(torch.mean(losses[0:i]))
                valid_perp[count] = _test_RNN(model, valid_loader, data_loader, optimizer, hidden_initial)


        print(f"\nTRAINING EPOCH-{count} FINISHED\n") 
        count += 1

    plt.figure(figsize = (10, 10))
    plt.plot(training_perp.squeeze().tolist())
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Training Perplexities")
    plt.savefig("training.png")
    print("\nTRAINING FINISHED\n")

    plt.figure(figsize = (10, 10))
    plt.plot(valid_perp.squeeze().tolist())
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Validation Perplexities")
    plt.savefig("valid.png")
    print("The Final Training Perplexity", training_perp[-1])
    print("The Final Validation Perplexity", valid_perp[-1])

    return model, hidden_initial

def _test_RNN(model, data_loader, train_loader, optimizer, hidden_final, device=torch.device("cuda:0")):
    """
    This function will evaluate a trained neural network on a validation
    set or testing set

    Returns accuracy
    """
    print("\nTESTING MODEL\n")
    loss_func = nn.CrossEntropyLoss(reduction="sum").cuda(0)
    losses = torch.zeros(len(data_loader), dtype=torch.float64, device= device)

    model.to(device)

    hidden_final = torch.zeros(hidden_final.shape, device= device)
    context_tensor = torch.zeros((20, 30), device=device).long()
    final_word_tensor = torch.zeros((20, len(train_loader.vocab)), device=device)

    for i, data in enumerate(data_loader):
        context, final_word = data
        batch_index = i % 20

        for idx, word in enumerate(context):
            if word not in train_loader.vocab:
                word = "<unk>"
            context_tensor[batch_index][idx] = train_loader.vocab2index[word]

        if final_word not in train_loader.vocab:
            final_word = "<unk>"
        final_word_tensor[batch_index][train_loader.vocab2index[final_word]] = 1

        if i % 20 == 0 and i != 0:
            hidden_final = hidden_final.detach()
            predicted_final_word, hidden_final  = model(context_tensor, hidden_final) # run the forward pass and get a prediction

            loss = loss_func(predicted_final_word[0,:,:].float(), final_word_tensor.float())# calculates loss between prediction and label
            losses[i] = loss.item()

            context_tensor = torch.zeros((20, 30), device=device).long()
            final_word_tensor = torch.zeros((20, len(train_loader.vocab)), device=device)
            hidden_final = torch.zeros(hidden_final.shape, device = device)

        
    print("\nFINISHED TESTING MODEL\n")
    avg_loss = torch.mean(losses)

    perp = torch.exp(avg_loss)
    print("Perplexity",perp)
    print(f"The average loss is {avg_loss}")
    return perp
 

if __name__ == "__main__":
    torch.cuda.init()
    print("CUDA version", torch.version.cuda)

    torch.autograd.set_detect_anomaly(True)

    train_dataloader = pickle.load(open("train_dataloader.p", "rb"))
    test_dataloader = pickle.load(open("test_dataloader.p", "rb"))
    valid_dataloader = pickle.load(open("valid_dataloader.p", "rb"))

    print("TEST Data Loader",len(test_dataloader))
    train_model_rnn = Recurrent_Neural_Network(len(train_dataloader.vocab))
    print(f"initialized? {torch.cuda.is_initialized()}")
    print(f"device name: {torch.cuda.get_device_name(0)}")
    
    train_model_rnn = train_model_rnn.cuda(0)
    test_optimizer_rnn = optim.SGD(train_model_rnn.parameters(), lr=0.001)
    train_model_rnn.train()
    model, hidden_rnn = _train_RNN(train_model_rnn, train_dataloader, valid_dataloader, test_optimizer_rnn)
    model.eval()
    _test_RNN(model, test_dataloader, train_dataloader,  test_optimizer_rnn, hidden_rnn)
    
