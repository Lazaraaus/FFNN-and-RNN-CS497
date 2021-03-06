import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pdb
from data.my_dataset import *
from models import *
   
def _train(model, data_loader, valid_loader, optimizer, device=torch.device("cuda:0")):
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
    print("\nTRAINING MODEL\n")
    # Set Loss Func
    loss_func = nn.CrossEntropyLoss(reduction='sum').cuda(0)# CHANGE: Put the loss func onto the GPU
    # Set Losses Numpy Array
    losses = torch.zeros(len(data_loader), dtype=torch.float64, device=device)#np.empty(shape=len(data_loader), dtype='f') # CHANGE: Numpy array to hold losses
    large_losses = 0
    training_perp = torch.zeros(20, dtype=torch.float64, device=device)
    valid_perp = torch.zeros(20, dtype=torch.float64, device=device)
    # Set Epoch Timer
    epoch_timer = time.time()
    # Set Count
    count = 0
    while count != 20:
        # Context Tensor
        # Get Context List of Word Embeddings
        model_input_tensor = torch.zeros((20, 5), device=device).long() # Tensor For Inputs
        final_word_indices = torch.zeros(20, device=device).long()      # Tensor For Idxs Target Token
        y_target = torch.zeros(1, len(data_loader.vocab), device=device).long() # Tensor to hold target probability distrubtion'

        for i, data in enumerate(data_loader):
            # Batch Idx
            batch_idx = i % 20
            # Get context words, ground truth
            context, final_word = data

            # Loop through context and build input tensor
            for idx, word in enumerate(context):
                # Lookup index for token and add to input tensor
                model_input_tensor[batch_idx][idx] = data_loader.vocab2index[word]
            # Lookup index for ground truth
            final_word_indices[batch_idx] = data_loader.vocab2index[final_word]
            # Timer for batch
            batch_timer = time.time()

            # Check if we've met our batch size
            if i % 20 == 0 and i != 0:
                # Zero Grads
                # If so, get ready to pass input to model
                for batch_input_count in range(20):
                    # Get Model Prediction
                    output_logits, log_probs = model(model_input_tensor[batch_input_count])
                    #log_probs = torch.reshape(output_logits, (1, len(data_loader.vocab))) # Reshape for CrossEntropyLoss
                    y_target[0][final_word_indices[batch_input_count]] = 1
                    # Calc CrossEntropyLoss
                    loss = loss_func(output_logits.float(), y_target.float()) # calculates loss
                    # Add to losses array
                    losses[i] = loss.item()

                # Update After Batch of 20
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Reset Tensors
                final_word_indices = torch.zeros(20, device=device).long()
                y_target = torch.zeros(1, len(data_loader.vocab), device=device).long()
                model_input_tensor = torch.zeros((20, 5), device=device).long()

            if i == len(data_loader) - 1:
                training_perp[count] = torch.exp(torch.mean(losses[0:i]))
                valid_perp[count] = _test(model, valid_loader, data_loader, optimizer)
                
        print(f"\nTRAINING EPOCH-{count + 1} FINISHED")
        print(f"The time for this epoch: {time.time() - epoch_timer}\n")
        # Incr Epoch Count
        count += 1

    # Make Plots - Train
    plt.figure(figsize=(10, 10))
    plt.plot(training_perp.squeeze().tolist())
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Training Perplexities")
    plt.savefig("training.png")

    # Make Plots - Valid    
    plt.figure(figsize=(10, 10))
    plt.plot(valid_perp.squeeze().tolist())
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Validation Perplexities")
    plt.savefig("valid.png")
    
    # Print 
    print("Final Training Perplexity", training_perp[-1])
    print("Final Validation Perplexity", valid_perp[-1])
    
    # Return
    print("\nTRAINING FINISHED\n")
    return model

def _test(model, data_loader, train_loader, optimizer, device=torch.device("cuda:0")):
    """
    This function will evaluate a trained neural network on a validation
    set or testing set
    Returns accuracy
    """
    print("\nTESTING MODEL\n")
    
    # Init
    loss_func = nn.CrossEntropyLoss().cuda(0)
    losses = torch.zeros(len(data_loader), dtype=torch.float64, device=device)
    model_input_tensor = torch.zeros((20, 5), device=device).long()
    final_word_indices = torch.zeros(20, device=device).long()
    y_target = torch.zeros(1, len(train_loader.vocab), device=device).long()
    
    # Loop through Test Data
    for i, data in enumerate(data_loader):
        # Batch Idx
        batch_idx = i % 20
        # Get context and ground truth tokens
        context, final_word = data

        # Loop through context
        for idx, word in enumerate(context):
            # Check if word out of vocab
            if word not in train_loader.vocab:
                # UNK it
                word = "<unk>"
            model_input_tensor[batch_idx][idx] = train_loader.vocab2index[word]

        # Check for words unknown to model
        if final_word not in train_loader.vocab:
            # If exist, UNK them
            final_word = "<unk>"
        # Assign to context_tensor
        final_word_indices[batch_idx] = train_loader.vocab2index[final_word]

        # Check if we have a 20 batch
        if i % 20 == 0 and i != 0:
            for batch_input_count in range(20):
                # Get Model Prediction
                output_logits, log_probs = model(model_input_tensor[batch_input_count]) # run the forward pass and get a prediction
                output_logits = torch.reshape(output_logits, (1, len(train_loader.vocab)))
                final_word_idx = train_loader.vocab2index[final_word]
                y_target[0][final_word_indices[batch_input_count]] = 1

                # Calc loss for each
                loss = loss_func(output_logits.float(), y_target.float()) # calculates loss
                losses[i] = loss.item()

            # Reset Context Tensors
            final_word_indices = torch.zeros(20, device=device).long()
            y_target = torch.zeros(1, len(train_loader.vocab), device=device).long()
            model_input_tensor = torch.zeros((20, 5), device=device).long()
    
    # Print Finished Msg
    print("\nFINISHED TESTING MODEL\n")
    avg_loss = torch.mean(losses)
    perplexity = torch.exp(avg_loss)
    print(f"The final perplexity is {perplexity}")
    # Return
    return perplexity


if __name__ == "__main__":
    # PyTorch Inits
    torch.cuda.init() # Cuda
    torch.cuda.empty_cache() # Empty Mem Cache

    # Load data_loaders from pickle files
    train_dataloader = pickle.load(open('train_dataloader.p', 'rb'))
    test_dataloader = pickle.load(open('test_dataloader.p', 'rb'))
    valid_dataloader = pickle.load(open('valid_dataloader.p', 'rb'))
    
    # Get Model
    train_model =  Feed_Forward(len(train_dataloader.vocab))

    # Check for cuda initialization and cuda device
    print(f"initialized? {torch.cuda.is_initialized()}")
    print(f"device name: {torch.cuda.get_device_name(0)}")
     
    # Move model to gpu 
    train_model = train_model.cuda(0) 
    # optimizer for model
    test_optimizer = optim.SGD(train_model.parameters(), lr=0.0001)

    # Train & Test 
    train_model.train() # Set model to training model 
    _train(train_model, train_dataloader, valid_dataloader, test_optimizer) # train

    train_model.eval() # Set model to eval mode 
    _test(train_model, test_dataloader, train_dataloader,  test_optimizer) # Test

 
