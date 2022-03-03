import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pdb
from data.my_dataset import *
from models import *
#import cProfile
import pickle
import math
import gc
import time
def run_model(model, running_mode='train', train_set=None, valid_set=None, test_set=None,
              batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):

    """
        model: 
        running_mode:
        train_set, valid_set, test_set:
        batch_size:
        learning_rate:
        n_epochs:
        stop_thr:
        shuffle:

    This function either trains or evaluates a model.

    training mode: the model is trained and evaluated on a validation set, if provided.
                   If no validation set is provided, the training is performed for a fixed
                   number of epochs.
                   Otherwise, the model should be evaluated on the validation set
                   at the end of each epoch and the training should be stopped based on one
                   of these two conditions (whichever happens first):
                   1. The validation loss stops improving.
                   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

    Inputs: 

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset 
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
    learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model 
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model 
    loss: dictionary with keys 'train' and 'valid'
          The value of each key is a list of loss values. Each loss value is the average
          of training/validation loss over one epoch.
          If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
         The value of each key is a list of accuracies (percentage of correctly classified
         samples in the dataset). Each accuracy value is the average of training/validation
         accuracies over one epoch.
         If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set. 
    accuracy: percentage of correctly classified samples in the testing set. 

    Summary of the operations this function should perform:
    1. Use the DataLoader class to generate training, validation, or test data loaders
    2. In the training mode:
       - define an optimizer (we use SGD in this homework)
       - call the train function (see below) for a number of epochs until a stopping
         criterion is met
       - call the test function (see below) with the validation data loader at each epoch
         if the validation set is provided

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results
    """
        
    # send the model to the GPU here

    pass


def _train(model, data_loader, optimizer, device=torch.device("cuda:0")):
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
    loss_func = nn.CrossEntropyLoss().cuda(0)
    # Set Losses Numpy Array
    losses = np.empty(shape=len(data_loader), dtype='f')
    # Set Epoch Timer
    epoch_timer = time.time()
    # Set Count
    count = 0
    while count != 1:
        # Context Tensor
        # Get Context List of Word Embeddings
        model_input_tensor = torch.zeros((20, 5), device=device).long() # Tensor For Inputs
        final_word_indices = torch.zeros(20, device=device).long()      # Tensor For Idxs Target Token
        y_target = torch.zeros(1, len(data_loader.vocab), device=device).long() # Tensor to hold target probability distrubtion
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
                # If so, get ready to pass input to model
                for batch_input_count in range(20):
                    # Get Model Prediction  
                    output_logits, log_probs = model(model_input_tensor[batch_input_count])
                    output_logits = torch.reshape(output_logits, (1, len(data_loader.vocab))) # Reshape for CrossEntropyLoss
                    y_target[0][final_word_indices[batch_input_count]] = 1 
                    # Calc CrossEntropyLoss
                    loss = loss_func(output_logits.float(), y_target.float()) # calculates loss 
                    # Add to losses array
                    losses[i] = loss.item()
                    # Check if 100,000 examples have been processed
                    if i % 100000 == 0 and batch_input_count == 19:
                        # If so, print stats
                        sub_arr = np.array(losses[0:i].tolist())
                        curr_loss_avg = np.mean(sub_arr)
                        print(f"Loop Iteration: {i} out of {len(data_loader.unlabeled_seqs)}\n")
                        print(f"The loss of this of this iteration  is: {losses[i]}\n")
                        print(f"The average loss so far is: {curr_loss_avg}")
                        print(f"The time for this batch: {time.time() - batch_timer}")
                        print(f"The perplexity of this example is: {torch.exp(loss/20)}")
                        curr_loss_avg_tensor = torch.tensor(curr_loss_avg, device=device, dtype=torch.float64)
                        print(f"The perplexity of this model (so far) is: {torch.exp(curr_loss_avg_tensor)}")
                        #pdb.set_trace()

                # Update After Batch of 20
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Reset Tensors
                final_word_indices = torch.zeros(20, device=device).long()
                y_target = torch.zeros(1, len(data_loader.vocab), device=device).long() 
                model_input_tensor = torch.zeros((20, 5), device=device).long()

            # Return Early
            #if i == 500000:
                #return model
        print(f"\nTRAINING EPOCH-{count + 1} FINISHED\n") 
        # Incr Epoch Count
        count += 1
    print("\nTRAINING FINISHED\n")
    print(f"The time for this epoch: {time.time() - epoch_timer}")
    return model

def _test(model, data_loader, train_loader, optimizer, device=torch.device("cuda:0")):
    """
    This function will evaluate a trained neural network on a validation
    set or testing set

    Returns accuracy
    """
    print("\nTESTING MODEL\n")
    loss_func = nn.CrossEntropyLoss().cuda()
    losses = np.empty(shape=len(train_loader), dtype='f')
    model_input_tensor = torch.zeros((20, 5), device=device).long()
    final_word_indices = torch.zeros(20, device=device).long()
    y_target = torch.zeros(1, len(train_loader.vocab), device=device).long()
    for i, data in enumerate(data_loader):
        # Batch Idx
        batch_idx = i % 20
        # Get context and ground truth tokens
        context, final_word = data
        for idx, word in enumerate(context):
            # Check if word out of vocab
            if word not in train_loader.vocab:
                # UNK it
                word = "<unk>"
            # Get embedding  & add to context tensor 
            #word_embedding = model.embeddings.weight[train_loader.vocab2index[word]]
            #context_tensor_row[idx,:] = word_embedding
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
                #batch_context_tensor = context_tensor[batch_input_count].flatten()
                #output_logits = model(batch_context_tensor) # run the forward pass and get a prediction
                # Get Model Prediction
                output_logits, probs = model(model_input_tensor[batch_input_count]) # run the forward pass and get a prediction
                output_logits = torch.reshape(output_logits, (1, len(train_loader.vocab)))
                final_word_idx = train_loader.vocab2index[final_word]
                y_target[0][final_word_indices[batch_input_count]] = 1
                # Calc loss for each 
                loss = loss_func(output_logits.float(), y_target.float()) # calculates loss
                losses[i] = loss.item()
            # Reset Context Tensors
            #context_tensor = torch.zeros((20, 500), device=device)
            #context_tensor_row = torch.zeros((5, 100), device=device)
            final_word_indices = torch.zeros(20, device=device).long()
            y_target = torch.zeros(1, len(train_loader.vocab), device=device).long()
            model_input_tensor = torch.zeros((20, 5), device=device).long()
    
    print("\nFINISHED TESTING MODEL\n")
    avg_loss = np.mean(losses)
    avg_loss_tensor = torch.tensor(avg_loss, device=device, dtype=torch.float64)
    perplexity = torch.exp(avg_loss_tensor)
    print(f"The perplexity is {perplexity}")

    return avg_loss, perplexity
    

if __name__ == "__main__":
    # PyTorch Inits
    torch.cuda.init() # Cuda
    gc.collect()      # Garbage Collection
    torch.cuda.empty_cache() # Empty Mem Cache
    # Load data_loaders from pickle files
    train_dataloader = pickle.load(open('train_dataloader.p', 'rb'))
    test_dataloader = pickle.load(open('test_dataloader.p', 'rb'))
    train_model =  Feed_Forward(len(train_dataloader.vocab))
    # Check for cuda initialization and cuda device
    print(f"initialized? {torch.cuda.is_initialized()}")
    print(f"device name: {torch.cuda.get_device_name(0)}")
    train_model = train_model.cuda(0) # Move model to gpu
    print(f"model param: {next(train_model.parameters()).device}")
    # optimizer for model
    test_optimizer = optim.SGD(train_model.parameters(), lr=0.0001)
    train_model.train() # Set model to training model
    _train(train_model, train_dataloader, test_optimizer) # train
    train_model.eval() # Set model to eval mode
    _test(train_model, test_dataloader, train_dataloader,  test_optimizer) # Test
