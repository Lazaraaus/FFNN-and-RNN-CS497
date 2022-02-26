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
    loss_func = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()
    print("\nTRAINING MODEL\n")
    #print(f"The Length of the Data Loader is: {len(data_loader)}\n")
    #print(f"The length of the vocab is: {len(data_loader.vocab)}")
    #print(f"The number of 5 token sequences is: {len(data_loader.unlabeled_seqs)}")
    #print(f"The number of labels is: {len(data_loader.labels)}")
    #pdb.set_trace()
    epoch_timer = time.time()
    count = 0
    train_loss = 0
    while count != 1:
        # COntext Tensor
        # Get Context List of Word Embeddings
        context_tensor = torch.zeros((20, 500), device=device)
        context_tensor_row = torch.zeros((5, 100), device=device)
        final_word_indices = torch.zeros(20, device=device).long()
        final_word_compare_tensor = torch.zeros(1, len(data_loader.vocab), device=device).long()
        for i, data in enumerate(data_loader): 
            # Run the forward pass
            context, final_word = data 
            for idx, word in enumerate(context):
                word_embedding = model.embeddings.weight[data_loader.vocab2index[word]]
                context_tensor_row[idx,:] = word_embedding 
            # Flatten Row and add to Tensor
            batch_idx = i % 20
            #print(f"The index is {idx}")
            context_tensor[batch_idx,:] = context_tensor_row.flatten() 
            final_word_indices[batch_idx] = data_loader.vocab2index[final_word]
            # Check if we've done 5 passes through our data_loader
            pdb.set_trace()
            batch_timer = time.time()
            if i % 20 == 0 and i != 0:
                #pdb.set_trace()
                for batch_input_count in range(20):
                    # Flatten batch_context_tensor
                    batch_context_tensor = context_tensor[batch_input_count].flatten()
                    # Build Tensors  
                    predicted_final_word = model(batch_context_tensor) # run the forward pass and get a prediction
                    predicted_final_word = torch.reshape(predicted_final_word, (1, len(data_loader.vocab)))
                    #high_prob_word_idx = torch.argmax(predicted_final_word)
                    final_word_compare_tensor[0][final_word_indices[batch_input_count]] = 1 
                    #pdb.set_trace()
                    loss = loss_func(predicted_final_word.float(), final_word_compare_tensor.float()) # calculates loss 
                    train_loss += loss.item()
                    # Print Stats
                    if i % 100000 == 0 and batch_input_count == 19:
                        print(f"Loop Iteration: {i} out of {len(data_loader.unlabeled_seqs)}\n")
                        print(f"The loss of this of this iteration  is: {loss.item()}\n")
                        print(f"The average loss so far is: {train_loss/(i/20)}")
                        print(f"The context tensor is: {context_tensor}\n")
                # Update After Batch 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Reset Context Tensor
                context_tensor = torch.zeros((20, 500), device=device)
                context_tensor_row = torch.zeros((5, 100), device=device)
                final_word_indices = torch.zeros(20, device=device).long()
                final_word_compare_tensor = torch.zeros(1, len(data_loader.vocab), device=device).long()
                print(f"The time for this batch: {time.time() - batch_timer}")
            # Return Early
            #if i == 100000:
                #return model
        print(f"\nTRAINING EPOCH-{count + 1} FINISHED\n") 
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
    loss_func = nn.CrossEntropyLoss()
    model.to(device)
    model_loss = 0
    perplexity = 0
    perplexity_2 = 0
    # Get Context List of Word Embeddings
    context_tensor = torch.zeros((20, 500), device=device)
    final_word_indices = torch.zeros(20, device=device).long()
    context_tensor_row = torch.zeros((5, 100), device=device)
    final_word_compare_tensor = torch.zeros(1, len(train_loader.vocab), device=device).long()
    for i, data in enumerate(data_loader):
        # Run the forward pass
        context, final_word = data
        for idx, word in enumerate(context):
            # Check if word out of vocab
            if word not in train_loader.vocab:
                word = "<unk>"
            # Get embedding  & add to context tensor 
            word_embedding = model.embeddings.weight[train_loader.vocab2index[word]]
            context_tensor_row[idx,:] = word_embedding
        # Check for words unknown to model
        if final_word not in train_loader.vocab:
            final_word = "<unk>" 
        # Assign to context_tensor
        batch_idx = i % 20
        flattened_tensor_row = context_tensor_row.flatten()
        context_tensor[batch_idx,:] = flattened_tensor_row
        final_word_indices[batch_idx] = train_loader.vocab2index[final_word]
        # Every 5 iterations (batch of 20) run through model
        if i % 20 == 0 and i != 0:
            for batch_input_count in range(20):
                # Flatten
                batch_context_tensor = context_tensor[batch_input_count].flatten()
                # Build Tensors
                predicted_final_word = model(batch_context_tensor) # run the forward pass and get a prediction
                #pdb.set_trace()
                predicted_final_word = torch.reshape(predicted_final_word, (1, len(train_loader.vocab)))
                final_word_idx = train_loader.vocab2index[final_word]
                final_word_compare_tensor[0][final_word_indices[batch_input_count]] = 1
                # Loop through final idxs
                # Calc loss for each 
                loss = loss_func(predicted_final_word.float(), final_word_compare_tensor.float()) # calculates loss
                model_loss += loss.item()
                perplexity += torch.exp(loss).item()
                # Calc Perplexity
                sum_model_output = torch.sum(predicted_final_word)
                sum_model_output * -1
                sum_model_output / len(predicted_final_word[0])
                perplexity_2 += math.exp(sum_model_output)
            # Reset Context Tensors
            context_tensor = torch.zeros((20, 500), device=device)
            context_tensor_row = torch.zeros((5, 100), device=device)
            final_word_indices = torch.zeros(20, device=device).long()
            final_word_compare_tensor = torch.zeros(1, len(train_loader.vocab), device=device).long()
    
    print("\nFINISHED TESTING MODEL\n")
    print(f"The total loss is: {loss}")
    seq_length = len(data_loader) / 20
    avg_loss = model_loss / seq_length
    avg_perplex = perplexity / seq_length
    avg_perplex_2 = perplexity_2 / seq_length
    tensor_avg_loss = torch.tensor(avg_loss, device=device)
    #perplexity = torch.exp(tensor_avg_loss)
    #avg_other_perplex = other_perplex / len(data_loader)
    print(f"The average loss is {avg_loss}")
    print(f"The average perplexity is {avg_perplex}")
    print(f"The average perplexity is {avg_perplex_2}")
    print(f"The perplexity is {perplexity}")
    print(f"The perplexity_2 is {perplexity_2}")

    return avg_loss, avg_perplex
    

if __name__ == "__main__":
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    train_dataloader = pickle.load(open('train_dataloader.p', 'rb'))
    test_dataloader = pickle.load(open('test_dataloader.p', 'rb'))
    train_model =  Feed_Forward(len(train_dataloader.vocab))
    print(f"initialized? {torch.cuda.is_initialized()}")
    print(f"device name: {torch.cuda.get_device_name(0)}")
    train_model = train_model.cuda(0)
    print(f"model param: {next(train_model.parameters()).device}")
    test_optimizer = optim.SGD(train_model.parameters(), lr=0.001)
    train_model.train()
    _train(train_model, train_dataloader, test_optimizer)
    train_model.eval()
    _test(train_model, test_dataloader, train_dataloader,  test_optimizer)
    print(2)
