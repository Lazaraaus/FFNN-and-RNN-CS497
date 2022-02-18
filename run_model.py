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
    pdb.set_trace()
    for i, data in enumerate(data_loader):
        print(f"Loop Iteration: {i} out of {len(data_loader.unlabeled_seqs)}\n")
        unk_1 = ''
        unk_2 = ''
        #pdb.set_trace()
        # Run the forward pass
        context, final_word = data
        #print(f"The context is: {context}\nThe final_word is: {final_word}\n")
        # Get Context List of Word Embeddings
        context_tensor = torch.zeros((5, 100), device=device)
        for idx, word in enumerate(context):
            #print(f"Index of current word is: {data_loader.vocab2index[word]}\n")
            #print(f"The embedding of the current word is: {model.embeddings.weight[data_loader.vocab2index[word]]}")
            word_embedding = model.embeddings.weight[data_loader.vocab2index[word]]
            context_tensor[idx] = word_embedding
            
        # Flatten
        context_tensor = context_tensor.flatten() 
        # Get Final Word Word Embedding
        final_word_embedding = model.embeddings.weight[data_loader.vocab2index[final_word]]
        #print(f"The context_list is: {context_tensor}\nThe final_word embedding is: {final_word_embedding}\n")
        # Build Tensors  
        tensor_final_word = torch.tensor(final_word_embedding, device=device) 
        #print(f"The final word casted to tensor is: {tensor_final_word}")
        #print(f"Type of tensor_final_word: {tensor_final_word.dtype}")
        predicted_final_word = model(context_tensor) # run the forward pass and get a prediction
        #pdb.set_trace()
        predicted_final_word = torch.reshape(predicted_final_word, (1, len(data_loader.vocab)))
        high_prob_word_idx = torch.argmax(predicted_final_word)
        final_word_idx = data_loader.vocab2index[final_word]
        #print(f"The attributes of the embeddings are: {dir(model.embeddings)}")
        #pdb.set_trace()
        loss = loss_func(predicted_final_word, torch.tensor([final_word_idx], device=device)) # calculates loss between prediction and label
        if i % 4 == 0: # zero out gradients every batche of size 20
            optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("\nTRAINING MODEL FINISHED\n") 
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

    avg_loss = 0
    avg_acc = 0

    for i, data in enumerate(data_loader):
        # Run the forward pass
        context, final_word = data
        #print(f"The context is: {context}\nThe final_word is: {final_word}\n")
        #pdb.set_trace()
        # Get Context List of Word Embeddings
        context_tensor = torch.zeros((5, 100), device=device)
        for idx, word in enumerate(context):
            # Check if word out of vocab
            if word not in train_loader.vocab:
                word = "<unk>"
            # Get embedding  & add to context tensor 
            word_embedding = model.embeddings.weight[train_loader.vocab2index[word]]
            context_tensor[idx]
        # Check for words unknown to model
        if final_word not in train_loader.vocab:
            final_word = "<unk>"
        # Flatten
        context_tensor = context_tensor.flatten() 
        # Get Final Word Word Embedding
        final_word_embedding = model.embeddings.weight[train_loader.vocab2index[final_word]]
        # Build Tensors  
        tensor_final_word = torch.tensor(final_word_embedding, device=device) 
        predicted_final_word = model(context_tensor) # run the forward pass and get a prediction
        #pdb.set_trace()
        predicted_final_word = torch.reshape(predicted_final_word, (1, len(train_loader.vocab)))
        high_prob_word_idx = torch.argmax(predicted_final_word)
        final_word_idx = train_loader.vocab2index[final_word]
        #pdb.set_trace()
        loss = loss_func(predicted_final_word, torch.tensor([final_word_idx], device=device)) # calculates loss between prediction and label
        avg_loss += loss.item()
        avg_acc += 1 if (high_prob_word_idx == final_word_idx) else 0 
    
    print("\nFINISHED TESTING MODEL\n")
    print(avg_loss / len(data_loader))
    print(avg_acc / len(data_loader) * 100)
    return (avg_loss / len(data_loader)), (avg_acc / len(data_loader)) * 100
    

if __name__ == "__main__":
    torch.cuda.init()
    train_dataloader = MyDataset("train")
    test_dataloader = MyDataset("test") 
    train_model =  Feed_Forward(len(train_dataloader.vocab))
    print(f"initialized? {torch.cuda.is_initialized()}")
    print(f"device name: {torch.cuda.get_device_name(0)}")
    train_model.cuda(0)
    print(f"model param: {next(train_model.parameters()).device}")
    test_optimizer = optim.SGD(train_model.parameters(), lr=0.01)
    _train(train_model, train_dataloader, test_optimizer)
    _test(train_model, test_dataloader, train_dataloader,  test_optimizer)
    print(2)
