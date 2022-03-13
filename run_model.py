import torch
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
    count = 0
    while count != 20:
        for i, data in enumerate(data_loader):
            #print(f"Loop Iteration: {i} out of {len(data_loader.unlabeled_seqs)}\n")
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
            if i % 20 == 0: # zero out gradients every batche of size 20
                optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("\nTRAINING EPOCH-{count} FINISHED\n") 
        count += 1
    print("\nTRAINING FINISHED\n")
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
            context_tensor[idx] = word_embedding
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
    avg_acc = avg_acc / len(data_loader)
    avg_acc = avg_acc * 100
    avg_loss = avg_loss / len(data_loader)
    print(f"The average accuracy is {avg_acc}")
    print(f"The average loss is {avg_loss}")
    return avg_loss, avg_acc
    
def _train_RNN(model, data_loader, optimizer, device=torch.device("cuda:0")):
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    
    while count != 5:
        hidden_initial = torch.zeros((1, 20, 100), device=device)
        context_tensor = torch.zeros((20, 30), device=device).long()
        final_word_tensor = torch.zeros((20, len(data_loader.vocab)), device=device).long()


        for i, data in enumerate(data_loader):
            context, final_word = data
            #if i == 500000:
             #   break
            batch_index = i % 20
            for idx, word in enumerate(context):
                context_tensor[batch_index][idx] = data_loader.vocab2index[word]

            final_word_tensor[batch_index, data_loader.vocab2index[final_word]] = 1
            if i % 20 == 0 and i != 0:
                #pdb.set_trace()
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
                if i % 100000 == 0:
                    print("The current loss", losses[i])
                    current_loss_avg = torch.mean(losses[0:i])
                    perp = torch.exp(current_loss_avg)
                    print("Perplexity", perp)

        print("\nTRAINING EPOCH-{count} FINISHED\n") 
        count += 1
        scheduler.step()
    print("\nTRAINING FINISHED\n")
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
    return avg_loss, perp
 

if __name__ == "__main__":
    torch.cuda.init()
    print("CUDA version", torch.version.cuda)
    torch.autograd.set_detect_anomaly(True)
    train_dataloader = pickle.load(open("train_dataloader.p", "rb"))
    test_dataloader = pickle.load(open("test_dataloader.p", "rb"))
    #pickle.dump(train_dataloader, open("train_dataloader.p", "wb"))
    #pickle.dump(test_dataloader, open("test_dataloader.p", "wb"))
    #train_model =  Feed_Forward(len(train_dataloader.vocab))
    print("TEST Data Loader",len(test_dataloader))
    train_model_rnn = Recurrent_Neural_Network(len(train_dataloader.vocab))
    print(f"initialized? {torch.cuda.is_initialized()}")
    print(f"device name: {torch.cuda.get_device_name(0)}")
    #train_model = train_model.cuda(0)
    
    train_model_rnn = train_model_rnn.cuda(0)
    #print(f"model param: {next(train_model.parameters()).device}")
    #test_optimizer = optim.SGD(train_model.parameters(), lr=0.01)
    #_train(train_model, train_dataloader, test_optimizer)
    #_test(train_model, test_dataloader, train_dataloader,  test_optimizer)
    #print(2)
    #print(f"model param: {next(train_model_rnn.parameters()).device}")
    test_optimizer_rnn = optim.SGD(train_model_rnn.parameters(), lr=0.001)
    train_model_rnn.train()
    model, hidden_rnn = _train_RNN(train_model_rnn, train_dataloader, test_optimizer_rnn)
    model.eval()
    _test_RNN(model, test_dataloader, train_dataloader,  test_optimizer_rnn, hidden_rnn)
    
