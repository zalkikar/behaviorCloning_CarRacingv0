from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import torch
from model import pyTorchModel

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # crop, 84 x 84, some pixels never change
    X_train = np.array([img[:-12,6:-6] for img in X_train])
    X_valid = np.array([img[:-12,6:-6] for img in X_valid])
    # grayscale
    X_train = np.array([np.dot(img[...,0:3], [0.299, 0.587, 0.114]) for img in X_train])
    X_valid = np.array([np.dot(img[...,0:3], [0.299, 0.587, 0.114]) for img in X_valid])
    # scaling/normalizing
    X_train = np.array([img/255.0 for img in X_train])
    X_valid = np.array([img/255.0 for img in X_valid])
    
    return X_train, y_train, X_valid, y_valid


#def train_model(X_train, y_train, X_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
def train_model(X_train,y_train,X_valid,y_valid,n_minibatches,batch_size,lr,model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")

    agent = pyTorchModel()
    optimizer = torch.optim.Adam(agent.parameters(),lr=lr) # adam optimization
    criterion = torch.nn.MSELoss() # MSE loss
    
    # form data loader for training
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
    train_loss, train_cor = 0,0
    n_iters = 0

    agent.train()
    for batch_idx, (inputs,targets) in enumerate(train_loader):
        """ permute pixels
        perm = torch.randperm(7056)
        pixel_dim = (inputs.shape[1],inputs.shape[2])
        inputs = inputs.view(-1,pixel_dim[0]*pixel_dim[1])
        inputs = inputs[:,perm]
        inputs = inputs.view(-1,pixel_dim[0],pixel_dim[1])
        """
        optimizer.zero_grad() # reset weights
        inputs = inputs.unsqueeze(1).float() # add channel for conv2d
        outputs = agent(inputs) # agent, pytorch
        loss = criterion(outputs,targets) # mse loss
        loss.backward() # backprop
        optimizer.step() # adam optim, gradient updates

        train_loss+=loss.item()
        n_iters+=1
        """ accuracy
        _, predicted = torch.max(torch.abs(outputs).detach(),1) 
        _, targetsbinary = torch.max(torch.abs(targets).detach(),1)
        n_correct = (predicted==targetsbinary).sum().item()
        train_cor+=n_correct
        """
        if batch_idx % 20 == 0:
            print(f'train batch loss for iter {batch_idx}: {loss.item()}')
    print(f'average train batch loss: {(train_loss / n_iters)}')
    #print(f'average train batch accuracy: {(train_cor / (batch_size*n_iters))}')
    
    # form data loader for validation (currently predicts on whole valid set)
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_valid),torch.from_numpy(y_valid))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=X_valid.shape[0],shuffle=False)
    valid_loss, valid_acc = 0,0
    agent.eval()
    with torch.no_grad():
        for i, (valid_inputs,valid_targets) in enumerate(valid_loader):
            valid_inputs = valid_inputs.unsqueeze(1).float()
            valid_outputs = agent(valid_inputs)
            valid_loss += criterion(valid_outputs,valid_targets).item()
            """ accuracy
            _, valid_predicted = torch.max(torch.abs(valid_outputs),1) 
            _, valid_targetsbinary = torch.max(torch.abs(valid_targets),1)
            valid_correct = (valid_predicted==valid_targetsbinary).sum().item()
            valid_acc+=(valid_correct/valid_targets.shape[0])
            """
        print(f'valid set loss: {valid_loss}')
        #print(f'valid set accuracy: {valid_acc}')
    
    torch.save(agent.state_dict(), os.path.join(model_dir,"agent_20kframes_100420.pkl"))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    #train_model(X_train, y_train, X_valid, n_minibatches=1000, batch_size=64, lr=0.0001)
    train_model(X_train,y_train,X_valid,y_valid,n_minibatches=1000,batch_size=64,lr=0.0001)
