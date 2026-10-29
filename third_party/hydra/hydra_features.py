import hydra_class
import numpy as np
import torch

def hydra_function_training(X_training, X_testing, k, g, kernel_length, stride, divisor, hydra_seed):
    X_training = torch.from_numpy(X_training)
    X_testing = torch.from_numpy(X_testing)
    
    transform = hydra_class.Hydra(X_training.shape[-1], k = k, g = g, kernel_length = kernel_length, stride = stride,  divisor = divisor, seed = hydra_seed)

    X_training_transform = transform.batch(X_training, batch_size = 4096)
    X_testing_transform = transform.batch(X_testing, batch_size = 4096)
    
    scaler = hydra_class.SparseScaler()
    X_training_transform = scaler.fit_transform(X_training_transform)
    X_testing_transform = scaler.transform(X_testing_transform)
    
    output = dict()
    output['X_training_transform'] = torch.Tensor.numpy(X_training_transform)
    output['X_testing_transform'] = torch.Tensor.numpy(X_testing_transform)
    output['transform'] = transform
    output['scaler'] = scaler
    
    return output

def hydra_function_prediction(X, transform, scaler):
    X = torch.from_numpy(X)
    
    X_transform = transform(X)
    X_transform = scaler.transform(X_transform)
    
    output = dict()
    output['X_transform'] = torch.Tensor.numpy(X_transform)
    
    return output