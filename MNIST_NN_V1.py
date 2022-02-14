import numpy as np 
import matplotlib as plt
import pandas as pd 

data_train = pd.read_csv('mnist_train.csv')
data_train = np.array(data_train)   #shape is 42000*785, first column is labels

X_train = data_train[:, 1:].astype("float32")/255         # (42000,784) shape
X_train = X_train.T                 # (784, 42000)
Y_train = data_train[:,0]             # 42000 vector

Y_onehot = np.zeros((Y_train.size, Y_train.max()+1))
Y_onehot[np.arange(Y_train.size), Y_train] = 1
Y_onehot = Y_onehot.T               # (10,42000) shape

data_test = pd.read_csv('mnist_test.csv')
data_test = np.array(data_test)
X_test = data_test[:, 1:].astype("float32")/255           # (28000,784) shape
X_test = X_test.T                 # (784, 28000)
Y_test = data_test[:,0]   

layer_sizes = [784, 20, 10]
L = len(layer_sizes)
m = X_train.shape[0]

def init_para(layer_sizes):
    parameters = {}
    for l in range(1,L):       
        parameters['W'+ str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1])
        parameters['b'+ str(l)] = np.zeros((layer_sizes[l],1))
    return parameters

def ReLU_activation (Z):
    return np.maximum(0, Z)

def Softmax (Z):
    S = Z - np.max(Z, keepdims=True)
    expo = np.exp(S)
    return expo/np.sum(expo, axis=0, keepdims=True)

def forward_prop(input, weights, biases, n_layers):
    Z = np.dot(weights,input) + biases
    if n_layers != L-1 : 
       A = ReLU_activation(Z)
    else :
        A = Softmax(Z)
    return Z, A
    
def backword_prop (dA, cache):
    Z, A_prev, W, b = cache
    dZ = dA
    dZ[Z <= 0] = 0                  #This step is only applicable for ReLU activation backprop
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def upgrade_para(parameters, grads, alpha):
    for l in range(1, L):
        parameters['W'+str(l)] = parameters['W'+str(l)] - alpha*grads['dW'+str(l)]
        parameters['b'+str(l)] = parameters['b'+str(l)] - alpha*grads['db'+str(l)]
    return parameters

def Accuracy(A_in, Y_in):
    pred = np.argmax(A_in, axis=0)
    return np.sum(pred == Y_in)/ Y_in.size

def model (X_train, Y_train, layer_sizes, iterations, alpha):
    m = X_train.shape[0]
    parameters = init_para(layer_sizes)
 
    for i in range(0, iterations):
        A_prev = X_train
        caches = []
        grads = {}
        for l in range(1,L):
            Z, A = forward_prop(A_prev,parameters['W'+str(l)], parameters['b'+str(l)], l)
            cache = (Z, A_prev, parameters['W'+str(l)], parameters['b'+str(l)])    
            A_prev = A
            #print(A[:, 0:10])
            caches.append(cache)

        dZ = A - Y_onehot
        accuracy = Accuracy(A, Y_train)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ","{:.6f}".format(accuracy))
            
        current_cache = caches[L-2]    
        A_prev = current_cache[1]
        grads['dW'+str(L-1)] = 1/m * np.dot(dZ, A_prev.T) 
        grads['db'+str(L-1)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(parameters['W'+str(L-1)].T, dZ)
        for l in reversed(range(1,L-1)):
            dA_prev, dW, db = backword_prop(dA, caches[0])
            dA = dA_prev
            grads['dW'+str(l)] = dW
            grads['db'+str(l)] = db

        parameters = upgrade_para(parameters, grads, alpha)    
    
    return parameters

parameters= model(X_train, Y_train, layer_sizes, iterations= 1000, alpha=0.01)   

def test_pred (X_test, Y_test, parameters):
    A_prev = X_test
    for l in range(1,L):
        Z, A = forward_prop(A_prev,parameters['W'+str(l)], parameters['b'+str(l)], l)
        A_prev = A
        predictions = np.argmax(A, axis = 0)

    result = np.stack((predictions, Y_test), axis = 1) # np.concatenate((predictions, Y_test), axis = 1)
    return result, Accuracy(A, Y_test)
    

Result, test_acc = test_pred(X_test, Y_test, parameters)
print("Test Accuracy: ","{:.6f}".format(test_acc))
#pd.DataFrame(Result).to_csv("result1.csv")
np.savetxt("result2.csv", Result, delimiter=",")