import numpy as np
import matplotlib.pyplot as plt

#Creating a 2 layer neural network(1 hidden layer) with sigmoid activation function and cross entropy loss 
class NN():
    def __init__(self, X,y):
        '''
        args: X=array; independent variables or features
              y=array; dependent variable or label
        '''
        self.X=X
        self.y=y
        self.xshape=np.shape(X)
        self.yshape=np.shape(y)
        self.xtranspose=np.transpose(X)
        assert self.xshape[0]==self.yshape[0]

    def init_weight(self, units):
        '''
        args: units=integer; number of neurons in hidden layer
        returns: w1=array; weight matrix of input layer
                 b1=array; bias of input layer
                 w2=array; weight matrix of hidden layer
                 b2=array; bias of hidden layer
        '''
        units=int(units)
        np.random.seed(42)
        w1=np.random.normal(size=(units, self.xshape[1]))
        b1=np.random.normal(size=1)
        w2=np.random.normal(size=(units))
        b2=np.random.normal(size=1)
        return w1, b1, w2, b2

    def create_equ(self, weights, x, bias):
        '''
        Generating equation z=w.T x + b
        args: weights=array; weight matrix 
              x=array; input matrix of features
              bias=array; bias
        returns z=array; linear combination of weights, features and bias 
        '''
        z=np.matmul(weights, x) + bias
        return z 

    def sigmoid(self, z):
        '''
        Sigmmoid activation function 1/(1+e-z) popularly used for binary classification
        args: z=array; linear equation of weights, features and bias
        returns y_pred=array; predicted y
        '''
        y_pred=1/(1+(np.exp(-z)))
        return y_pred

    def loss_function(self,y, y_pred):
        '''
        Cross entropy loss function loss=(y*log(y_pred) + (1-y)*log(1-y_pred)) utilized as cost function for classification algorithms
        args: y=array; label of dataset
              y_pred=array; predicted y after sigmoid activation
        returns: loss=float; cross entropy cost function
        '''
        loss= -1/y.shape[0]*(np.sum((y*np.log(y_pred)) + ((1-y) * np.log(1-y_pred))))
        return loss 

    def forward_pass(self, w1, b1, w2, b2):
        '''
        Forward propagation step
        args: w1=array; weight matrix of input layer
              b1=array; bias of input layer
              w2=array; weight matrix of hidden layer
              b2=array; bias of hidden layer
        returns: a1=array; result matrix of hidden layer
                 y_pred=array; predicted y after forward pass 
        '''
        z1=self.create_equ(w1, self.xtranspose, b1) 
        a1=self.sigmoid(z1)
        z2=self.create_equ(w2, a1, b2)
        y_pred=self.sigmoid(z2)
        return a1,y_pred

    def dz(self, ypred):
        '''
        Computes derivative of z (dL/dz)-differential of loss with respect to linear output of hidden layer 
        args: ypred=array; predicted y
        returns: dz=array; derivative of z 
        '''
        dz=ypred-self.y
        return dz

    def hidden_layer_backprop(self,ypred,a1):
        '''
        Backpropagation step to update weight and bias for hidden layer
        args: ypred=array; predicted y
              a1=array; hidden layer matrix
        returns: dw2=array; derivative of hidden layer weight matrix (dL/dw2)
                 db2=array; derivative of hidden layer bias (dL/db2)
        '''
        dz=self.dz(ypred)
        dw2=1/ypred.shape[0] * (np.matmul(dz, np.transpose(a1)))
        db2=1/ypred.shape[0] * (np.sum(dz, axis=0,keepdims=True))
        return dw2, db2
     
    def input_layer_backprop(self,ypred,w2 ):
        '''
        Backpropagation step to update weight and bias for input layer
        args: ypred=array; predicted y
              w2=array; weight matrix of hidden layer
        returns: dw_1=array; derivative of input layer weight matrix (dL/dw1)
                 db_1=array; derivative of input layer bias (dL/db1)
        '''
        dz=self.dz(ypred)
        #dL/da-derivative of hidden layer matrix
        da=dz*(w2.reshape(w2.shape[0],1))
        dz_1=np.exp(-da) / np.square((1+ np.exp(-da)))
        dw_1=1/ypred.shape[0] *(np.matmul(dz_1, self.X))
        db_1=1/ypred.shape[0] * ( np.sum(dz_1, axis=1, keepdims=True))
        return dw_1, db_1

    def backward_pass(self, ypred, a1, w2):
        '''
        Combining both back propagation steps
        args: ypred=array; predicted y
              a1=array; hidden layer matrix
              w2=array; weight matrix of hidden layer
        returns: dw2=array; derivative of hidden layer weight matrix (dL/dw2)
                 db2=array; derivative of hidden layer bias (dL/db2)
                 dw1=array; derivative of input layer weight matrix (dL/dw1)
                 db1=array; derivative of input layer bias (dL/db1)

        '''
        dw2, db2=self.hidden_layer_backprop(ypred, a1)
        dw1, db1=self.input_layer_backprop(ypred,w2)
        return dw2,db2,dw1,db1

    def compile(self, units, iterations, learn_rate):
        '''
        Combining both forward and backward propagation with gradient descent to update weights and bias 
        args: units=integer; number of neurons in hidden layer
              iterations=integer; number of iterations to run
              learn_rate=float; learning rate
        returns: lis_loss=list; list of cost function after each iteration
        '''
        lis_loss=[]
        w1, b1, w2, b2=self.init_weight(units)
        for i in range(iterations):
            a1, y_pred=self.forward_pass(w1, b1, w2, b2)
            loss=self.loss_function(self.y, y_pred)
            lis_loss.append(loss)
            dw2,db2,dw1,db1=self.backward_pass(y_pred,a1,w2)
            w1=w1-(learn_rate * dw1)
            w2=w2-(learn_rate * dw2)
            b1=b1-(learn_rate * db1)
            b2=b2-(learn_rate * db2)
        return lis_loss

    def plot_loss(self, units, iterations, learn_rate):
        '''
        Plot of loss for each iterative step
        args: units=integer; number of neurons in hidden layer
              iterations=integer; number of iterations to run
              learn_rate=float; learning rate
        returns: matplotlib plot
        '''
        loss=self.compile(units, iterations, learn_rate)
        plt.figure(figsize=(5,6))
        plt.plot(range(1, len(loss)+1), loss)
        plt.show()
        

