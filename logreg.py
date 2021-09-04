from typing import List
import random
import numpy as np
import math
np.random.seed(42);
class LogisticRegression:
    def __init__(self, learning_rate: float,epoch:int, batch_size:int):
        self.learning_rate = learning_rate;
        self.batch_size = batch_size;
        self.epoch=epoch;
        self.learning_decay=1;
        self.decay_factor=0.75;
        self.momentum=0
        self.mu=1
        pass
    
    def fit(self, X: List[List[int]], y: List[List[int]]):
        train_loss_list, train_accu_list = [],[]
        X_np=np.array(X);
        y_np=np.array(y);
        self.y=y_np;
        num_inputs = X_np.shape[1]
        num_classes = 10;
     
        self.param = self.initialize(num_inputs,num_classes)
        
        if bool(self.momentum) == True:
            w_velocity = np.zeros(self.param['w'].shape)
            b_velocity = np.zeros(self.param['b'].shape)
        for epoch in range(self.epoch):
            rand_indices = np.random.choice(X_np.shape[0],X_np.shape[0],replace=False)
            num_batch = int(X_np.shape[0]/self.batch_size)
            batch_loss50 = 0
            for batch in range(num_batch):
                index = rand_indices[self.batch_size*batch:self.batch_size*(batch+1)]
                x_batch = X_np[index]
                y_batch = y_np[index]
                # calculate the gradient w.r.t w and b
                dw, db, batch_loss = self.mini_batch_gradient(self.param, x_batch, y_batch)
                batch_loss50 += batch_loss
                if bool(self.momentum) == True:
                    w_velocity = self.mu * w_velocity + self.learning_rate * dw
                    b_velocity = self.mu * b_velocity + self.learning_rate * db
                    self.param['w'] -= w_velocity
                    self.param['b'] -= b_velocity
                else:
                    self.param['w'] -= self.learning_rate * dw
                    self.param['b'] -= self.learning_rate * db
                if batch % 100 == 0:
                    message = 'Epoch %d, Batch %d, Loss %.2f' % (epoch+1, batch, batch_loss)
                    print(message)
                    batch_loss50 = 0;
            train_loss, train_accu = self.eval2(self.param,X_np,y_np);
            print("Train loss: ")
            print(train_loss)
            print("Train Accuracy: ")
            print(train_accu)
            train_loss_list.append(train_loss)
            train_accu_list.append(train_accu)
        return train_loss, train_accu;
        
            
            
        pass
    def predict(self, X: List[List[int]]):
        x_data=np.array(X);
        loss_list = []
        w = self.param['w'].transpose()
        dist = np.array([np.squeeze(self.softmax(np.matmul(x_data[i], w))) for i in range(x_data.shape[0])])
        result = np.argmax(dist,axis=1)
        result=result.tolist();
        return result;
        
    def initialize(self,num_inputs,num_classes):
        w = np.random.randn(num_classes, num_inputs) / np.sqrt(num_classes*num_inputs) # (10*784)
        b = np.random.randn(num_classes, 1) / np.sqrt(num_classes) # (10*1) 
        param = {
        'w' : w, # (10*784)
        'b' : b  # (10*1)
        }
        return param
        
    def mini_batch_gradient(self,param, x_batch, y_batch):
        batch_size = x_batch.shape[0]
        w_grad_list = []
        b_grad_list = []
        batch_loss = 0
        for i in range(batch_size):
            x,y = x_batch[i],y_batch[i]
            x = x.reshape((784,1)) # x: (784,1)
            E = np.zeros((10,1)) #(10*1)
            E[y][0] = 1 
            pred = self.softmax(np.matmul(param['w'], x)+param['b']) #(10*1)

            loss = self.neg_log_loss(pred, y)
            batch_loss += loss

            w_grad = E - pred
            w_grad = - np.matmul(w_grad, x.reshape((1,784)))
            w_grad_list.append(w_grad)

            b_grad = -(E - pred)
            b_grad_list.append(b_grad)

        dw = sum(w_grad_list)/batch_size
        db = sum(b_grad_list)/batch_size
        return dw, db, batch_loss
    def softmax(self,z):
         exp_list = np.exp(z)
         result = 1/sum(exp_list) * exp_list
         result = result.reshape((len(z),1))
         assert (result.shape == (len(z),1))
         return result
    def neg_log_loss(self,pred, label):
        loss = -np.log(pred[int(label)])
        return loss
    def eval2(self,param, x_data, y_data):
        loss_list = []
        w = param['w'].transpose()
        dist = np.array([np.squeeze(self.softmax(np.matmul(x_data[i], w))) for i in range(len(y_data))])

        result = np.argmax(dist,axis=1)
        accuracy = sum(result == y_data)/float(len(y_data));

        loss_list = [self.neg_log_loss(dist[i],y_data[i]) for i in range(len(y_data))]
        loss = sum(loss_list)
        return loss, accuracy
        
        