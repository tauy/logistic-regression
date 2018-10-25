import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,Imputer

train = pd.read_csv('C:\\Users\\Administrator\\Desktop\\train.csv')
indata = train.ix[0:,2:15]

test = pd.read_csv('C:\\Users\\Administrator\\Desktop\\test.csv')
intest = test.ix[0:,1:14]

imputer = Imputer(strategy='median')
imputer.fit(indata)
indata = imputer.transform(indata)
intest = imputer.transform(intest)

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s    

data_train = indata[0:1599,:]
result_train = indata[0:1599,1:2].T

minmax_data_train = MinMaxScaler() 
data_train_std = minmax_data_train.fit_transform(data_train).T

input_size = 13 
hidden_size = 20 
output_size = 1

iters_num = 1000
train_size = data_train.shape[0] 
learning_rate = 0.035

weight_init_std = 0.01 
W1 = weight_init_std * np.random.randn(hidden_size, input_size) 
b1 = np.zeros([hidden_size,1]) 
W2 = weight_init_std * np.random.randn(output_size, hidden_size) 
b2 = np.zeros([output_size,1])

for i in range(iters_num): 
    z1 = np.dot(W1,data_train_std) + b1 
    h = sigmoid(z1)
    z2 = np.dot(W2, h) + b2 
    y = sigmoid(z2)
    
    delta2 = result_train - y 
    dW2 = np.dot(delta2,np.transpose(h)) 
    db2 = np.dot(delta2,np.ones([train_size,1]))
    
    delta1 = np.dot(np.transpose(W2), delta2) * h * (1-h) 
    dW1 = np.dot(delta1,np.transpose(data_train_std)) 
    db1 = np.dot(delta1,np.ones([train_size,1]))
    
    W2 += learning_rate * dW2 
    b2 += learning_rate * db2 
    W1 += learning_rate * dW1 
    b1 += learning_rate * db1

z1 = np.dot(W1,data_train_std) + b1 
h = sigmoid(z1) 
z2 = np.dot(W2, h) + b2 
y_train = z2

minmax_data_test = MinMaxScaler() 
data_test_std = minmax_data_test.fit_transform(intest).T

test_z1 = np.dot(W1,data_test_std) + b1
test_h = sigmoid(test_z1)
test_z2 = np.dot(W2, test_h) + b2 
test_y = test_z2
ave=test_y.mean()

for i in range(test_y.shape[1]):
    if test_y[0,i]>=ave:
        test_y[0,i]=1
    else:
        test_y[0,i]=0

output = pd.read_csv('C:\\Users\\Administrator\\Desktop\\submission.csv')
data_output = output.ix[0:,1].as_matrix()
data_output = data_output.reshape(1,400)

a=0
for i in range(test_y.shape[1]):
    if test_y[0,i]==data_output[0,i]:
        a+=1
print(a)
