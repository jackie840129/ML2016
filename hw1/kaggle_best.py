import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import read_data as rd
X = np.load('data/X_b.npy')
Y_ = np.load('data/Y_b.npy')
V = np.load('data/V_b.npy')
Y_v = np.load('data/Y_vb.npy')
T = np.load('data/T.npy')
num_W = X.shape[1]
W = np.dot(np.linalg.pinv(X),Y_)
# W = np.zeros(num_W,dtype='float32')
print('successffuly load X,Y_,V,Y_v,T !')
print('X shape,Y_shape,V shape,Y_v shape, Test shape')
print(X.shape,Y_.shape,V.shape,Y_v.shape,T.shape)
print('=========start====training!=====')
print('lr = ',0.0000016,' Epoch = 200000')
lr = 0.0000016 
Epoch =10000#35
loss_list=[]
vali_list=[]
index = []
for i in range(Epoch):
    if i%1000 == 0 :#and i!=0:
        loss = np.sqrt((np.linalg.norm(Y_-np.dot(X,W))**2)/ len(X))
        loss_vali = np.sqrt((np.linalg.norm(Y_v-np.dot(V,W))**2) / len(V))
        index.append(i/1000)
        loss_list.append(loss)
        vali_list.append(loss_vali)
        if i%3 ==0:
            print(' training .  %d iteration\r'%(i),end='')
        elif i%3 ==1:
            print(' training  . %d iteration\r'%(i),end='')
        else:
            print(' training   .%d iteration\r'%(i),end='')

    Grad =(2*np.dot(np.transpose(X),(np.dot(X,W)-Y_))) / len(X)    
    W = W - lr*Grad  #(lr/np.sqrt(sum_g))*Grad
print('\n',loss_list[-1],vali_list[-1])
print('End Training!')
print('=============')

print('retrain!!!!!')
Epoch = 50000
print('Epoch=100000')
print(X.shape,Y_.shape)
X = np.concatenate((X,V),axis=0)
Y_ = np.concatenate((Y_,Y_v),axis=0)
for i in range(Epoch):
    if i%1000 == 0 :#and i!=0:
        loss = np.sqrt((np.linalg.norm(Y_-np.dot(X,W))**2)/ len(X))
        if i%3 ==0:
            print(' training .  %d iteration\r'%(i),end='')
        elif i%3 ==1:
            print(' training  . %d iteration\r'%(i),end='')
        else:
            print(' training   .%d iteration\r'%(i),end='')

    Grad =(2*np.dot(np.transpose(X),(np.dot(X,W)-Y_))) / len(X) #+2*lamda[r]*(W)
    W = W - lr*Grad  #(lr/np.sqrt(sum_g))*Grad
###########################
print('\n',loss)
# '''

np.save('data/W_b.npy',W)


W = np.load('data/W_b.npy')
An = []
Cor = []
file3  = open ('output/kaggle_best.csv','w')
file3.write('id,value\n')
for i in range(240):
    an = ((np.dot(T[i],W)))
    An.append(an)
    file3.write('id_'+str(i)+','+str(an)+'\n')


