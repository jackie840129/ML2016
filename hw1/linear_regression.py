import csv
import numpy as np
import sys
import read_data as rd
file = open('data/train.csv','r',encoding='big5')
X,Y_,V,Y_v,T = rd.almost_concat_remove_parameter(file,0)
num_W = X.shape[1]
W = np.dot(np.linalg.pinv(X),Y_)
# W = np.zeros(num_W,dtype='float32')
print('successffuly load X,Y_,V,Y_v,T !')
print('=========start====training!=====')

lr = 0.0000016 #*np.array([10,5,1.8,1.5,1.0,0.1])#float(sys.argv[1])#0.0001
# lamda = [0,10,50,200,1000]#float(sys.argv[1])#10
Epoch =100000#35
# sum_g = 0
# plt.figure(1)
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

    Grad =(2*np.dot(np.transpose(X),(np.dot(X,W)-Y_))) / len(X) #+2*lamda[r]*(W)
    # sum_g+=np.square(Grad)
    W = W - lr*Grad  #(lr/np.sqrt(sum_g))*Grad
print('\n',loss_list[-1],vali_list[-1])
print('End Training!')
print('=============')

W_ = np.dot(np.linalg.pinv(X),Y_)
print('pinv answer',np.sqrt((np.linalg.norm(Y_-np.dot(X,W_))**2)/ len(X)))

print('retrain!!!!!')
Epoch = 100000
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
    # sum_g+=np.square(Grad)
    W = W - lr*Grad  #(lr/np.sqrt(sum_g))*Grad
print('\n',loss)
###########################


np.save('data/W.npy',W)

# W_see = np.absolute(W[:-1].reshape(8,19).T)
# W_min = np.min(W_see,axis=1)
# W_max = np.max(W_see,axis=1)
# print (W_min)
# print(W_max)

W = np.load('data/W.npy')
An = []
Cor = []
file3  = open ('output/linear_regression.csv','w')
file3.write('id,value\n')
for i in range(240):
    an = ((np.dot(T[i],W)))
    An.append(an)
    file3.write('id_'+str(i)+','+str(an)+'\n')
# file_o = open('data/W_see.txt','w')

# for i in range(len(W_see)):
    # file_o.write('P%d:' %i)
    # for x in range(len(W_see[i])):
        # file_o.write(str(W_see[i][x])+' ')
    # file_o.write('\n')

