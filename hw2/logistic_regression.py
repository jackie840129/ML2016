import csv
import numpy as np
import sys

input_path = sys.argv[1]
model_name = sys.argv[2]
file = open(str(input_path),'r',encoding='utf-8')
X=[]
Y_=[]
for  row in csv.reader(file):
    X.append(row[1:-1])
    Y_.append([row[-1]])
X = np.array(X).astype('float32')
Y_ = np.array(Y_).astype('float32')
index = np.load('index.npy')
Valid = X[index[-400:]]
X = X[index[:-400]]
V_y = Y_[index[-400:]]
Y_ = Y_[index[:-400]]

# X = np.delete(X,[53,54,55],axis=1)
# print(X.shape)
one = np.ones((len(X),1))
X = np.concatenate((X,one),axis=1)
one = np.ones((len(Valid),1))
Valid = np.concatenate((Valid,one),axis=1)
print('Training data ->X:',X.shape,' Label:',Y_.shape)
print('Validation data -> Valid:',Valid.shape,' Label:',V_y.shape)
print('=========start====training!=====')
W_num = X.shape[1]
W = np.random.normal(0.0,0.2,size=(W_num,1))/100
print('Weight -> W:',W.shape)

def ValiTest(W_):
    global Valid,V_y
    e = 1e-20
    test = 1.0/(1.0+np.exp(-np.dot(Valid,W_)))
    loss = np.mean(-(V_y*np.log(test+e)+(1-V_y)*np.log(1-test+e)))
    return loss
temp_loss = 10000
min_W = 0
min_epoch = 0
Epoch = 100000
lr = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
Adam = {'m':0,'v':0,'t':0,'lr':0}
e = 1e-20
for i in range(Epoch):
    Y = 1.0/(1.0+np.exp(-np.dot(X,W)))
    loss = np.mean(-(Y_*np.log(Y+e)+(1-Y_)*np.log(1-Y+e)))
    vali_loss = ValiTest(W)
    if i%500 ==0:
        print(i,loss,vali_loss)
        if vali_loss > temp_loss and i>1000: 
            print(vali_loss,temp_loss)
            break
        temp_loss = vali_loss
    min_W = W
    min_epoch = i
    Grad = -np.dot(X.T,(Y_-Y))/len(X)
    Adam['t'] = Adam['t']+1
    Adam['lr'] = lr* np.sqrt(1-beta2**Adam['t'])/(1-beta1**Adam['t'])
    Adam['m'] = beta1*Adam['m']+(1-beta1)*Grad
    Adam['v'] = beta2*Adam['v']+(1-beta2)*Grad*Grad
    W = W - Adam['lr']*Adam['m']/(np.sqrt(Adam['v'])+epsilon)
print(min_epoch,temp_loss)




print('Retraining!')
Epoch = 60000
W = min_W
X = np.concatenate((X,Valid),axis = 0)
Y_ = np.concatenate((Y_,V_y),axis = 0)

Adam = {'m':0,'v':0,'t':0,'lr':0}
temp_loss =0
for i in range(Epoch):
    Y = 1.0/(1.0+np.exp(-np.dot(X,W)))
    loss = np.mean(-(Y_*np.log(Y+e)+(1-Y_)*np.log(1-Y+e)))
    if i%1000 ==0:
        print(i,loss)
    Grad = -np.dot(X.T,(Y_-Y))/len(X)
    Adam['t'] = Adam['t']+1
    Adam['lr'] = lr* np.sqrt(1-beta2**Adam['t'])/(1-beta1**Adam['t'])
    Adam['m'] = beta1*Adam['m']+(1-beta1)*Grad
    Adam['v'] = beta2*Adam['v']+(1-beta2)*Grad*Grad
    W = W - Adam['lr']*Adam['m']/(np.sqrt(Adam['v'])+epsilon)

ind = 0
for i in range(len(model_name)):
    if model_name[i]=='.':
        ind = i
if ind != 0:
    model_name = model_name[:ind]
np.save(str(model_name)+'.npy',W)
file.close()

