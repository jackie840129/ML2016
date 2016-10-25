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
one = np.ones((len(X),1))
X = np.concatenate((X,one),axis = 1)
index = np.load('index.npy')

Valid = X[index[-400:]]
V_y = Y_[index[-400:]]
# X = X[index[:-400]]
# Y_ = Y_[index[:-400]]
def sigmoid(z):
    np.seterr(all='ignore')
    return 1.0/(1.0+np.exp(-z))

def forward_for_batch(X,W):
    a_2 = sigmoid(np.dot(X,W[0]))
    a_2.T[0].fill(1.0)
    a_3 = sigmoid(np.dot(a_2,W[1]))
    return a_2,a_3
def Stop_condition(Y,temp_Y,e):
    global Valid,V_y,count
    distance = np.sqrt(np.mean(np.square(Y-temp_Y)))
    if distance < 0.0004:
        if count == 3:
            print('STOP!!!',e)
        else:
            count+=1
            return False
    else:
        count = 0
        return False
    test = (forward_for_batch(Valid,W)[1]+0.5).astype('int').astype('float32')
    acc = 1-np.mean(np.abs(test-V_y))
    print(e,distance,acc)
    return True
print('Training data ->X:',X.shape,'Label:',Y_.shape)
print('===============start training!==============')
Epoch = 20000
Hidden1 = 41
outputlayer =1
input_size = X.shape[1]
lr = 0.001
E = 1e-30
W = []
W_1 = np.random.normal(0.0,0.5,size=(input_size,Hidden1+1))
W_2 = np.random.normal(0.0,0.5,size=(Hidden1+1,outputlayer))
W.append(W_1)
W.append(W_2)
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
Adam = {'m1':0,'m2':0,'v1':0,'v2':0,'t':0,'lr':0}
temp_Y = np.zeros(Y_.shape)
count = 0
lamda = 0.01

for e in range(Epoch):
    a_2,a_3 = forward_for_batch(X,W)
    delta_3 = -(Y_*1/(a_3+E)-(1-Y_)*1/(1-a_3+E))*(a_3*(1-a_3))
    delta_2 = np.dot(delta_3,W[1].T)*(a_2*(1-a_2))
    Grad_1 = np.dot(X.T,delta_2)
    Grad_2 = np.dot(a_2.T,delta_3)
    
    Y = np.array(a_3)
    loss = np.mean(-(Y_*np.log(Y+E)+(1-Y_)*np.log(1-Y+E)))
    if e %50 ==0:
        print(e,loss)
    if e%5 == 0:
        S = Stop_condition(Y,temp_Y,e)
        if S:
            break
        temp_Y = Y
    Grad_1 =( Grad_1+lamda*W[0])
    Grad_2 =( Grad_2+lamda*W[1])
    
    Adam['t'] = Adam['t']+1
    Adam['lr'] = lr* np.sqrt(1-beta2**Adam['t'])/(1-beta1**Adam['t'])
    Adam['m1'] = beta1*Adam['m1']+(1-beta1)*Grad_1
    Adam['v1'] = beta2*Adam['v1']+(1-beta2)*Grad_1*Grad_1
    W[0] = W[0] - Adam['lr']*Adam['m1']/(np.sqrt(Adam['v1'])+epsilon)
    Adam['m2'] = beta1*Adam['m2']+(1-beta1)*Grad_2
    Adam['v2'] = beta2*Adam['v2']+(1-beta2)*Grad_2*Grad_2
    W[1] = W[1] - Adam['lr']*Adam['m2']/(np.sqrt(Adam['v2'])+epsilon)
W = np.array(W)

ind = 0
for i in range(len(model_name)):
    if model_name[i]=='.':
        ind = i
if ind != 0:
    model_name = model_name[:ind]

np.save(str(model_name)+'.npy',W)
'''
W = np.load('wwww.npy')
T = []
file2 = open('spam_data/spam_test.csv','r',encoding='utf-8')
for row in csv.reader(file2):
    T.append(row[1:])
T = np.array(T).astype('float32')
one = np.ones((len(T),1))
T = np.concatenate((T,one),axis=1)
answer = (forward_for_batch(T,W)[1]+0.5).astype('int').reshape(len(T))
print(answer.shape)
file3  = open ('nnfinal_repro.csv','w')
file3.write('id,label\n')
for i in range(len(answer)):
    file3.write(str(i+1)+','+str(answer[i])+'\n')
'''


