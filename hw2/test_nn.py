import csv
import numpy as np
import sys

model_name = sys.argv[1]
test_path = sys.argv[2]
output = sys.argv[3]

def sigmoid(z):
    np.seterr(all='ignore')
    return 1.0/(1.0+np.exp(-z))
def forward_for_batch(X,W):
    a_2 = sigmoid(np.dot(X,W[0]))
    a_2.T[0].fill(1.0)
    a_3 = sigmoid(np.dot(a_2,W[1]))
    return a_2,a_3

ind = 0
for i in range(len(model_name)):
    if model_name[i]=='.':
        ind = i
if ind != 0:
    model_name = model_name[:ind]

W = np.load(str(model_name)+'.npy')
T = []

file2 = open(str(test_path),'r',encoding='utf-8')

for row in csv.reader(file2):
    T.append(row[1:])
T = np.array(T).astype('float32')
one = np.ones((len(T),1))
T = np.concatenate((T,one),axis = 1)
answer = (forward_for_batch(T,W)[1]+0.5).astype('int').reshape(len(T))

file3 = open(str(output),'w')
file3.write('id,label\n')
for i in range(len(answer)):
    file3.write(str(i+1)+','+str(answer[i])+'\n')
file3.close()
          

