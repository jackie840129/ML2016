import numpy as np
import sys
import csv
model_name = sys.argv[1]
test_path = sys.argv[2]
output = sys.argv[3]

ind = 0
for i in range(len(model_name)):
    if model_name[i]=='.':
        ind = i
if ind != 0:
    model_name = model_name[:ind]

W = np.load(str(model_name)+'.npy')

T = []
file2 = open(test_path,'r',encoding='utf-8')
for row in csv.reader(file2):
    T.append(row[1:])
T = np.array(T).astype('float32')
one = np.ones((len(T),1))
T = np.concatenate((T,one),axis=1)

file3  = open (output,'w')
file3.write('id,label\n')
answer = 1.0/(1.0+np.exp(-np.dot(T,W)))
answer = (answer+0.5).astype('int').astype('float32')
for i in range(len(answer)):
    file3.write(str(i+1)+','+str(answer[i][0].astype('int'))+'\n')

file2.close()
file3.close()


