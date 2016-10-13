import csv
import numpy as np
def normalize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    X=X.T
    for i in range(len(X)-1):
        X[i]=(X[i]-mean[i])/std[i]
    return X.T
def almost_concat_remove_parameter(file,normalize_num):
    data = []
    for row in csv.reader(file):
        for i in range(len(row)):
            if row[i] == 'NR':
                row[i] = '0'
        data.append(row)
    file.close()
    for i in range(len(data)):
        data[i] = data[i][3:]
    data = np.array(data)
    data = np.delete(data,[0],0) 
    ################################# data's shape = (18x240, 24)
    all_data = (np.split(data,240,0))
    X = []
    Y = []
    for i in range(len(all_data)):#240
        # all_data[i] = np.array(all_data[i][:])#4~10
        all_data[i] =np.delete(all_data[i],[0,1,2,3,4,10,11,13,14,15,16,17],0)        
        all_data[i] = np.transpose(all_data[i]).astype('float32')
    for i in range(12):
        New = all_data[20*i]
        for x in range(1,20):
            New = np.concatenate((New,all_data[20*i+x]))
        for x in range(len(New)-9):
            X.append(np.append(New[x:x+9].reshape(9*all_data[0].shape[1]),1.0))
            Y.append(New[x+9][4])
    X,Y = np.array(X),np.array(Y)
    
    T = read_testdata()
    if normalize_num == 1:
        temp = normalize(np.concatenate((X,T),axis=0))
        X = temp[:len(X)][:]
        T = temp[len(X):][:]

    index = np.random.permutation(len(X))
    X_i,V_i = index[:5000],index[5000:]
    X, V = X[X_i,:],X[V_i,:]
    Y,Y_v = np.array(Y)[X_i],np.array(Y)[V_i]
    print('Training data\'s shape = ',X.shape)
    return X,Y,V,Y_v,T
    
def read_testdata():
    file2 = open('data/test_X.csv','r')
    data = []
    for row in csv.reader(file2):
        row = row[2:]
        for i in range(len(row)):
            if row[i] == 'NR':
                row[i] = '0'
        data.append(row)
    file2.close()
    data = np.array(data)
    test_data = np.split(data,240,0)
    for i in range(len(test_data)):
        test_data[i] = np.delete(test_data[i],[0,1,2,3,4,10,11,13,14,15,16,17],0)
        test_data[i] = test_data[i].astype('float32').T #4~10
        test_data[i] = np.append(test_data[i].reshape(test_data[i].shape[0]*test_data[i].shape[1]),1.0)
    test_data = np.array(test_data)
    return test_data

if __name__ == '__main__':
    file = open('data/train.csv','r',encoding='big5')
    print('Use cross vali_ remove para_method" !')
    X,Y,V,Y_v,T = almost_concat_remove_parameter(file,0)
    np.save('data/X.npy',X)
    np.save('data/Y_.npy',Y)
    np.save('data/V.npy',V)
    np.save('data/Y_v.npy',Y_v)
    print('Successfully save X.npy & Y_.npy & V.npy')
    print('Read Test Data')
    np.save('data/T.npy',T)
    print(X.shape,Y.shape,V.shape,Y_v.shape,T.shape)
    print('================================')

