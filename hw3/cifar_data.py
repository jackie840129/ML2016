import pickle as p
import numpy as np
def load_all_label(path):
    l_data = p.load(open(str(path),'rb'))
    l_data = np.array(l_data)
    case = l_data.shape[0]
    num = l_data.shape[1]
    pixels = l_data.shape[2]
    height = int(np.sqrt(pixels/3))
    weight = int(np.sqrt(pixels/3))
    l_data = l_data.reshape([case*num,3,height,weight]).transpose(0,2,3,1)
    '''l_data's shape = (5000,32,32,3)'''
    label = []
    for c in range(case):
        for n in range(num):
            temp = np.zeros(10)
            temp[c]= 1.0
            label.append(temp)
    label = np.array(label)
    '''label's shape = (5000,10)'''
    return l_data,label

def load_all_unlabel(path):
    l_data = p.load(open(str(path),'rb'))
    l_data = np.array(l_data)
    num = l_data.shape[0]
    pixels = l_data.shape[1]
    height = int(np.sqrt(pixels/3))
    weight = int(np.sqrt(pixels/3))
    l_data = l_data.reshape([num,3,height,weight]).transpose(0,2,3,1)
    '''_data's shape = (45000,32,32,3)'''
    return l_data
def load_test(path):
    l_data = p.load(open(str(path),'rb'))
    l_data = np.array(l_data['data'])
    num = l_data.shape[0]
    pixels = l_data.shape[1]
    height = int(np.sqrt(pixels/3))
    weight = int(np.sqrt(pixels/3))
    l_data = l_data.reshape([num,3,height,weight]).transpose(0,2,3,1)
    '''_data's shape = (10000,32,32,3)'''
    return l_data
if __name__ == '__main__':
    l_data,label = load_all_label('data/all_label.p')
    u_data = load_all_unlabel('data/all_unlabel.p')
    t_data = load_test('data/test.p')
    np.save('l_data_keras.npy',l_data)
    np.save('label_leras.npy',label)
    np.save('u_data_keras.npy',u_data)
    np.save('t_data_keras.npy',t_data)
      

