from keras.preprocessing.image import ImageDataGenerator
import numpy as np
def split_valid(l_data,label,ratio=0.1,is_seed=False,seed=0):
    Valid = []
    V_label = []
    Train = []
    T_label = []
    for i in range(len(l_data)):
        
        if i%500 < int(500*ratio):
            Valid.append(l_data[i])
            V_label.append(label[i])
        else :
            Train.append(l_data[i])
            T_label.append(label[i])
    Valid = np.array(Valid)
    V_label = np.array(V_label)
    Train = np.array(Train)
    T_label = np.array(T_label)
    if is_seed:
        np.random.seed(seed)
    index_t = np.random.permutation(len(Train))
    index_v = np.random.permutation(len(Valid))
    Train = Train[index_t]
    T_label = T_label[index_t]
    Valid = Valid[index_v]
    V_label = V_label[index_v]
    return Train,T_label,Valid,V_label

def IG(ro=20,w=0.2,h=0.2,s=0,z=0):
    datagen = ImageDataGenerator(
        rotation_range=ro,
        width_shift_range=w,
        height_shift_range=h,
        shear_range=s,
        zoom_range=z,
        horizontal_flip=False,
        fill_mode='nearest')
    return datagen

