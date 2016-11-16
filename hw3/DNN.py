import numpy as np
import data_preprocessing as dp
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Convolution2D,MaxPooling2D
from keras.optimizers import Adam
import keras.backend.tensorflow_backend
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import cifar_data as cd

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


mc_path = sys.argv[2]   # './DNN.h5'
mc = ModelCheckpoint(mc_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')

path = sys.argv[1]

def construct_model(feature_size):
    model = Sequential()
    model.add(Dense(128,input_dim=feature_size,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0003),metrics=['accuracy'])
    return model
print('load pickle!')
l_data,label = cd.load_all_label(path+'all_label.p')
u_data = cd.load_all_unlabel(path+'all_unlabel.p')
l_data = l_data.astype('float32')/255.
label = label.astype('float32')
u_data = u_data.astype('float32')/255.

encoder = load_model('encoder.h5')

Train,T_label,Valid,V_label = dp.split_valid(l_data,label,ratio=0.1,is_seed=False)

Train = encoder.predict(Train,verbose=1).reshape(Train.shape[0],-1)
Valid = encoder.predict(Valid,verbose=1).reshape(Valid.shape[0],-1)
u_data = encoder.predict(u_data,verbose=1).reshape(u_data.shape[0],-1)
print('Label data\'s feature:',Train.shape)
print('Valid data\'s feature:',Valid.shape)
print('u_data data\'s feature:',u_data.shape)

batch_size= 128
nb_epoch = 10

count=0
DNN = construct_model(Train.shape[1])
while(1):
    if count == 0:
        nb_epoch = 100
    else:
        nb_epoch = 40
    DNN.fit(Train,T_label,batch_size= batch_size,nb_epoch=nb_epoch,validation_data=(Valid,V_label),verbose=1,
            callbacks=[mc,es])
    predict = DNN.predict(u_data,verbose=1)#(45000,10)
    if count==10:
        break   
    print('\n%d data are predicted,shape = %r' %(predict.shape[0],predict.shape))
    max_eles = np.max(predict,axis=1)#(45000,)
    index = np.argwhere(max_eles>0.8).flatten() #(?,)
    compare = max_eles.reshape((max_eles.shape[0],1))[index] #(?,1)
    new_label = np.equal(predict[index],compare).astype('float32') #(?,10)
    confident_data = u_data[index] #(?,32,32,3)
    new_data = np.concatenate((Train,confident_data),axis=0)
    new_label = np.concatenate((T_label,new_label),axis=0) 
    print('%d data are confident'%(confident_data.shape[0]))
    print('%d data are retrained'%(new_data.shape[0]))
    
    DNN.fit(new_data,new_label,batch_size= batch_size,nb_epoch=30,validation_data=(Valid,V_label),verbose=1,
            callbacks=[mc,es])
    count+=1

if keras.backend.tensorflow_backend._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   keras.backend.tensorflow_backend._SESSION.close()
   keras.backend.tensorflow_backend._SESSION = None
