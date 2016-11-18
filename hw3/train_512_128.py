import numpy as np
import cifar_data as cd
import data_preprocessing as dp
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Convolution2D,MaxPooling2D
from keras.optimizers import Adam
import keras.backend.tensorflow_backend
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
from keras.layers.normalization import BatchNormalization
from keras import backend as K
K.set_image_dim_ordering('tf')
######
from  keras.layers.advanced_activations import ELU
elu = ELU(alpha=1.0)
######
data_path = sys.argv[1]
model_path = str(sys.argv[2])
print('load pickle....')
l_data,label = cd.load_all_label(data_path+'all_label.p')
u_data = cd.load_all_unlabel(data_path+'all_unlabel.p')
t_data = cd.load_test(data_path+'test.p')

l_data = l_data.astype('float32')
label = label.astype('float32')
u_data = u_data.astype('float32')
t_data = t_data.astype('float32')
u_data = np.concatenate((u_data,t_data),axis=0)
####################################################
mc_path = model_path
mc = ModelCheckpoint(mc_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto')

batch_size = 32
nb_classes = 10

Train,T_label,Valid,V_label = dp.split_valid(l_data,label,ratio=0.1,is_seed=False,seed=6)

def construct_model():

    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same',input_shape=(32,32,3),dim_ordering='tf'))
    model.add(BatchNormalization())
    model.add(elu)
    model.add(Convolution2D(64, 3,3))
    model.add(BatchNormalization())
    model.add(elu)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(elu)
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization())
    model.add(elu)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
        
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(elu)
    model.add(Convolution2D(256, 3, 3))
    model.add(BatchNormalization())
    model.add(elu)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),metrics=['accuracy'])
    return model


model = construct_model()
print('Valid\'s shape',Valid.shape)
print('V_label\'s shape',V_label.shape)
print('Train\'s shape ',Train.shape)
print('T_label\'s shape',T_label.shape)
train_datagen = dp.IG(ro=0,s=0.0,z=0.0)

count = 0
while(1):
    nb_epoch = 20
    if count == 0:
        nb_epoch = 18
    train_datagen.fit(Train)
    t_history = model.fit_generator(train_datagen.flow(Train,T_label,batch_size=batch_size),
                        samples_per_epoch=4500*5,nb_epoch=nb_epoch,callbacks=[mc,es],
                        validation_data=(Valid,V_label))
    if count==6:
        break
    nb_epoch = 20
    predict = model.predict(u_data,verbose=1)#(45000,10)
    print('\n%d data are predicted,shape = %r' %(predict.shape[0],predict.shape))
    max_eles = np.max(predict,axis=1)#(45000,)
    # if c >3:
    index = np.argwhere(max_eles>0.99).flatten() #(?,)
    # else:
        # index = np.argwhere(max_eles>0.99).flatten() #(?,)
    compare = max_eles.reshape((max_eles.shape[0],1))[index] #(?,1)
    new_label = np.equal(predict[index],compare).astype('float32') #(?,10)
    confident_data = u_data[index] #(?,32,32,3)
    new_data = np.concatenate((Train,confident_data),axis=0)
    new_label = np.concatenate((T_label,new_label),axis=0) 
    print('%d data are confident'%(confident_data.shape[0]))
    print('%d data are retrained'%(new_data.shape[0]))
    train_datagen.fit(new_data)
    model.fit_generator(train_datagen.flow(new_data, new_label,batch_size=batch_size),
                        samples_per_epoch=new_data.shape[0],nb_epoch=nb_epoch,callbacks=[mc,es],
                        validation_data=(Valid,V_label))
    count+=1


if keras.backend.tensorflow_backend._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   keras.backend.tensorflow_backend._SESSION.close()
   keras.backend.tensorflow_backend._SESSION = None
