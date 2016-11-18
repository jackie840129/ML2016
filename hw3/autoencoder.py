import numpy as np
import sys
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adamax ,RMSprop
import keras.backend.tensorflow_backend
import data_preprocessing as dp
import cifar_data as cd
from keras import backend as K
K.set_image_dim_ordering('tf')

path = sys.argv[1]
print('load pickle....')
l_data,label = cd.load_all_label(path+'all_label.p')
u_data = cd.load_all_unlabel(path+'all_unlabel.p')
l_data = l_data.astype('float32')/255.
u_data = u_data.astype('float32')/255.
label = label.astype('float32')

Train,T_label,Valid,V_label=dp.split_valid(l_data,label,ratio=0.1,is_seed=False)
all_data = np.concatenate((Train,u_data),axis=0)
np.random.shuffle(all_data)

learning_rate = 0.007
input_img = Input(shape=(32,32,3))
nb_epoch = 50
batch_size = 256
x = Convolution2D(32, 2, 2, activation='relu', border_mode='same',dim_ordering='tf')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 2, 2, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 2, 2, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)
# at this point the representation is (2, 8, 8) i.e. 128-dimensional

x = Convolution2D(8, 2, 2, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 2, 2, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 2, 2, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img,encoded)
autoencoder.compile(optimizer=Adamax(lr=learning_rate), loss='mse')
autoencoder.fit(all_data, all_data,batch_size=batch_size,nb_epoch=nb_epoch,shuffle=True,
                validation_data=(Valid,Valid))
# autoencoder.save('autoencode.h5')
encoder.save('encoder.h5')
# pred = autoencoder.predict(Valid[0].reshape(1,32,32,3))
# skimage.io.imsave('origin.png',Valid[0])
# skimage.io.imsave('after.png',pred.reshape(32,32,3))
if keras.backend.tensorflow_backend._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   keras.backend.tensorflow_backend._SESSION.close()
   keras.backend.tensorflow_backend._SESSION = None

