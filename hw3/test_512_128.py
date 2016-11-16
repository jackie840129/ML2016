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
from keras.models import load_model
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
from keras.layers.normalization import BatchNormalization
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))
data_path = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[3]
print('load pickle...')
t_data = cd.load_test(data_path+'test.p')
t_data = t_data.astype('float32')


model = load_model(model_path)
print('test_data\'s shape',t_data.shape)
answer = model.predict_classes(t_data)
print('\n')


file = open(output_path,'w')

file.write('ID,class\n')
for i in range(len(answer)):
    file.write((str(i)+','+str(answer[i])+'\n'))
file.close()
if keras.backend.tensorflow_backend._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   keras.backend.tensorflow_backend._SESSION.close()
   keras.backend.tensorflow_backend._SESSION = None



