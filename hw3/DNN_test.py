import numpy as np
import keras.backend.tensorflow_backend
import sys
from keras.models import load_model
import tensorflow as tf
import cifar_data as cd

path = sys.argv[1]
model = sys.argv[2]
output = sys.argv[3]

encoder = load_model('encoder.h5')
print('load pickle......')
t_data = cd.load_test(path+'test.p')
t_data = t_data.astype('float32')/255.
t_data = encoder.predict(t_data,verbose=1).reshape(t_data.shape[0],-1)
print('t_data data\'s feature:',t_data.shape)


DNN = load_model(model)
answer = DNN.predict_classes(t_data)  #(1000,)
print('\n')
predict_path = output
file = open(predict_path,'w')

file.write('ID,class\n')
for i in range(len(answer)):
    file.write((str(i)+','+str(answer[i])+'\n'))
if keras.backend.tensorflow_backend._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   keras.backend.tensorflow_backend._SESSION.close()
   keras.backend.tensorflow_backend._SESSION = None
