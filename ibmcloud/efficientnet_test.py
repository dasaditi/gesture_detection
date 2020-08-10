import os
import os
import numpy as np
np.random.seed(251)
os.environ['PYTHONHASHSEED']=str(251)
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import class_weight
from keras_efficientnets import EfficientNetB0
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

project_path = '/root/w251-data-local/data/'
dir_openpose = 'processed'
dir_train = 'NEW4_image_transfer_trial4'
dir_test = 'NEW4_manual_optical_flow_output_trial4'
model_name = 'Efficientnet_model_weights_NEW4_trial4.h5'

model_saved = load_model(os.path.join(project_path, model_name))

HEIGHT = 224
WIDTH = 224

class_list = ['AGAIN', 'ALL', 'AWKWARD', 'BASEBALL', 'BEHAVIOR', 'CAN', 'CHAT', 'CHEAP', 
               'CHEAT', 'CHURCH', 'COAT', 'CONFLICT', 'COURT', 'DEPOSIT', 'DEPRESS', 
               'DOCTOR', 'DRESS', 'ENOUGH', 'NEG']

def conv_index_to_vocab(ind):
    temp_dict = dict(enumerate(class_list))
    return temp_dict[ind]
def conv_vocab_to_index(vocab):
    temp_dict = dict(zip(class_list,range(len(class_list))))
    return temp_dict[vocab]

print(conv_index_to_vocab(0))
print(conv_vocab_to_index('NEG'))

correct_count = 0
count = 0
test_files_lst = [f for f in os.listdir(os.path.join(project_path, dir_openpose, dir_test))
                  if 'val' in f and 'sim0' in f]
for file in test_files_lst:
    img = image.load_img(os.path.join(project_path,dir_openpose, dir_test,file), target_size=(HEIGHT, WIDTH))
    x = image.img_to_array(img)
    # print(x.shape)
    x = np.expand_dims(x, axis=0)
    # print(x.shape)
    x = preprocess_input(x)
    y_pred = model_saved.predict(x)
    print('-----------------')
    print('Actual: ', file.split('_')[1])
    print('Prediction: ', conv_index_to_vocab(np.argmax(y_pred)))
    # print(y_pred)
    count += 1
    if file.split('_')[1] == conv_index_to_vocab(np.argmax(y_pred)):
        correct_count += 1

print('Accuracy = ', correct_count/len(test_files_lst))

