import numpy as np
import pandas as pd
import keras
import keras.callbacks as cb
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dense,Activation,Dropout,Input
from sklearn.preprocessing import LabelEncoder
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
from helpers import *

#data dir
file = 'data/atlas_rucio-events-2017.08.06.csv'
model = load_lstm()

data = pd.read_csv(file)
# data = data[:1000]
# print(data.info)
data = preprocess_data(data)
print(data.shape)
indices = data.index
names=data['name']
scopes=data['scope']
durations = data['duration']
data = data.drop(['name','duration', 'scope'], axis=1)
data = data[['bytes', 'delay', 'activity', 'dst-rse', 'dst-type',
             'protocol', 'src-rse', 'src-type', 'transfer-endpoint']]
data, durations = rescale_data(data, durations)
data = data.as_matrix()
durations = durations.as_matrix()
print('preparing model inputs ...')


#n=data.shape[0]
#gen_iterations = n-99
# pred = model.predict_generator(input_batch_generator(data, durations, num_timesteps=100), steps=50,verbose=1)

data, durations = prepare_model_inputs(data, durations, num_timesteps=100)
pred=model.predict(data,batch_size=1024, verbose=1)

print('done making predictions..')
data = return_to_original(data, durations, pred, index=indices, file_names=names, scopes=scopes)
print(data.shape)
print('saving to csv as rucio_transfer-events-2017.08.06.csv ...')
data.to_csv('rucio_transfer-events-2017.08.06.csv')
