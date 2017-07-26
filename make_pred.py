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
file = 'data/atlas_rucio-events-2017.07.24.csv'
model = load_lstm()

data = pd.read_csv(file)
# data = data[:10000]
data = preprocess_data(data)
print(data.shape)
indices = data.index
durations = data['duration']
data = data.drop(['duration'], axis=1)
data = data[['bytes', 'delay', 'activity', 'dst-rse', 'dst-type',
             'protocol', 'src-rse', 'src-type', 'transfer-endpoint']]
data, durations = rescale_data(data, durations)
data = data.as_matrix()
durations = durations.as_matrix()
print('preparing model inputs ...')
# data, durations = prepare_model_inputs(data, durations, num_timesteps=100)
n=data.shape[0]
gen_iterations = n-99
pred = model.predict_generator(model_input_generator(data, durations, num_timesteps=100), steps=gen_iterations,workers=4)

data = return_to_original(data, durations, pred, index=indices)
print(data.shape)
print('saving to csv..')
data.to_csv('rucio_transfer-events-2017.07.24.csv')



