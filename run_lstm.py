
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
import datetime 
import seaborn as sns
import os
import time
from sklearn.preprocessing import LabelEncoder
import keras
import tensorflow as tf
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dense,Activation,Dropout,Input
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
import keras.callbacks as cb
from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import LSTM
from keras_tqdm import TQDMNotebookCallback

from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model


path = 'data/'

def load_encoders():
    src_encoder = LabelEncoder()
    dst_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    activity_encoder = LabelEncoder()
    protocol_encoder = LabelEncoder()
    t_endpoint_encoder = LabelEncoder()
    
    src_encoder.classes_ = np.load('encoders/ddm_rse_endpoints.npy')
    dst_encoder.classes_ = np.load('encoders/ddm_rse_endpoints.npy')
    type_encoder.classes_ = np.load('encoders/type.npy')
    activity_encoder.classes_ = np.load('encoders/activity.npy')
    protocol_encoder.classes_ = np.load('encoders/protocol.npy')
    t_endpoint_encoder.classes_ = np.load('encoders/endpoint.npy')
    
    return (src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder)

def train_encoders(rucio_data, use_cache=True):
    
    if use_cache:
        if os.path.isfile('encoders/ddm_rse_endpoints.npy') and os.path.isfile('encoders/activity.npy'):
            print('using cached LabelEncoders for encoding data.....')
            src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder=load_encoders()
        else:
            print('NO cache found')
    else:
        print('No cached encoders found ! Training Some New Ones using input data!')
        src_encoder = LabelEncoder()
        dst_encoder = LabelEncoder()
        type_encoder = LabelEncoder()
        activity_encoder = LabelEncoder()
        protocol_encoder = LabelEncoder()
        t_endpoint_encoder = LabelEncoder()

        src_encoder.fit(rucio_data['src-rse'].unique())
        dst_encoder.fit(rucio_data['dst-rse'].unique())
        type_encoder.fit(rucio_data['src-type'].unique())
        activity_encoder.fit(rucio_data['activity'].unique())
        protocol_encoder.fit(rucio_data['protocol'].unique())
        t_endpoint_encoder.fit(rucio_data['transfer-endpoint'].unique())

        np.save('encoders/src.npy', src_encoder.classes_)
        np.save('encoders/dst.npy', dst_encoder.classes_)
        np.save('encoders/type.npy', type_encoder.classes_)
        np.save('encoders/activity.npy', activity_encoder.classes_)
        np.save('encoders/protocol.npy', protocol_encoder.classes_)
        np.save('encoders/endpoint.npy', t_endpoint_encoder.classes_)
    
    return (src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder)


# In[3]:

def preprocess_data(rucio_data, use_cache=True):
    
    fields_to_drop = ['account','reason','checksum-adler','checksum-md5','guid','request-id','transfer-id','tool-id',
                      'transfer-link','name','previous-request-id','scope','src-url','dst-url', 'Unnamed: 0']
    timestamps = ['started_at', 'submitted_at','transferred_at']

    #DROP FIELDS , CHANGE TIME FORMAT, add dataetime index
    rucio_data = rucio_data.drop(fields_to_drop, axis=1)
    for timestamp in timestamps:
        rucio_data[timestamp]= pd.to_datetime(rucio_data[timestamp], infer_datetime_format=True)
    rucio_data['delay'] = rucio_data['started_at'] - rucio_data['submitted_at']
    rucio_data['delay'] = rucio_data['delay'].astype('timedelta64[s]')
    
    rucio_data = rucio_data.sort_values(by='submitted_at')
    
    # Reindex data with 'submitted_at timestamp'
    rucio_data.index = pd.DatetimeIndex(rucio_data['submitted_at'])
    
    #remove all timestamp columns
    rucio_data = rucio_data.drop(timestamps, axis=1)
    
    # encode categorical data
 
    if use_cache==True:
        src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = train_encoders(rucio_data, use_cache=True)
    else:
        src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = train_encoders(rucio_data, use_cache=False)

    rucio_data['src-rse'] = src_encoder.transform(rucio_data['src-rse'])
    rucio_data['dst-rse'] = dst_encoder.transform(rucio_data['dst-rse'])
    rucio_data['src-type'] = type_encoder.transform(rucio_data['src-type'])
    rucio_data['dst-type'] = type_encoder.transform(rucio_data['dst-type'])
    rucio_data['activity'] = activity_encoder.transform(rucio_data['activity'])
    rucio_data['protocol'] = protocol_encoder.transform(rucio_data['protocol'])
    rucio_data['transfer-endpoint'] = t_endpoint_encoder.transform(rucio_data['transfer-endpoint'])
    
    return rucio_data


# In[4]:

def rescale_data(rucio_data, durations):
    # Normalization
    # using custom scaling parameters (based on trends of the following variables)

    durations = durations / 1e3
    rucio_data['bytes'] = rucio_data['bytes'] / 1e10
    rucio_data['delay'] = rucio_data['delay'] / 1e5
    rucio_data['src-rse'] = rucio_data['src-rse'] / 1e2
    rucio_data['dst-rse'] = rucio_data['dst-rse'] / 1e2
    
    return rucio_data, durations

def plot_graphs_and_rescale(data):
    
    durations = data['duration']
    durations.plot()
    plt.ylabel('durations(seconds)')
    plt.show()

    filesize = data['bytes']
    filesize.plot(label='filesize(bytes)')
    plt.ylabel('bytes')
    plt.show()

    delays = data['delay']
    delays.plot(label='delay(seconds)')
    plt.ylabel('delay')
    plt.show()
    
    print('rescaling input continuous variables : filesizes, queue-times, transfer-durations')
    data, byte_scaler, delay_scaler, duration_scaler = rescale_data(data)

    plt.plot(data['bytes'], 'r', label='filesize')
    plt.plot(data['duration'], 'y', label='durations')
    plt.plot(data['delay'],'g', label='queue-time')
    plt.legend()
    plt.xticks(rotation=20)
    plt.show()
    
    return data, byte_scaler, delay_scaler, duration_scaler


# In[5]:

def prepare_model_inputs(rucio_data,durations, num_timesteps=50):
    
    #slice_size = batch_size*num_timesteps
    print(rucio_data.shape[0], durations.shape)
    n_examples = rucio_data.shape[0]
    n_batches = (n_examples - num_timesteps +1)
    print('Total Data points for training/testing : {} of {} timesteps each.'.format(n_batches, num_timesteps))
    
    inputs=[]
    outputs=[]
    for i in range(0,n_batches):
        v = rucio_data[i:i+num_timesteps]
        w = durations[i+num_timesteps-1]
        inputs.append(v)
        outputs.append(w)
    inputs = np.stack(inputs)
    outputs = np.stack(outputs)
    print(inputs.shape, outputs.shape)
    
    return inputs, outputs



def get_rucio_files(path='../', n_files =100):
    abspaths = []
    for fn in os.listdir(path):
        if 'atlas_rucio' in fn:
            abspaths.append(os.path.abspath(os.path.join(path, fn)))
    print("\n Found : ".join(abspaths))
    print('\n total files found = {}'.format(len(abspaths)))
    return abspaths

def load_rucio_data(file, use_cache = True, limit=None):
    print('reading : {}'.format(file))
    data = pd.read_csv(file)
    if limit != None:
        data= data[950000: 950000+limit]
        print('Limiting data size to {} '.format(limit))
#     print(data)
    print('preprocessing data... ')
    data = preprocess_data(data)
    print('Saving indices for later..')
    indices = data.index
    durations = data['duration']
    data = data.drop(['duration'], axis=1)
    data = data[['bytes', 'delay', 'activity', 'dst-rse', 'dst-type',
                 'protocol', 'src-rse', 'src-type', 'transfer-endpoint']]
    data, durations = rescale_data(data, durations)
    data = data.as_matrix()
    durations = durations.as_matrix()
    return data, durations, indices


# In[7]:

# path='data/'
# a= get_rucio_files(path=path)
# x, y, indices = load_rucio_data(a[1], limit=5)

# print(x ,'\n', y, '\n', indices)


# In[8]:

# x,y = prepare_model_inputs(x,y,num_timesteps=2)


# In[10]:

def return_to_original(x, y, preds, index=None):
    #print(x.shape, y.shape)
    #print(x[0,1])
    n_steps = x.shape[1]
    #print(index[:n_steps])
    #print(index[n_steps-1:])
    index = index[n_steps-1:]
    
    cols = ['bytes', 'delay', 'activity', 'dst-rse', 'dst-type','protocol', 'src-rse', 'src-type', 'transfer-endpoint']
    data = list(x[0])
    for i in range(1,x.shape[0]):
        data.append(x[i,n_steps-1,:])
    
    data = data[n_steps-1:]
    #print(len(data))
    data = pd.DataFrame(data, index=index, columns=cols)
    data['bytes'] = data['bytes'] * 1e10
    data['delay'] = data['delay'] * 1e5
    data['src-rse'] = data['src-rse'] * 1e2
    data['dst-rse'] = data['dst-rse'] * 1e2
    
    data = data.round().astype(int)
    print(data.shape)
    data = decode_labels(data)
    data['duration'] = y
    data['prediction'] = preds
    data['duration'] = data['duration'] * 1e3
    data['prediction'] = data['prediction'] * 1e3
    
    return data

# return_to_original(x,y, index=indices)


# In[11]:

def decode_labels(rucio_data):
    src_encoder,dst_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = load_encoders()
    
    rucio_data['src-rse'] = src_encoder.inverse_transform(rucio_data['src-rse'])
    rucio_data['dst-rse'] = dst_encoder.inverse_transform(rucio_data['dst-rse'])
    rucio_data['src-type'] = type_encoder.inverse_transform(rucio_data['src-type'])
    rucio_data['dst-type'] = type_encoder.inverse_transform(rucio_data['dst-type'])
    rucio_data['activity'] = activity_encoder.inverse_transform(rucio_data['activity'])
    rucio_data['protocol'] = protocol_encoder.inverse_transform(rucio_data['protocol'])
    rucio_data['transfer-endpoint'] = t_endpoint_encoder.inverse_transform(rucio_data['transfer-endpoint'])
    
    return rucio_data

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


# In[12]:

def build_model(num_timesteps=50, batch_size = 512, parallel=False):

    model = Sequential()
    layers = [512, 512, 512, 512, 128, 1]
    
    model.add(LSTM(layers[0], input_shape=(num_timesteps, 9), return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(layers[1], return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(layers[3]))
    model.add(Activation("relu"))
    
    model.add(Dense(layers[4]))
    model.add(Activation("relu"))
    
    model.add(Dense(layers[5]))
    model.add(Activation("linear"))
    
    start = time.time()
    
    if parallel:
        model = make_parallel(model,4)
    
    model.compile(loss="mse", optimizer="adam")
    print ("Compilation Time : ", time.time() - start)
    return model


# In[13]:

def plot_losses(losses):
    sns.set_context('poster')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    print(len(losses))
    fig.show()
        

def load_lstm():
    json_file = open('models/lstm_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/lstm_model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss="mse", optimizer="adam")
    print('Model model compiled!!')
    return loaded_model

def evaluate_network(limit=None, n_timesteps=100, path="data/",model=None):
    
    print('\n Locating training data files...')
    a= get_rucio_files(path=path)
    

    for i,file in enumerate(a):
        print("Training on file :{}".format(file))
        x, y, indices = load_rucio_data(file, limit=limit)
        print('\n Data Loaded and preprocessed !!....')
        x, y = prepare_model_inputs(x, y, num_timesteps=n_timesteps)
        print('Data ready for Evaluation')
        
    
        start_time = time.time()
        print('making predictions...')
        model = load_lstm()
        predictions = model.predict(x)
        #score = model.evaluate(x,y)
        #print('score: {}'.format(score))
        end = time.time() - start_time
        print('Done !! in {} min'.format(end/60))


        # print('plotting graphs')

        # plt.plot(y, 'g')
        # plt.plot(predictions, 'y')
        # plt.show()

        data = return_to_original(x, y, predictions, index=indices)
        error = data['duration'] - data['prediction']
        print("Mean : {} ; std :{} ; mAX : {} ; min: {} ".format(np.mean(error),np.std(error),np.max(error),np.min(error)))
        data['mae']= error
        filename = "predictions/pred_"+str(i)+".csv"
        data.to_csv(filename)

        # plt.plot(data['duration'], 'g')
        # plt.plot(data['prediction'], 'y')
        # plt.title('Network predictions')
        # plt.ylabel('durations in seconds')
        # plt.show()

        
        # plt.show()


# In[16]:

evaluate_network(path='data/', limit = 10000)


# In[17]:

# a= get_rucio_files(path=path)
# x, y, indices = load_rucio_data(a[2], limit=1000)
# print('\n Data Loaded and preprocessed !!....')
# x, y = prepare_model_inputs(x, y, num_timesteps=100)
# with tf.device('/gpu:0'):
#     model = load_lstm()
#     pred= model.predict(x)
#     print('done')


# In[18]:

# plt.plot(pred)


# In[ ]:



