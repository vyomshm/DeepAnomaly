
# coding: utf-8

# In[31]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
import datetime 
import seaborn as sns
import os
import time
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import keras
import tensorflow as tf
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dense,Activation,Dropout,Input
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
import keras.callbacks as cb
from keras.callbacks import ModelCheckpoint
from keras.layers.recurrent import LSTM
#from keras_tqdm import TQDMNotebookCallback
from multi_gpu import to_multi_gpu
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model


# In[3]:

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


# In[4]:

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


# In[5]:

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


# In[6]:

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


# In[56]:

path = '../' # Change this as you need.

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
        data= data[:limit]
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


# In[57]:

# a= get_rucio_files(path=path)
# x, y, indices = load_rucio_data(a[12], limit=5)

# print(x ,'\n', y, '\n', indices)


# In[58]:

# x,y = prepare_model_inputs(x,y,num_timesteps=2)


# In[59]:

def return_to_original(x, y, index=None):
    y = y * 1e3
    print(x.shape, y.shape)
    print(x[0])
    cols = ['bytes', 'delay', 'activity', 'dst-rse', 'dst-type','protocol', 'src-rse', 'src-type', 'transfer-endpoint']
    n_steps = x.shape[1]
    data = list(x[0])
    for i in range(1,x.shape[0]):
        data.append(x[i,n_steps-1,:])
    data = pd.DataFrame(data, index=indices, columns=cols)
    data['bytes'] = data['bytes'] * 1e10
    data['delay'] = data['delay'] * 1e5
    data['src-rse'] = data['src-rse'] * 1e2
    data['dst-rse'] = data['dst-rse'] * 1e2
    
    data = data.round().astype(int)
    data = decode_labels(data)
    return data


# In[60]:

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



# def make_parallel(model, gpu_count):
#     def get_slice(data, idx, parts):
#         shape = tf.shape(data)
#         size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
#         stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
#         start = stride * idx
#         return tf.slice(data, start, size)

#     outputs_all = []
#     for i in range(len(model.outputs)):
#         outputs_all.append([])

#     #Place a copy of the model on each GPU, each getting a slice of the batch
#     for i in range(gpu_count):
#         with tf.device('/gpu:%d' % i):
#             with tf.name_scope('tower_%d' % i) as scope:

#                 inputs = []
#                 #Slice each input into a piece for processing on this GPU
#                 for x in model.inputs:
#                     input_shape = tuple(x.get_shape().as_list())[1:]
#                     slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
#                     inputs.append(slice_n)                

#                 outputs = model(inputs)
                
#                 if not isinstance(outputs, list):
#                     outputs = [outputs]
                
#                 #Save all the outputs for merging back together later
#                 for l in range(len(outputs)):
#                     outputs_all[l].append(outputs[l])

#     # merge outputs on CPU
#     with tf.device('/cpu:0'):
#         merged = []
#         for outputs in outputs_all:
#             merged.append(merge(outputs, mode='concat', concat_axis=0))
            
#         return Model(input=model.inputs, output=merged)

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
        model = to_multi_gpu(model,4)
    
    model.compile(loss="mse", optimizer="adam")
    print ("Compilation Time : ", time.time() - start)
    return model


# In[65]:

def plot_losses(losses):
    sns.set_context('poster')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    print(len(losses))
    fig.show()

def train_network(model=None,limit=None, data=None, epochs=1,n_timesteps=100, batch=128, path="data/",parallel=True):
    
    if model is None:
        model = build_model(num_timesteps=n_timesteps, parallel=parallel)
        history = LossHistory()
        #os.makedir('/tmp')
        checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
        print('model built and compiled !')
    
    print('\n Locating training data files...')
    a= get_rucio_files(path=path)
    
    try:
        for i,file in enumerate(a):
            print("Training on file :{}".format(file))
            x, y, indices = load_rucio_data(file, limit=limit)
            print('\n Data Loaded and preprocessed !!....')
            x, y = prepare_model_inputs(x, y, num_timesteps=n_timesteps)
            print('Data ready for training.')

            start_time = time.time()

            print('Training model...')
            if parallel:
                training = model.fit(x, y, epochs=epochs, batch_size=batch*4,
                                     validation_split=0.1, callbacks=[history, checkpointer])
            else:
                training = model.fit(x, y, epochs=epochs, batch_size=batch,
                                     validation_split=0.1, callbacks=[history, checkpointer])

            print("Training duration : {0}".format(time.time() - start_time))
            #score = model.evaluate(x, y, verbose=0)
            #print("Network's Residual training score [MSE]: {0} ; [in seconds]: {1}".format(score,np.sqrt(score)))
            print("Training on {} finished !!".format(file))
            print('\n Saving model to disk..')
            # serialize model to JSON
            model_json = model.to_json()
            with open("models/lstm_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("models/lstm_model.h5")
            print("Saved model to disk")
            #print('plotting losses..')
            #plot_losses(history.losses)

        print('Training Complete !!')
        
        return training, model, indices, history.losses

    except KeyboardInterrupt:
            print('KeyboardInterrupt')
            return model, history.losses


# In[ ]:

train_network(n_timesteps=100,limit=None, batch=300, parallel= True)


