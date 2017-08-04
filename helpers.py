import numpy as np
import pandas as pd
import os
import keras
import keras.callbacks as cb
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dense,Activation,Dropout,Input
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM

def save_live_data(msgs, live_path = 'live_data/msgs.csv'):
    data = pd.DataFrame(msgs)
    data.to_csv(live_path)
    
    print('Messages saved as {} !  Data Size : {}'.format(live_path, data.shape))
    

def load_live_data(path='live_data/msgs.csv'):
    data = pd.read_csv('live_data/msgs.csv')
    data = preprocess_data(data, use_cache=True)
    durations = data['duration']
    data = data.drop(['duration'], axis=1)
    inputs = data.as_matrix()
    outputs = durations.as_matrix()
    inputs, outputs = split_data(inputs, outputs, num_timesteps=30)

    print('inputs shape : {} ; outputs shape : {}'.format(inputs.shape, outputs.shape))
    return inputs, outputs



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


def rescale_data(rucio_data, durations):
    # Normalization
    # using custom scaling parameters (based on trends of the following variables)

#     durations = durations / 1e3
    rucio_data['bytes'] = rucio_data['bytes'] / 1e8
    rucio_data['delay'] = rucio_data['delay'] / 1e3
#     rucio_data['src-rse'] = rucio_data['src-rse'] / 1e2
#     rucio_data['dst-rse'] = rucio_data['dst-rse'] / 1e2
    
    return rucio_data, durations

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
    print(len(inputs))
    inputs = np.stack(inputs)
    outputs = np.stack(outputs)
    print(inputs.shape, outputs.shape)
    
    return inputs, outputs

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
    data['bytes'] = data['bytes'] * 1e8
    data['delay'] = data['delay'] * 1e3
    # data['src-rse'] = data['src-rse'] * 1e2
    # data['dst-rse'] = data['dst-rse'] * 1e2
    
    data = data.round().astype(int)
    print(data.shape)
    data = decode_labels(data)
    data['duration'] = y
    data['prediction'] = preds
    # data['duration'] = data['duration'] * 1e3
    # data['prediction'] = data['prediction'] * 1e3
    
    return data

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

def model_input_generator(data, durations, num_timesteps=100):
    n=data.shape[0]
    n_events = (n - num_timesteps +1)
    print('Total Data points/Events: {} of {} timesteps each.'.format(n_events, num_timesteps))

    inputs=[]
    outputs=[]
    for i in range(0,n_events):
        print('Batch :{} of {}'.format(i, n_events))
        v = data[i:i+num_timesteps]
        w = durations[i+num_timesteps-1]
        v = np.reshape(v, [1,num_timesteps, 9])
        yield v,w

def input_batch_generator(rucio_data,durations, num_timesteps=50):
    
    #slice_size = batch_size*num_timesteps
    #print(rucio_data.shape[0], durations.shape)
    n_examples = rucio_data.shape[0]
    n_batches = (n_examples - num_timesteps +1)
    print('Total Data points for training/testing : {} of {} timesteps each.'.format(n_batches, num_timesteps))
    batch_size = n_batches//50
    print('batch size :', batch_size)
    n_batches= batch_size*50
    print('n_batches:', n_batches)
    for start_index in range(0, n_batches, batch_size):
        inputs=[]
        outputs=[]
        print('batch:',start_index)
        for i in range(start_index,start_index+batch_size):
            v = rucio_data[i:i+num_timesteps]
            w = durations[i+num_timesteps-1]
            inputs.append(v)
            outputs.append(w)
        inputs = np.stack(inputs)
        outputs = np.stack(outputs)
        #print(inputs.shape, outputs.shape)
        yield (inputs, outputs)
