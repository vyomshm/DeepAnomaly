import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import keras.callbacks as cb
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dense,Activation,Dropout,Input
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
from keras_tqdm import TQDMNotebookCallback

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


def train_encoders(rucio_data, use_cache=True):
    
    if use_cache:
        if os.path.isfile('encoders/src.npy'):
            print('using cached LabelEncoders for encoding data.....')
            src_encoder,dst_encoder,scope_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder=load_encoders()
        else:
            print('NO cache found')
    else:
        print('No cached encoder ! Training some new ones....')
        src_encoder = LabelEncoder()
        dst_encoder = LabelEncoder()
        scope_encoder = LabelEncoder()
        type_encoder = LabelEncoder()
        activity_encoder = LabelEncoder()
        protocol_encoder = LabelEncoder()
        t_endpoint_encoder = LabelEncoder()

        src_encoder.fit(rucio_data['src-rse'].unique())
        dst_encoder.fit(rucio_data['dst-rse'].unique())
        scope_encoder.fit(rucio_data['scope'].unique())
        type_encoder.fit(rucio_data['src-type'].unique())
        activity_encoder.fit(rucio_data['activity'].unique())
        protocol_encoder.fit(rucio_data['protocol'].unique())
        t_endpoint_encoder.fit(rucio_data['transfer-endpoint'].unique())

        np.save('encoders/src.npy', src_encoder.classes_)
        np.save('encoders/dst.npy', dst_encoder.classes_)
        np.save('encoders/scope.npy', scope_encoder.classes_)
        np.save('encoders/type.npy', type_encoder.classes_)
        np.save('encoders/activity.npy', activity_encoder.classes_)
        np.save('encoders/protocol.npy', protocol_encoder.classes_)
        np.save('encoders/endpoint.npy', t_endpoint_encoder.classes_)
    
    return (src_encoder,dst_encoder,scope_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder)

def load_encoders(path='encoders/'):
    src_encoder = LabelEncoder()
    dst_encoder = LabelEncoder()
    scope_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    activity_encoder = LabelEncoder()
    protocol_encoder = LabelEncoder()
    t_endpoint_encoder = LabelEncoder()
    
    src_encoder.classes_ = np.load('encoders/src.npy')
    dst_encoder.classes_ = np.load('encoders/dst.npy')
    scope_encoder.classes_ = np.load('encoders/scope.npy')
    type_encoder.classes_ = np.load('encoders/type.npy')
    activity_encoder.classes_ = np.load('encoders/activity.npy')
    protocol_encoder.classes_ = np.load('encoders/protocol.npy')
    t_endpoint_encoder.classes_ = np.load('encoders/endpoint.npy')
    
    return (src_encoder,dst_encoder,scope_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder)

def preprocess_data(rucio_data, use_cache=True):
    
    fields_to_drop = ['account','reason','checksum-adler','checksum-md5','guid','request-id','transfer-id','tool-id',
                      'transfer-link','name','previous-request-id','src-url','dst-url', 'Unnamed: 0']
    timestamps = ['started_at', 'submitted_at','transferred_at']

    #DROP FIELDS , CHANGE TIME FORMAT
    rucio_data = rucio_data.drop(fields_to_drop, axis=1)
    for timestamp in timestamps:
        rucio_data[timestamp]= pd.to_datetime(rucio_data[timestamp], infer_datetime_format=True)
    rucio_data['delay'] = rucio_data['started_at'] - rucio_data['submitted_at']
    rucio_data['delay'] = rucio_data['delay'].astype('timedelta64[s]')
    
    rucio_data = rucio_data.sort_values(by='submitted_at')

    rucio_data = rucio_data.drop(timestamps, axis=1)
    
    # Normalization
    
    #filesize_max = np.max(rucio_data['bytes'].as_matrix())
    filesize_mean = np.mean(rucio_data['bytes'].as_matrix())
    filesize_std = np.std(rucio_data['bytes'].as_matrix())
    
    delay_mean = np.mean(rucio_data['delay'].as_matrix())
    delay_std = np.std(rucio_data['delay'].as_matrix())
    
    rucio_data['bytes'] = (rucio_data['bytes'] - filesize_mean) / filesize_std
    rucio_data['delay'] = (rucio_data['delay'] - delay_mean) / delay_std
    
    # encode categorical data
 
    if use_cache==True:
        src_encoder,dst_encoder,scope_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = train_encoders(rucio_data, use_cache=True)
    else:
        src_encoder,dst_encoder,scope_encoder,type_encoder,activity_encoder,protocol_encoder,t_endpoint_encoder = train_encoders(rucio_data, use_cache=False)

    rucio_data['src-rse'] = src_encoder.transform(rucio_data['src-rse'])
    rucio_data['dst-rse'] = dst_encoder.transform(rucio_data['dst-rse'])
    rucio_data['scope'] = scope_encoder.transform(rucio_data['scope'])
    rucio_data['src-type'] = type_encoder.transform(rucio_data['src-type'])
    rucio_data['dst-type'] = type_encoder.transform(rucio_data['dst-type'])
    rucio_data['activity'] = activity_encoder.transform(rucio_data['activity'])
    rucio_data['protocol'] = protocol_encoder.transform(rucio_data['protocol'])
    rucio_data['transfer-endpoint'] = t_endpoint_encoder.transform(rucio_data['transfer-endpoint'])
    
    return rucio_data


# # Load and preprocess data

# In[12]:

def get_and_preprocess_data(path='data/atlas_rucio-events-2017.06.01.csv', use_cache=True):
    
    rucio_data = pd.read_csv(path)
#     rucio_data = rucio_data[:800000]
    rucio_data = preprocess_data(rucio_data, use_cache=use_cache)
    durations = rucio_data['duration']
    rucio_data = rucio_data.drop(['duration'], axis=1)
    inputs = rucio_data.as_matrix()
    outputs = durations.as_matrix()
    print(inputs.shape, outputs.shape)
    return inputs, outputs

# # splitting data into test and training set

def split_data(rucio_data,durations, num_timesteps=30):
    
    # slice_size = batch_size*num_timesteps
    # print(rucio_data.shape[0])
    n_examples = rucio_data.shape[0]
    n_batches = (n_examples - num_timesteps )
    print('Total Batches : {}'.format(n_batches))
    
    inputs=[]
    outputs=[]
    for i in range(0,n_batches):
        v = rucio_data[i:i+num_timesteps]
        w = durations[i+num_timesteps]
        inputs.append(v)
        outputs.append(w)
    
    inputs = np.stack(inputs)
    outputs = np.stack(outputs)
    print(inputs.shape, outputs.shape)

    return inputs, outputs

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)
