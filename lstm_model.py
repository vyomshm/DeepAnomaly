
# coding: utf-8

# In[75]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
from sklearn.preprocessing import LabelEncoder
from datetime import datetime,timedelta
import keras
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dense,Activation,Dropout,Input
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
# from keras_tqdm import TQDMNotebookCallback
# from ipywidgets import interact

from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
# output_notebook()

np.random.seed(7)
# get_ipython().magic('matplotlib inline')

print("All dependencies imported!! TF: {} ; Keras :{}".format(tf.__version__,keras.__version__))

def plot_data_stats(rucio_data):
    sns.set_context('poster')
    
    ax = sns.countplot(x='activity',data= rucio_data)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()
    gx= sns.countplot(x='transfer-endpoint', data = rucio_data)
    gx.set_xticklabels(gx.get_xticklabels(), rotation=30)
    plt.show()
    vx = sns.countplot(x='protocol', data = rucio_data)
    plt.show()
    bx= sns.countplot(x='src-type', data=rucio_data)
    plt.show()
    cx= sns.countplot(x='dst-type', data=rucio_data)
    plt.show()
    
def preprocess_data(rucio_data):
    
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

def get_and_preprocess_data(path='may.csv'):
    
    rucio_data = pd.read_csv(path)
    rucio_data = rucio_data[0:30000]
    rucio_data = preprocess_data(rucio_data)
    durations = rucio_data['duration']
    rucio_data = rucio_data.drop(['duration'], axis=1)
    inputs = rucio_data.as_matrix()
    outputs = durations.as_matrix()
    print(inputs.shape, outputs.shape)
    return inputs, outputs

# # splitting data into test and training set

# In[54]:

def split_data(rucio_data,durations, batch_size=512, num_timesteps=1, split_frac=0.9):
    
    slice_size = batch_size*num_timesteps
    print(rucio_data.shape[0])
    n_batches = int(rucio_data.shape[0] / slice_size)
    print('Total Batches : {}'.format(n_batches))
    
    
    rucio_data = rucio_data[0:n_batches*slice_size]
    durations = durations[0:n_batches*slice_size]
    
    print(rucio_data.shape, durations.shape)
    x = np.stack(np.split(rucio_data, n_batches*batch_size))
    y = np.stack(np.split(durations, n_batches*batch_size))
    
    print(x.shape, y.shape)
    
    split_idx = int(x.shape[0]*split_frac)
    trainX, trainY = x[:split_idx], y[:split_idx]
    testX, testY = x[split_idx:], y[split_idx:]
    print('Training Data shape:',trainX.shape, trainY.shape)
    print('Test Data shape: ',testX.shape, testY.shape)
    return trainX, trainY, testX, testY

def build_model():

    model = Sequential()
    layers = [15, 15, 10, 10, 1]
    
    model.add(LSTM(layers[0], input_shape=(1, 10), return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(layers[1], return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(layers[2]))
    model.add(Activation("sigmoid"))
    
    model.add(Dense(layers[3]))
    model.add(Activation("sigmoid"))
    
    model.add(Dense(layers[4]))
    model.add(Activation("linear"))
    
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print ("Compilation Time : ", time.time() - start)
    return model

import keras.callbacks as cb

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)



# def build_model(n_steps):
    
#     layers = [10, 10, 1]
#     model_inputs = Input(shape=[None,n_steps, 10])
    
#     layer_1 = LSTM(layers[0], return_sequences=True)(model_inputs)
#     layer_2 = LSTM(layers[1], return_sequences=False)(layer_1)
    
#     model_output = Dense(layer[2], activation='linear')
    
#     model = Model(input=model_inputs, output=model_output)

def plot_batch_losses(losses):
    sns.set_context('poster')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    fig.show()
    print(len(losses))


def run_network(model=None,data=None, epochs=1,n_timesteps=1, batch=128):
    
    print('\n Loading data...')
    if data is None:
        rucio_data, durations = get_and_preprocess_data()

        print('\n Data Loaded and preprocesses!!....')
        print('\n Moving on to splitting and reshaping data...')
        trainX, trainY, testX, testY = split_data(rucio_data, durations,batch_size=batch,num_timesteps=1, split_frac=0.9)
        print('\n Data split into train and test sets.. ')
    else:
        trainX, trainY, testX, testY = data
        print('\n Data split into train and test sets.. ')

    try:
        start_time = time.time()
        
        if model is None:
            model = build_model()

            history = LossHistory()

            print('Training model...')
            training = model.fit(trainX, trainY, epochs=epochs, batch_size=batch,
                                 validation_split=0.1, callbacks=[history], verbose=1)

            print("Training duration : {0}".format(time.time() - start_time))
            score = model.evaluate(trainX, trainY, verbose=0)

            print("Network's training score [MSE]: {0}, in seconds : {1}".format(score, np.sqrt(score)))
            print("Training finished !!!!!!")
            
            print('\n Saving model to disk..')
            # serialize model to JSON
            model_json = model.to_json()
            with open("model_new.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model_new.h5") 
            print("Saved model to disk")
            data = (trainX, trainY, testX, testY)
            return training, data, model, history.losses
        
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        return model, history.losses

  # # Plotting Performance

    # In[70]:
def performance_plots(training):
    sns.set_context('poster')

    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper right')
    plt.show(block=False)


    # In[71]:
def test_lstm(model, data):
    trainX, trainY, testX, testY = data
    test_score = model.evaluate(testX, testY, verbose=2)

    predictions=model.predict(testX)
    assert len(predictions)==len(testY)
    mae = testY - predictions
    plt.plot(predictions[0:100],'y' )
    plt.plot(testY[0:100], 'g')
    plt.show(block=False)

    return test_score, mae

def run_lstm():
    training, data, model, losses = run_network(data=None, epochs=1, batch=128)
    trainX, trainY, testX, testY = data
    plot_batch_losses(losses)

    # # Reloading Saved MOdel

    # load model json and compile model
    json_file = open('model_new.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_new.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss="mse", optimizer="rmsprop")
    print('Model model compiled!!')

    score = loaded_model.evaluate(trainX, trainY, verbose=0)
    print("Network's training score [MSE]: {0}, in seconds : {1}".format(score, np.sqrt(score)))

    performance_plots(training)

    test_score, mae=test_lstm(model=loaded_model, data=data)
    print('Results: MSE - {:.6f}  ;  seconds = {:.3f}'.format(score, np.sqrt(score)))
    print('Maximum error : {} ; Minimum error : {}'.format(np.max(mae), np.min(mae)))



run_lstm()

