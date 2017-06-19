
# coding: utf-8

# In[92]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from datetime import datetime,timedelta
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
from keras_tqdm import TQDMNotebookCallback
from ipywidgets import interact

from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()
import mpld3
mpld3.enable_notebook()

get_ipython().magic('matplotlib inline')

print("All dependencies imported!! TF: {} ; Keras :{}".format(tf.__version__,keras.__version__))

get_ipython().system('(date +%d\\ %B\\ %G)')


# In[17]:

def get_indices(es):
    indices = es.indices.get_aliases().keys()
    # len(indices)
    rucio= (index for index in indices if('atlas_rucio-events' in index))
    events = []
    for event in rucio :
        events.append(event)
    print('total event indices:',len(events),'\n')
    #print(events)
    indices_dict = {}
    for event in events:
        i = es.count(index=event)
        indices_dict[event] = i['count']
    # print('total data points:',sum(int(list(indices_dict.values()))))
    print(indices_dict)
    return indices, indices_dict
    

# saves data to a dataframe
def extract_data(index, query, scan_size, scan_step):
    resp = es.search(
    index = index,
    scroll = '20m',
    body = query,
    size = scan_step)

    sid = resp['_scroll_id']
    scroll_size = resp['hits']['total']
    results=[]
    for hit in resp['hits']['hits']:
        results.append(hit['_source']['payload'])
    #plot_data_stats(results)
    steps = int((scan_size-scan_step)/ scan_step)

    # Start scrolling

    for i in range(steps):
        if i%10==0:
            print("Scrolling index : {} ; step : {} ...\n ".format(index,i))
        resp = es.scroll(scroll_id = sid, scroll = '20m')
        # Update the scroll ID
        sid = resp['_scroll_id']
        # Get the number of results that we returned in the last scroll
        scroll_size = len(resp['hits']['hits'])
        if i%10==0:
            print("scroll size: " + str(scroll_size))
        for hit in resp['hits']['hits']:
            results.append(hit['_source']['payload'])
    
    print("\n Done Scrolling through {} !! \n".format(index))
    results = pd.DataFrame(results)
    print(results.info(), '\n')
    return results

def get_indices_data(indices_list, query, scan_size, scan_step):
    data = pd.DataFrame()
    for index in indices_list:
        tmp_data = extract_data(index, query, scan_size, scan_step)
        plot_data_stats(tmp_data)
        data = data.append(tmp_data, ignore_index=True)
    print(data.info())
    plot_data_stats(data)
    print("\n------------------- ALL DONE !!!!! -----------------------")
    return data


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


# # get data

# In[7]:

rucio_data = pd.read_csv('may.csv')


# In[8]:

# rucio_data.info()


# In[11]:

rucio_data = rucio_data[0:20000]


# In[12]:

rucio_data.info()


# In[18]:

rucio_data = preprocess_data(rucio_data)
rucio_data.head(10)


# In[20]:

durations = rucio_data['duration']
rucio_data = rucio_data.drop(['duration'], axis=1)


# In[23]:

durations.shape


# In[30]:

train_fraction = 0.8
data_size = rucio_data.shape[0]
split_index = int(train_fraction * data_size)
# print(split_index)
trainX,trainY,testX,testY=rucio_data[:split_index],durations[:split_index],rucio_data[split_index:],durations[split_index:]


# In[46]:

print(trainX.shape,'\t', trainY.shape)


# In[70]:

import keras.callbacks as cb

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


# In[ ]:

def init_model():
    start_time= time.time()
    print('Compiling model ...')
    
    model = Sequential()
    model.add(Dense(512, input_dim=10,init='normal', activation='tanh'))

    model.add(Dense(512,init='normal', activation='linear'))
    model.add(LeakyReLU(alpha=0.001))

    model.add(Dense(512, init='normal', activation='tanh'))
    model.add(Dense(256, init='normal', activation='relu'))  # relu
    # model.add(LeakyReLU(alpha=0.001))

    model.add(Dense(256, init='normal', activation='linear'))
    model.add(Dense(256, init='normal', activation = 'relu'))  # extra
    model.add(Dense(128, init='normal', activation='relu'))    # relu
    model.add(Dense(1))
    
    adam = Adam(lr=0.001, decay= 0.0002)
#     adam = Adam(lr=0.01, decay = 0.0001)
    model.compile(loss='mean_squared_error', optimizer=adam, metric=['mean_squared_error'])
    print('Model compield in {0} seconds'.format(time.time() - start_time))
    return model


# In[101]:

def run_network(model=None, epochs=20, batch=256):
    try:
        start_time = time.time()

        if model is None:
            model = init_model()

        history = LossHistory()

        print('Training model...')
        training = model.fit(trainX.as_matrix(), trainY.as_matrix(), epochs=epochs, batch_size=batch,
                             validation_split=0.1, callbacks=[history], verbose=1)

        print("Training duration : {0}".format(time.time() - start_time))
        score = model.evaluate(trainX.as_matrix(), trainY.as_matrix(), verbose=0)

        print("Network's training score [MSE]: {0}".format(score))
        print("Training finished !!!!!!")
        return training, model, history.losses
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        return model, history.losses


# In[102]:

def plot_losses(losses):
    sns.set_context('poster')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    fig.show()
    print(len(losses))


# In[104]:

training, model, losses = run_network(epochs= 20, batch=2048)
plot_losses(losses)


# In[105]:

sns.set_context('poster')

plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')
plt.show()


# In[106]:

score = model.evaluate(testX.as_matrix(), testY.as_matrix(), verbose=2)


# In[107]:

print('Results: MSE - {:.6f}  ;  seconds = {:.3f}'.format(score, np.sqrt(score)))


# In[108]:

predictions=model.predict((testX[:1000]).as_matrix())


# In[109]:

sns.set_context('poster')

# plotting a portion of the test values against the predicted output

plt.plot(predictions[:300,0],'y' )
plt.plot(testY[:300].as_matrix(), 'g')

plt.show()


# In[110]:



p = figure(plot_width=950, plot_height=500)
p.line(x=list(range(len(predictions[:200,0]))), y=predictions[:200,0], color='#008000')
p.line(x=list(range(len(predictions[:200,0]))), y=testY[:200])
show(p)

# q = figure(plot_width=950, plot_height=500)
# q.line(x=list(range(len(predictions[:200,0]))), y=testY[:200])
# show(q)


# In[ ]:



