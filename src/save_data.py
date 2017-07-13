
# coding: utf-8

# In[3]:

#  Dependencies
import requests
from elasticsearch import Elasticsearch,helpers
import elasticsearch_dsl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import uuid
import random
import time
import json



# In[4]:
# host_name = 'https://es-atlas.cern.ch/kibana'
# es-py __version__ is 2.3.0
# es_dsl __version__ is 5.3.0
# initialise elastic search with authorisation

es = Elasticsearch(['es-atlas.cern.ch:9203'],
                                 timeout=10000,
                                 use_ssl=True,
                                 verify_certs=True,
                                 ca_certs='/etc/pki/tls/certs/CERN-bundle.pem',
                                 http_auth='roatlas:la2Dbd5rtn3df!sx')


# In[5]:

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
    print('total hits in {} : {}'.format(index,scroll_size))
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

def get_indices_data(indices_list, query, scan_step, scan_size=None, indices_dict=None):
    
    hdf = pd.HDFStore('raw_datastore.h5')
#     data = pd.DataFrame()

    for index in indices_list:
        if scan_size == None:
            scan_size = indices_dict[index]
        tmp_data = extract_data(index, query, scan_size, scan_step)
        # plot_data_stats(tmp_data)
        file_name = (index+'.csv')
#         hdf.put(index, tmp_data, format='table', data_columns=True)
        tmp_data.to_csv(file_name)
        print('Saved {} data to disk !!'.format(index))
#         data = data.append(tmp_data, ignore_index=True)
#     print(data.info())
#     plot_data_stats(data)
    print("\n------------------- ALL DONE !!!!! -----------------------")
    return None
#     return data


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


# In[6]:

# print(es.ping())
# es.info()


# # In[7]:

# es.cluster.health()


# In[8]:

indices, indices_to_count = get_indices(es)


# # Total Data Points in the atlas_rucio-events-* index

# In[9]:

count=es.count(index='atlas_rucio-events-*')
print('total documents at- {} : {}'.format(time.strftime("%c"), count['count']) )


# In[10]:

indices_to_count['atlas_rucio-events-2017.05.25']


# # Extracting Data from 2-3 Months

# In[12]:

k = [index for index in indices if ('atlas_rucio-events-2017.05' in index)] #may-indices
l = [index for index in indices if ('atlas_rucio-events-2017.06' in index)] #june-indices
print(len(k), '\n', len(l))


# In[13]:

myquery = {
    "query": {
        "term": {
            '_type': 'transfer-done'
            }
        }
    }
# l.remove('atlas_rucio-events-2017.06.03')
# len(l)
df_june = get_indices_data(indices_list=l, query=myquery,
                           scan_step= 10000,
                           scan_size=None,
                           indices_dict=indices_to_count)