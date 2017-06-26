
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
import json

get_ipython().magic('matplotlib inline')


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
        plot_data_stats(tmp_data)
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

print(es.ping())
es.info()


# In[7]:

es.cluster.health()


# In[8]:

indices, indices_to_count = get_indices(es)


# # Total Data Points in the atlas_rucio-events-* index

# In[9]:

import time
count=es.count(index='atlas_rucio-events-*')
print('total documents at- {} : {}'.format(time.strftime("%c"), count['count']) )


# In[10]:

indices_to_count['atlas_rucio-events-2017.05.25']


# In[11]:

stats = es.field_stats(index='atlas_rucio-events-2016-06-30', fields=['payload.created_at'])
#atlas_rucio-events-2016-06-30
print(stats)


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
l.remove('atlas_rucio-events-2017.06.03')
len(l)
df_june = get_indices_data(indices_list=l, query=myquery,
                           scan_step= 10000,
                           scan_size=None,
                           indices_dict=indices_to_count)


# In[ ]:

# myquery={
#       'query': {
#         'bool': {
#           'must': {
#             'match': {'description': 'fix'}
#           },
#           'must_not': {
#             'term': {'files': 'test_elasticsearch'}
#           }
#         }
#       }
#     }

# # es.get_source(index='atlas_rucio-events-*', doc_type=, id= )     , filter_path='hits.hits._source'
# es.search(index='atlas_rucio-events-*', size=5)


# In[69]:

dFF=extract_data(index='atlas_rucio-events-2017.06.08',query=myquery,scan_size=300000, scan_step=10000)


# In[71]:

a=dFF.groupby(df.columns.tolist(),as_index=False).size()
a.unique()


# In[48]:

index= 'atlas_rucio-events-2017.05.17'
myquery = {
    "query": {
        "term": {
            '_type': 'transfer-done'
            }
        }
    }

# # scan_size=indices_dict['atlas_rucio-events-2017.05.17'],
# #                     scan_step=10000

# data = extract_data(index=index,
#                     query=myquery,
#                     scan_size=100000,
#                     scan_step=1000)


# In[54]:

rucio_data = df


# In[52]:

z=(df['previous-request-id'][0]) 


# In[60]:

# df.to_csv('may.csv')
# id = (df['previous-request-id']!=z)
# id
a=df['previous-request-id'].notnull()


# In[62]:

b=df['previous-request-id'][a]
b


# In[13]:

# from elasticsearch_dsl import Search
# from elasticsearch_dsl.query import MultiMatch, Match

# s = Search(using=es, index='atlas_rucio-events-2017.05.17', size=indices_dict['atlas_rucio-events-2017.05.17'])

# response = s.execute()
# print('Total %d hits found.' % response.hits.total)

# # indices_dict['atlas_rucio-events-2017.05.17']
# # print((response))
# # response.to_dict()
# print(response.success())
# print(response.took)
# print(response.hits.total)
# print(len(response.hits.hits))
# response.hits.hits[0]


# In[ ]:




# # Preprocessing data

# In[36]:

rucio_data = pd.read_csv('may.csv')


# In[37]:

rucio_data.head()


# In[38]:

rucio_data.info()


# In[39]:

duplicates = rucio_data.groupby(rucio_data.columns.tolist(),as_index=False).size()
duplicates


# In[56]:

names = (rucio_data['name'].unique())
print(names[0:5])
len(names)


# In[57]:

s_urls = rucio_data['src-url'].unique()
print(s_urls, '\n', len(s_urls))


# In[58]:

d_urls = rucio_data['dst-url'].unique()
print(d_urls, '\n', len(d_urls))


# In[40]:

# reasons = rucio_data['reason'].unique()
# print(reasons,'\n',len(reasons))

rucio_data['reason']


# In[41]:

src = rucio_data['src-rse'].unique()
print(src,'\n',len(src))


# In[42]:

dest = rucio_data['dst-rse'].unique()
print(dest,'\n',len(dest))


# In[43]:

scopes = rucio_data['scope'].unique()
print(scopes, '\n', len(scopes))


# In[44]:

adlers = rucio_data['checksum-adler'].unique()
len(adlers)


# In[45]:

tool_ids = rucio_data['tool-id'].unique()
tool_ids


# In[46]:

transfer_links = rucio_data['transfer-link'].unique()
print(transfer_links, len(transfer_links)) 


# In[47]:

t_end = rucio_data['transfer-endpoint'].unique()
t_end


# In[48]:

activities = rucio_data['activity'].unique()
activities


# In[49]:

# rucio_data['reason']
protocols = rucio_data['protocol'].unique()
protocols


# In[50]:

dst_types = rucio_data['dst-type'].unique()
dst_types


# In[51]:

src_types = rucio_data['src-type'].unique()
src_types


# # Preprocessing
# 

# 

# In[30]:

rucio_data = pd.read_csv('may.csv')


# In[25]:

rucio_data.columns


# In[26]:

from sklearn.preprocessing import LabelEncoder
from datetime import datetime,timedelta
# hdf = pd.HDFStore('datastore.h5')


# In[31]:

rucio_data=preprocess_data(rucio_data)


# In[32]:

rucio_data.head(1000)


# In[23]:

plot_data_stats(rucio_data)


# In[91]:




# In[97]:

rucio_data.head(30)


# # Basic HDF5 Datastore for storing extracted-preprocessed data

# In[96]:

hdf = pd.HDFStore('datastore.h5')
print(hdf)


# In[98]:

hdf.put('may/t1', rucio_data, format='table', data_columns=True)
rucio_data.to_csv('may_preprocessed.csv')


# In[99]:

print(hdf)


# In[ ]:

df = pd.read_hdf5

