import json
import logging
import os
import random
import stomp
import time
import pandas as pd
import numpy as np
from helpers import *


class EventListener(stomp.ConnectionListener):

    def on_error(self, headers, message):
        print 'error: %s' % message

    def on_message(self, headers, message):
   
        msg = json.loads(message)
        if msg['event_type'] == 'transfer-done':
		msgs.append(dict(msg['payload']))
		print msg	
        #print len(msg), type(msg), msg.keys(), msg






def setup():

    #there's a lot of messages, connect to one only to test ;-)
    #brokers = ['188.185.227.80', '188.184.82.150', '188.184.87.120', '188.185.227.37', '188.184.88.164']
    brokers = ['188.185.227.80', '188.184.82.150']
 
    conns = []
    for broker in brokers:
        print 'connecting to', broker,
        conns.append(stomp.Connection(host_and_ports=[(broker, 61013)], keepalive=True))
        print 'done'

    for conn in conns:
        print 'registering listener'
        conn.set_listener('', EventListener())

        print 'connecting to topic',
        conn.start()
        conn.connect('atlas', 'ddm', wait=True)
        conn.subscribe(destination='/topic/rucio.events', id='vyom-listener-%i' % random.randint(1,2**16))

        print 'done'




if __name__ == '__main__':
    msgs=[]
    setup()
    print 'waiting for events'
    time.sleep(10)
    save_live_data(msgs)
    inputs, outputs = load_live_data()
    model = load_lstm()

    predictions = model.predict(inputs)
    mae = np.subtract(np.reshape(outputs, [len(outputs), 1]), predictions)
    

    # making predictions


    
