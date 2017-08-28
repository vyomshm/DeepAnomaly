# DeepAnomaly

Main Code repository for GSOC-2017 project under CERN-HSF (https://summerofcode.withgoogle.com/projects/#4916590398144512)

You can read my final repost [here](https://medium.com/towards-data-science/gsoc-2017-working-on-anomaly-detection-at-cern-hsf-49766ba6a812).

# Overview

This project is aimed towards building a framework which monitors incoming ATLAS Computing Operations data for anomalies and then autonomously, acts on this information by either solving the problem or by proposing the best solution. The solution proposed for this would require two components at it's heart -

* One a recurrent network to actually predict anomalies in the incoming real-time data. These predicted anomalies will then be analysed to come up with a list of potential solutions to the problem

* Second an algorithm to rank the proposed solutions based on a feedback. This algorithm will provide increasingly efficient outputs over time, thus improving the overall efficiency of the framework.


# Project Status

### (as of 29th August 2017)

* I succeded in laying a very basic foundation for further work on this topic. Crucial tasks like that of data extraction and preprocessing for a deep learning model have already been taken care of.
* I tried training multiple architectures of both simple feedforward networks as well as advanced Long Short Tern Memory Models. As expected Recurrent architectures were much better at estimating the duration of a file transfer tahn the simple feedforward neural nets.
* Some best performing architectures of LSTM networks (and feedforward networks) have been saved for reference. The best saved model provides predictions, of which, > 90% are within the error threshold of just 3 minutes. 
* I used a very basic error threshold of 600 seconds to tag anomalies. This is not a very good approach as it triggers roughly 5-10 % of all events which is too large. I believe this can be solved using the statistics from the PerfSonar data.
* I wasn't able to implement a way to do the ML processing in real-time i.e. at the same time as the events are themselves being indexed into the ES instances.


![These figures were genrated from model predictions on a set of 200,000 consecutive transfers](https://github.com/vyomshm/DeepAnomaly/blob/master/automate.png) 



# Project Complexity and challenges

* All through the summer I remained limited to only dealing wth the "Transfer-done" events under Rucio DDM data. The raw data present in the two main ES instances is incredibly varied and messy. The sheer variety of the number of variables and the innate complexity within this timeseries makes the identification of anomalies a herculean task. 
* The sheer amount of data, while always a good news for a Deep learning engineeer likeme, makes it really to implement any sort of realtime processing on it.
* Training my LSTM networks which had a sequence size of a 100 transfers was very difficult. I was fortunate enough to have GPU access for this summer which allowed me to train these networks on millions of transfer events at a time. This was and can be a very time intensive process. I never trained any network on the entire bulk of available data. At any given time, data from the past 30-31 days is available. Assuming an average of 1.2 million transfers per index, this value comes at about 36 miilion transfers. As per the sequence size of 100, this means that the total number of timesteps created for training such a LSTM would amount to roughly 36 billion (each having 9 fields).

# ORGANIZATION - CERN-HSF

# MENTORS -

* [Mario Lassnig](mario.lassnig@cern.ch)
* [Alessandro Di Girolamo](alessandro.di.girolamo@cern.ch)

## STUDENT DETAILS

Vyom Sharma - vyomshm@gmail.com

## CATEGORY - atlas

#### ~Fin~

![](https://github.com/vyomshm/DeepAnomaly/blob/master/automate.png)

