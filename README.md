# Initializing Convolutional Filters with Semantic Features for Text Classification
Source code of the EMNLP 2017 paper *Initializing Convolutional Filters with Semantic Features for Text Classification*.


**** The purpose of the project
The project implements the weight initialization method in the submitted paper.
Recently, CNN [1] is proven to be effective for sentence classification.  
Our method further improves the CNN baseline significantly with little additional costs.
One can easily obtain the state-of-the-art or competitive results via running demo.sh in our project. 


**** Requirements and preprocessing
Code is written in Python (2.7) and requires Tensorflow (>=1.0.0, because of the order of artf.concat).
The data preprocessing and hyper-parameter setting strictly follow the implementation in https://github.com/yoonkim/CNN_sentence [1], whose preprocessing code is reused in our project.
Pre-trained embedding 'GoogleNews-vectors-negative300.bin' is available at https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing.


**** How to use
To process the raw data:
>> python process_data.py vectors_path

This will create a pickle object called 'mr.p' in the same folder, which contains the dataset in the right format.

To obtain the pre-trained features for weight initialization
>> python initialize_filter.py [--ngram NGRAM] mr.p

This will create a pickle object called 'weights_NGRAM.pkl', containing parameters 
for initializing weights in the CNN filters.

To train the model
>> python cnn.py mr.p weights_NGRAM.pkl

This will train and test the model.


**** Reference
Yoon Kim. 2014. Convolutional neural networks for sentence classification. EMNLP
