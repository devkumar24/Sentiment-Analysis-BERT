# In the model.py file, here we will be creating a model
# that will be used to create for our sentimental analysis 
# on different types of dataset,
# we will training our model on IMDB dataset.


# We are using pre-trained BERT model for our analysis
# requirements : We first need to install a package i.e., transformers
# STEPS TO INSTALL : 
# Open The Terminal
# Use command `pip install transformers`
# It will install all the dependencies that will be required to install transformers


# import necessary methods/function/classes that we need in our model
from tensorflow.keras import metrics
from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer
from transformers import InputExample, InputFeatures
from utils import *

import tensorflow as tf

# Helper Class that is used to show text in different colors,
# this class has no effect on the main class i.e., Model Class
class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# Tis is the main class that is used to help to create a model for out task
# this class is used to implement sentimental analysis using BERT model.
class Model:
    # constructor that is used to take as user input which is necessary for our model
    def __init__(self, optimizer, loss, metrics,epochs, tokenizer : bool, show_summary : bool) -> None:
        self.show_summary = show_summary
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.tokenizer = tokenizer

    #---------------------------------------------------------------------------------------------------------------------
    def define_model(self):

        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = model
        if self.tokenizer and self.show_summary:
            self.tokenizer = tokenizer
            self.model.summary()
            
            return (self.model, self.tokenizer)

        elif self.tokenizer:
            self.tokenizer = tokenizer
            return (self.model, self.tokenizer)

        else:
            return self.model
    #---------------------------------------------------------------------------------------------------------------------
    def compile(self, **kwargs):
        """
        This function takes an input as model, if model is not given by user, then it will 
        automatically takes as an pre-defined and created model by using define_model function
        
        Other arguements takes as an arguements are: 
                    1. optimizer: It is better to use Adam optimizer
                    2. loss: use MSE
                    3. metrics: accuracy

        
        It will compile our model, as per given input, is we don't enter the input,
        it will automatically takes input by its own.

        Return : compiled model
        """
        keys = list(kwargs.keys())
        if "model" in keys:
            if "optimzer" in keys:
                if "loss" in keys:
                    if "metrics" in keys:
                        model = kwargs['model']
                        optimizer = kwargs['optimizer']
                        loss = kwargs['loss']
                        metrics = kwargs['metrics']
                        model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

        else:
            model = self.model
            model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)
            self.model = model

    #---------------------------------------------------------------------------------------------------------------------
    def fit(self, data, **kwargs):
        keys = list(kwargs.keys())

        if "model" in keys and "epochs" in keys:
            model = kwargs['model']
            epochs = kwargs['epochs']
            model.fit(data, epochs = epochs)
            self.model = model

        else:
            if "epochs" in keys:
                model = self.model
                epochs = kwargs['epochs']
                model.fit(data,epochs = epochs)
                self.model = model
            else:
                model = self.model
                model.fit(data, epochs = self.epochs)
                self.model = model
    
    #---------------------------------------------------------------------------------------------------------------------
    def save_model(self, fname : str = "", **kwargs):
        """
        It is used to save the trained model in the desired location.
        """
        keys = list(kwargs.keys())
        if "model" in keys:
            model = kwargs['model']
            model.save_pretrained(fname) #save the model with the specifies path if model parameter is present.
        else:
            model = self.model # make a model variable and store the predined model in  the class.
            model.save_pretrained(fname) # save the model with specific path






def predict(text, model, tokenizer, return_pred : bool, show_pred : bool):
    """
        It has made to predict the results of our trained model, 
        it will takes an input sentence in the form of list, it mean it can n number of sentences as input 
    """
    category = {
        'positive' : 1,
        'negative' : 0
        }

    tf_batch = tokenizer(text, max_length=32, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    labels = ['Negative','Positive']
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    pred = list()
    if return_pred:
        for i in range(len(text)):
            pred.append(label[i])

        return pred
    if show_pred:
        for i in range(len(text)):
            print("The text is : {} {} {} \nThe prediction of the given text is : {} {}".format(
                    Color.BOLD,
                    text[i],
                    Color.BOLD,
                    Color.RED,
                    category[label[i]]
                ))

#---------------------------------------------------------------------------------------------------------------------
def accuracy(model,x_data, y_data, batch_size = 32):
    """
    It will predict the accuracy of the provided data.
    """
    df = create_dataFrame(x_data, y_data)
    data = list(df['Body_of_Review'])
    acc_ = list()
    y_data = np.array(y_data)
    for i in range(int(df.shape[0]/batch_size - 1)):
        pred_ = predict(data[i*batch_size : (i+1)*batch_size], model, return_pred = True, show_pred = False)
        acc__ = 100 * ( sum(y_data[i*batch_size : (i+1)*batch_size] == np.array(pred_) ) / batch_size )
        acc_.append(acc__)
        # print("Accuracy of Batch {} : {}".format(i+1, acc__))
    acc = np.mean(acc_)
    return acc