import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

# from tensorflow.python.ops.control_flow_ops import group


from utils import *
from model import *
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification

# Path to the data
dataset = "A:\Projects\Sentimental Analysis Using Bert\Dataset\IMDB Dataset.csv"
model_path = "A:\Projects\Sentimental Analysis Using Bert\sentiment_analysis"
category = {
    'positive' : 1,
    'negative' : 0
    }

# global variables
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5,epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
epochs = 2
model = tf.saved_model.load(model_path)

# Load dataset
df = pd.read_csv(dataset)
labels = np.array(df['sentiment'])
for i in range(labels.shape[0]):
    df.iloc[i]['sentiment'] = category[labels[i]]

X = df['review']
y = df['sentiment']
# create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# commanand line parser
parser = argparse.ArgumentParser(prog='main', description="Sentimental Analysis Using bert")
parser.add_argument('--accuracy', help='Calculate the accuracy of train or test set.(train/test)', choices=['train', 'test'], type=str)
parser.add_argument('--predict', help='Predict the Output of entered sentence', type=str)
group = parser.add_mutually_exclusive_group()
group.add_argument('-a', action = "store_true", help = "For Accuracy")
group.add_argument('-p', action = "store_true", help = "For Predictions")
args = parser.parse_args()


# main function that is used to train and save the model
def main():
    data = create_dataFrame(X_train,y_train)
    columns = data.columns

    model = Model(optimizer = optimizer, loss=loss, metrics = metrics, epochs = epochs, tokenizer=True,show_summary=True)
    bert_model, tokenizer = model.define_model()
    examples = create_data_to_example(data = data, Data_column=columns[0], Label_column = columns[1])
    train_data = convert_examples_to_tf_dataset(list(examples),tokenizer)
    train_data = train_data.shuffle(100).batch(32).repeat(2)

    model.compile()
    model.fit(train_data)
    model.save_model(fname="sentimental_analysis")


# it display the results of the trained model
def show_result(sentence, prediction : bool, accuracy_ : bool, **kwargs):
    if prediction:
        predict(sentence, model, tokenizer, return_pred = False, show_pred = True)

    if accuracy_ :
        keys = list(kwargs.keys())
        if "X" in keys and "y" in keys:
            X_test = kwargs['X']
            y_test = kwargs['y']
            acc = accuracy(model, X_test, y_test, batch_size = 256)
            print("The Accuracy on data is : {}".format(acc))

# gives us the predictiono of the entered sentence
def prediction(sentence : str):
    pred = predict(sentence, model, tokenizer,return_pred = True, show_pred = False)
    print("The text is : {} {} {} \nThe prediction of the given text is : {} {}".format(
                    Color.BOLD,
                    sentence[i],
                    Color.BOLD,
                    Color.RED,
                    pred
                ))

# gives the accuracy of the train/test data
def accuracy_(data : str):
    if data == "train":
        acc = accuracy(model, X_train, y_train, batch_size = 512)
        print("Accuracy is :".format(acc))
    if data == "test":
        acc = accuracy(model, X_test, y_test, batch_size = 128)
        print("Accuracy is :".format(acc))


 

if __name__ == "__main__":
    if args.p:
        prediction(args.predict)
    if args.a:
        accuracy_(args.accuracy)    
    else : 
        raise "InputError"