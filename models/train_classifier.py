'''
Train the Classifier - Disaster Response Pipeline Project

Execution sample :
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Parameters:
   1) SQLite database path that contains cleaned dataset
   2) Pickle file name to save the model
'''

#import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk
import os, sys
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# from numba import jit, cuda

def load_data(db_filepath):
    '''Load data from sqllite database

    :param db_filepath (str) :  Database filepath
    :return: X (DataFrame) : Features
             y (DataFrame) : Responses
    '''

    # load data from database
    conn = create_engine('sqlite:///{}'.format(db_filepath))

    # Print table names
    # print(conn.table_names())

    df = pd.read_sql_table("df", conn)

    # Select feature and response data
    X = df['message']
    y = df.iloc[:, 4:]
    categories = y.columns

    return X, y, categories

def tokenize(text):
    '''Read, clean and tokenize text data

    :param text (str) : Messages
    :return: clean_tokens (List) : Token List
    '''

    #     Normalize and Tokenize the text
    text = re.sub(r'[^A-Za-z0-9]', '', text.lower())
    tokens = word_tokenize(text)

    #     Remove stop words
    words = [w for w in tokens if w not in stopwords.words('english')]

    #     Lemmatize and Strip
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).strip()
        clean_tokens.append(clean_word)

    return clean_tokens

# function optimized to run on gpu
# @jit(target ="cuda")
def build_model(X_train, y_train):
    '''Build the model

    :return: model (): Optimal model
    '''

    # Build the model pipeline
    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ]
    )

    #print(pipeline.get_params().keys())

    # Configure the parameters for the grid search
    param = {
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2']
    }

    # Optimize model
    model = GridSearchCV(pipeline, param_grid=param)

    print('Training the model')
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test, category_names):
    '''Evaluate the model performance

    :param model: Trained classification model to be evaluated
    :param X_test (DataFrame): Testing Features
    :param y_test: Testing Labels
    :param category_names: List of Category names
    :return: Show the accuracy score
    '''

    y_pred = model.predict(X_test)
    print("Accuracy scores per category\n")
    for i in range(len(category_names)):
        print('Label : {}'.format(category_names[i]))
        print(classification_report(y_test.values[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    '''

    :param model (): Saves trained model as Pickle file
    :param model_filepath (): destination path to save .pkl file
    :return: None
    '''

    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    '''Build and Train classifier functionality for the Machine Learning Pipeline:

    => Extracting the data from SQLite db - DisasterResponse.db
    => Training Machine Learning model using 80% of the available dataset
    => Evaluating the model using test dataset
    => Saving the model as pickle file

    :return: None
    '''
    if len(sys.argv) == 3:
        db_filepath, model_filepath = sys.argv[1:]

        print('Loading data \n   DATABASE: {}'.format(db_filepath))
        X, Y, category_names = load_data(db_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)

        print('Building the model')
        model = build_model(X_train, y_train)

        # print('Training the model')
        # model.fit(X_train, y_train)

        print('Evaluating the model')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('This program accepts db_filepath of the disaster messages database, ' \
              'path to pickle file as first and second arguement' \
              'Ex : python train_classifier.py ../data/DisasterResponse.db cls.pkl'
              )

if __name__ == '__main__':
    main()