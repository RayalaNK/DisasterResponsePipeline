'''
Preprocessing Disaster Response Pipeline Project

Execution sample :
> python process_data.py disaster_message.csv disaster_categories.csv DisasterResponse.db

Parameters:
    CSV file containing messages
    CSV file containing categories
    SQLite database
'''


import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    '''function to load the messages and categories datasets and merge the into a dataframe

    :param messages_filepath: messages filepath
    :param categories_filepath: categories filepath
    :return:
            df (DataFrame) : the merged dataframe
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id'])
    return df

def clean_data(df):
    '''

    :param df (DataFrame): Dataframe that contains merged dataset from messages and categories
    :return:
            df (DataFrame): Dataframe with duplicates removed and categories exploded
    '''

    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])

    # row = row.str[:-2]
    # category_colnames = row.unique()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
        categories[column] = categories[column].replace(2, 1)
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(subset=None, keep='first', inplace=False)
    return df

def save_data(df, db_filename):
    '''Function to export the cleaned dataset into a Sqlite database

    :param df (DataFrame): the dataset we want to export
    :param db_filename (DataFrame):
    :return:
            None
    '''

    engine = create_engine("sqlite:///{}".format(db_filename))
    df.to_sql('df', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, db_filepath = sys.argv[1:]

        print('Loading data...\n   Messages: {}\n    Categories: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    Database: {}'.format(db_filepath))
        save_data(df, db_filepath)

        print('Cleaned data saved to database file at {}!'.format(db_filepath))

    else:
        print('Provide the messages and categories filepaths as the first ' \
              'and second argument respectively, and the filepath to the database' \
              'to contain cleaned data as the third argument. ' \
              '\nExample: python process_data.py disaster_messages disaster_categories.csv ' \
              'DisasterResponse.db')

if __name__ == '__main__':
    main()