"""

ETL Disaster Response Pipeline 

Sample Script Syntax:
> python process_data.py <dabase path> <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>

Sample Script Execution:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

Arguments Description:
    
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response_db.db)
    
"""

# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def main(db_path, messages_csv='messages.csv', categories_csv='categories.csv'):
    
    """ 
    
    ETL pipeline for disaster response.
    
    Attributes:
        db_path (string) path to save the clean data set in a sqlite database 
        messages_csv (string): file path to messages.csv data
        categories_csv (string): file path to categories.csv data
            
    """
        # import the messages dataset
        messages = pd.read_csv(messages_csv)
        
        # import the categories dataset
        categories = pd.read_csv(categories_csv)
        
        # merge datasets
        df = messages.merge(categories, how = 'inner', on = 'id')
        
        # Split categories into separate category columns
        categories = df['categories'].str.split(pat=';', expand=True)
        
        # select the first row of the categories dataframe
        row = categories.iloc[0, :]

        # extract a list of new column names for categories.
        category_colnames = row.apply(lambda x:x.split('-')[0]).values.tolist()
    
        # rename the columns of `categories`
        categories.columns = category_colnames
        
        for column in categories:
            # set each value to be the last character of the string
            categories[column] = categories[column].apply(lambda x: x.split('-')[1])

            # convert column from string to numeric
            categories[column] = categories[column].astype(int)
            
        # drop the original categories column from `df`
        df.drop(['categories'], axis=1, inplace=True)
        
        # concatenate the data sets
        df = pd.concat([df, categories], axis=1)
        
        # drop duplicates
        df.drop_duplicates(inplace=True)
        
        # save the clean dataset into an sqlite database
        engine = create_engine('sqlite:///'+db_path)
        df.to_sql('disastertab', engine, index=False)
        
if __name__ == '__main__':
    main()
