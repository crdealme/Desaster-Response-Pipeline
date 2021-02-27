"""

ML Disaster Response Pipeline 

Sample Script Syntax:
> python process_data.py <dabase path> <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>

Sample Script Execution:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

Arguments Description:
    
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response_db.db)
    
"""

import pickle
import re
import sys
import warnings
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine


def load_data(database_filepath):
    """Load cleaned data from database into dataframe.
    Args:
        database_filepath: String. It contains cleaned data table.
        table_name: String. It contains cleaned data.
    Returns:
       X: numpy.ndarray. Disaster messages.
       Y: numpy.ndarray. Disaster categories for each messages.
       category_name: list. Disaster category names.
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM all_messages', con=engine)

    category_names = df.columns[4:]

    X = df[['message']].values[:, 0]
    y = df[category_names].values

    return X, y

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    
    """Function that tokenizes a text.
    Args:
        text: String. A disaster message.
        lemmatizer: nltk.stem.Lemmatizer.
    Returns:
        list containing tokens.
    """
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def new_build_model():
    """Build model.
    Returns:
        pipeline: sklearn.model_selection.GridSearchCV. 
    """
    # Set pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
                learning_rate=0.3,
                n_estimators=200
            )
        ))
    ])

    # Set parameters for gird search
    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.3],
        'clf__estimator__n_estimators': [100, 200]
    }

    # Set grid search
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, scoring='f1_weighted', verbose=3)

    return cv


def evaluate_model(y_pred, Y_test, category_names):
    
    """Evaluate model
    Args:
        y_pred: numpy.ndarray. Predicted disaster message for each category.
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    
    # Predict categories of messages.
#     category_names = y.columns
    Y_pred = y_pred

    # Print accuracy, precision, recall and f1_score for each categories
    for i in range(0, len(category_names)):
        print(category_names[i])
        print("\tAccuracy: {:.4f}\t\t% Precision: {:.4f}\t\t% Recall: {:.4f}\t\t% F1_score: {:.4f}".format(
            accuracy_score(Y_test[:, i], Y_pred[:, i]),
            precision_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            recall_score(Y_test[:, i], Y_pred[:, i], average='weighted'),
            f1_score(Y_test[:, i], Y_pred[:, i], average='weighted')
        ))


def save_model(model, model_filepath):
    """Save model
    Args:
        model: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
        model_filepath: String. Trained model is saved as pickel into this file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    X, y = load_data(('all_messages.db'))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    category_names = df.iloc[:, 4:].columns
    evaluate_model(y_pred, y_test, category_names)
    
    save_model(model, 'pickle')

if __name__ == '__main__':
    main()
