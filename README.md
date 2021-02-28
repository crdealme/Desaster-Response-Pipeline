# Disaster-Response-Pipeline

# Installation

This repository requires the following Python packages: pandas, numpy, re, os, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys, warnings.

# Project Overview 

The goal of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The data set comes from real messages that were sent during disaster events. The created machine learning pipeline aims to categorize these events so that you can send the messages to an appropriate disaster relief agency.

# File Description

process_data.py: This python excutuble code takes as its input csv files containing message data and message categories (labels), and then creates a SQL database
train_classifier.py: This code trains the ML model with the SQL data base
ETL Pipeline Preparation.ipynb: process_data.py development procces
ML Pipeline Preparation.ipynb: train_classifier.py. development procces
data: This folder contains sample messages and categories datasets in csv format.
app: cointains the run.py to iniate the web app.

# Instructions

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/ Or Go to http://localhost:3001/
