Table of Contents

Installation

Project Motivation

File Descriptions

Instructions

Results

Licensing, Authors, and Acknowledgements

Installation

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.  Organization of the folder should not be changed as instructions depend on certain filepaths.

Project Motivation

For this project, I wanted to train a machine learning classifier model to categorize disaster response messages in a way that would make them easier to respond to.  This process involves several steps.

1. Cleaning and transforming the raw disaster response data.
2. Training a classifier model on the processed data.
3. Displaying resutls using a web app and categorizing user input data.

File Descriptions

process_data.py in the data folder cleans up the disaster response message and categories data and stores the data in a SQL db file in the same location.

train_classifier.py in the models folder trains a random forest classifier based on a tokenized version of the disaster response data.  The model is saved as a pickle file in the same location

run.py in the app folder runs a web app displaying the classifier results and allows you to test the classifier on your own input text.

Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

Results

The findings of the project can be seen by running the 

Licensing, Authors, Acknowledgements

Prelabeled disaster response data was made available from Figure8 as part of Udacity's Disaster Respone Pipeline Project. Feel free to use the code here as you would like!
