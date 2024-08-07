import sys
import nltk
nltk.download(['stopwords','punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import pandas as pd

def load_data(database_filepath):
    """ 
    Args:
    database_filepath: str. The filepath for the SQL db file to be loaded

    Returns:
    X: array. Text strings for all messages in the database
    Y: array. Values corresponding to if messages corresponded to the list of categories
    category_names: array.  List of column titles for the categories data 
    """ 
    from sqlalchemy import create_engine
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    # Open the link to the database
    engine = create_engine('sqlite:///' + database_filepath)
    # Select all data from database file
    df = pd.read_sql('SELECT * FROM DisasterDatabase', engine)
    # Set X as the message values
    X = df['message'].values
    # Set Y as only the output
    Y = df.drop(columns=['id','message','original','genre'])
    # Pull column names from Y as categories
    category_names = Y.columns
    # Transform Y to values for processing later
    Y = Y.values
    return X, Y, category_names

def tokenize(text):
    """ Transform message data into usable token data
    Args:
    text: array. The list of message strings

    Returns:
    clean_tokens: array.  Messages transformed into a list of lemmatized, lowercase, individiual words
    """ 
    # Load necessary packages
    from nltk.tokenize import word_tokenize
    from nltk.stem.wordnet import WordNetLemmatizer
    # Transform text into word tokens
    tokens = word_tokenize(text)
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Clean up and lemmatize tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok) 
    # Remove stopwords from tokens
    words = [t for t in tokens if t not in stopwords.words("english")]    
    # Return clean tokens as output
    return(clean_tokens)

# Import packages for StartingVerbExtractor
from sklearn.base import BaseEstimator, TransformerMixin
# Define a class for the Starting verb extractor
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        """ Find if the start of each sentence is a verb
        Args:
        text: str. Message text from the disaster response database

        Returns:
        Returns True if starting word was a verb and False if not
        """ 
       
        # Transform text into tokenized sentences
        sentence_list = nltk.sent_tokenize(text)
        # Extract starting verbs
        for sentence in sentence_list:
            # Tag positions for words in sentences
            pos_tags = nltk.pos_tag(tokenize(sentence))
            # Find the first word and it's tag
            first_word, first_tag = pos_tags[0]
            # If the first word is a verb, return True
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """ Build and train a classifier model for relating messages and categories
    Args: none

    Returns:
    cv: Pipeline.  A pipeline for a multioutput random forrest classifier
    """ 
    # Load necessary packages
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    # Build the ML Pipleline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Define parameters for grid search
    parameters = {
        'clf__estimator__bootstrap': [True, False],
        #'clf__estimator__n_estimators': [50, 100, 200],
        #'clf__estimator__criterion': ['gini', 'entropy', 'log_loss']
    }
    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters)
    # Return the model
    return cv
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    """ Calculate model performance metrics for each category
    Args:
    model: Pipeline. A pipeline for a multioutput random forrest classifier trained on message and category training data
    X_test: array. Messages data reserved for testing
    Y_test: array. Category data reserved for testing
    category_names: array. List of column titles for the categories data 

    Function outputs F1, precision, and recall score metrics for the model and testing data for each category
    """ 
    # Load calculations for F1, precision, and recall
    from sklearn.metrics import f1_score, precision_score, recall_score
    import pandas as pd
    # Calculate predictions for y
    Y_pred = model.predict(X_test)
    # Transform data to usable dataframe format
    df_y_test = pd.DataFrame(Y_test)
    df_y_pred = pd.DataFrame(Y_pred)
    # Calculate accuracy, precision, and recall metrics
    for index, label in enumerate(category_names):
        # Calculate metrics for each catetegory
        accuracy = f1_score(df_y_test[index],df_y_pred[index], average='micro')
        precision = precision_score(df_y_test[index],df_y_pred[index], average='micro')
        recall = recall_score(df_y_test[index],df_y_pred[index], average='micro')
        # Print results for each metric
        print(label)
        print('F1 score = ', accuracy)
        print('Precision score = ', precision)
        print('Recall score = ', recall)
        print('-------------------------')


def save_model(model, model_filepath):
    """ Save the classifier model as a pickle file
    Args:
    model: Pipeline. A pipeline for a multioutput random forrest classifier trained on message and category training data
    model_filepath: str. The filepath for where the trained model is to be saved

    The function creates a pickle file of the model at the model_filepath location
    """ 
    # Import pickle
    import pickle
    # Export the model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Train and output the classifier model
    This function performs all steps to load disaster response data and train a classifier for categorizing disaster response messages
    """
    from sklearn.model_selection import train_test_split
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
