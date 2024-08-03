import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterDatabase', engine)

# load model and necessary support
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
import nltk
nltk.download(['stopwords','punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
# Load pkl file
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    Y = df.drop(columns=['id','message','original','genre', 'related'])
    message_types = Y.columns
    message_proportions = Y.sum().div(Y.values.sum())
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=message_types,
                    y=message_proportions
                )
            ],

            'layout': {
                'title': 'Distribution of Message Types',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Message Type"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
