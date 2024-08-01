import sys


def load_data(messages_filepath, categories_filepath):
    """Load the messages and categories data.

    Args:
    messages_filepath: str. The filepath to the messages csv file
    categories_filepath: str. The filepath to the categories csv file

    Returns:
    df: dataframe.  A combineed dataframe of messages and categories data
    """ 
    # Import pandas for opening csv files
    import pandas as pd
    # Load files for messages and categories using given filepaths
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge messages and categories
    df = messages.merge(categories)
    return df

def clean_data(df):
    """ Transform the database data into a more useful format
    Args:
    df: dataframe. The combined messages and categories dataframe

    Returns:
    df: dataframe.  The messages and categories dataframe where categories data is converted into 1 or 0 values and columns are labeled appropriately 
    """ 
    import pandas as pd
    # Split category data into different columns
    categories = df['categories'].str.split(';',expand=True)
    # Define row for finding the column names
    row = categories.iloc[:1]
    # Category column are text minus the numerical values
    category_colnames = row.apply(lambda x: x.str[:-2])
    # Set column names using category_colnames
    categories.columns = category_colnames.values.tolist()[0]
    # Convert category values to 1 or 0
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = list(map(lambda x: x[-1], categories[column]))
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # Replace categories column with new, cleaned up values
    df=df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    # Remove duplicate values
    df = df.drop_duplicates()
    return df
    

def save_data(df, database_filename):
    """ Save the message and category data to a SQL database file
    Args:
    df: dataframe. The clean messages and categories dataframe
    database_filename: str.  The filename to which the database will be saved
    
    The function saves the df input dataframe as a SQL db file
    """ 
    # Import creat_engine so we can use SQL databases
    from sqlalchemy import create_engine
    # Save clean data to SQLite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterDatabase', engine, index=False)

def main():
    """ Run the steps for processing the disaster response data
    The "main()" function runs the data cleaning and saving process to transform the input data into a usable SQL db file.
    """ 
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
