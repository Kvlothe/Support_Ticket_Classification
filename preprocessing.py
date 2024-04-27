# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import string
import re
from model import model

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# Function for training a new model Parameter is the created data frame
# Calls the preprocess_data function and receives the data frame, y after one hot encoding,
# vocabulary size and the number of classes or unique values in the topic group
# After preprocessing the function passes the variables returned by the preprocessing function and
# passes them to the model function for model creating then prints out the variables being used to create the model
def train_new_model(df):
    df, y, vocab, length, num_classes = preprocess_data(df)
    model(df, y, vocab, length, num_classes)
    print("Model would be trained here with vocab={}, length={}, num_classes={}".format(vocab, length, num_classes))


# Function for detecting non-english words and symbols
def contains_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric shapes extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental arrows-C
                               u"\U0001FA00-\U0001FA6F"  # Chess symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return bool(emoji_pattern.search(text))


# Function for cleaning the text, lower casing all text, removing punctuation and tokenizing the data,
# removing stopwords and non-alphabetic tokens
def clean_text(text):
    # Lowercase the text
    text = text.lower()

    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    return tokens


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# Function for lemmatizing the dataset, this considers the context and converts the word to its meaningful base for,
# which is called Lemma
def lemmatize_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens if token.isalpha()]
    return ' '.join(lemmatized_tokens)


# Function for stemming the dataset, a process that stems or removes last few characters from a word -
# This often leads to incorrect meaning and spellings - Faster process than Lemmatizing
def stem_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token.isalpha()]  # Stemming
    return ' '.join(stemmed_tokens)


# Stopwords in English
stop_words = set(stopwords.words('english'))


def preprocess_data(df, task):
    """
        Preprocesses the document data based on the specified task.

        Parameters:
            df (DataFrame): The input DataFrame containing the document and topic.
            task (int): Determines the type of text processing to apply:
                        1 - Basic,
                        3 - Lemmatization,
                        4 - Stemming.

        Returns:
            tuple: A tuple containing processed features and labels, vocabulary size, max sequence length,
            and number of classes.
        """
    # Print out first 5 columns/rows to make sure DF was created correctly
    # print(df.head())
    # Get a count of the dataset
    # print(df.count)

    print("Missing Values")
    missing_values_count = df.isna().sum()
    print(missing_values_count[missing_values_count > 0])

    df.dropna(inplace=True)  # Handling missing values by dropping

    # Encoding labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['topic'])
    y_one_hot = to_categorical(y_encoded)
    num_classes = y_one_hot.shape[1]

    print("Correct output shape:", y_one_hot.shape)

    # Apply the contains_emoji function to each row in the 'sentence' column
    df['contains_emoji'] = df['document'].apply(lambda x: contains_emoji(x))
    # print(df.head())
    true_count = df['contains_emoji'].sum()
    print(f"Number of rows with emojis: {true_count}")
    # Apply the function clean_text to your dataframe
    # df['cleaned_text'] = df['document'].apply(clean_text)

    # Text processing
    df['cleaned_text'] = df['document'].apply(clean_text)
    if task == 3:
        df['lemmatized_text'] = df['document'].apply(lemmatize_text)
    elif task == 4:
        df['stemmed_text'] = df['document'].apply(stem_text)

    all_words = [word for tokens in df['cleaned_text'] for word in tokens]
    word_counts = Counter(all_words)

    # Tokenization and padding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['document'])
    sequences = tokenizer.texts_to_sequences(df['document'])
    vocab_size = len(tokenizer.word_index) + 1
    max_length = np.max([len(seq) for seq in sequences])
    x_padded = pad_sequences(sequences, maxlen=max_length)

    return x_padded, y_one_hot, vocab_size, max_length, num_classes
