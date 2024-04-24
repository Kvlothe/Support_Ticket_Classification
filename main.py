import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import string
import re
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


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


def lemmatize_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens if token.isalpha()]  # Lemmatize
    return ' '.join(lemmatized_tokens)


df = pd.read_csv('support_tickets.csv', names=['document', 'topic'])

# Stopwords in English
stop_words = set(stopwords.words('english'))

# Print out first 5 columns/rows to make sure DF was created correctly
print(df.head())

# Check for any missing values/rows
print("Missing Values")
missing_values_count = df.isna().sum()
missing_values_count = missing_values_count[missing_values_count > 0]
print(missing_values_count)

unique_labels = df['topic'].unique()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['topic'])
num_classes = np.unique(y_encoded).shape[0]
y_one_hot = to_categorical(y_encoded, num_classes=num_classes)

print("Correct output shape:", y_one_hot.shape)


# Get a count of the dataset
# print(df.count)

# Apply the contains_emoji function to each row in the 'document' column
df['contains_emoji'] = df['document'].apply(lambda x: contains_emoji(x))
print(df.head())
true_count = df['contains_emoji'].sum()
print(f"Number of rows with emojis: {true_count}")

# Apply the function clean_text to your dataframe
df['cleaned_text'] = df['document'].apply(clean_text)
df['lemmatized_text'] = df['document'].apply(lemmatize_text)

all_words = [word for tokens in df['lemmatized_text'] for word in tokens]
word_counts = Counter(all_words)

# Vocabulary size
vocab_size = len(word_counts)
print(f"Vocabulary size: {vocab_size}")

# Plot the distribution of lengths
sequence_lengths = [len(tokens) for tokens in df['lemmatized_text']]
plt.hist(sequence_lengths, bins=30)
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.show()

# Choosing a maximum sequence length
avg_length = np.mean(sequence_lengths)
std_length = np.std(sequence_lengths)
max_length = int(avg_length + 2 * std_length)
print(f"Suggested maximum sequence length: {max_length}")

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df['document'])

# Convert sentences to sequences of integers
sequences = tokenizer.texts_to_sequences(df['document'])

# Pad sequences to ensure uniform length
X_padded = pad_sequences(sequences, maxlen=max_length, padding='post')
# print(X_padded[:5])
clean_df = pd.DataFrame(X_padded)
clean_df.to_csv('preped_data.csv', index=False)


# y = df['topic']
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_one_hot, test_size=0.2, random_state=42)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor the model's validation loss
                               patience=3,  # Stop after 3 epochs if the validation loss hasn't improved
                               restore_best_weights=True)  # Restore model weights from the epoch with the best validation loss

model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, output_dim=50, input_length=max_length))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(unique_labels), activation='softmax'))

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# Define and compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Input shape:", X_padded.shape)
print("Output shape:", y_one_hot.shape)

# Summarize the model to check layer configurations
model.summary()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[early_stopping])


