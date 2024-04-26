import pandas as pd
from preprocessing import preprocess_data
from model import model

df = pd.read_csv('support_tickets.csv', names=['document', 'topic'])

df, y, unique, vocab, length, num_classes = preprocess_data(df)
model(df, y, unique, vocab, length, num_classes)
