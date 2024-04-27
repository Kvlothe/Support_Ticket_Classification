import pandas as pd
# from tensorflow.keras.models import load_model
from model import model
# from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data, train_new_model


df = pd.read_csv('support_tickets.csv', names=['document', 'topic'])
# df, y, vocab, length, num_classes = preprocess_data(df)
# model(df, y, unique, vocab, length, num_classes)

# Preprocess data with different techniques
x_basic, y_basic, vocab, length, num_classes = preprocess_data(df, 2)
x_lem, y_lem, vocab_lem, length_lem, num_classes_lem = preprocess_data(df, 3)
x_stem, y_stem, vocab_stem, length_stem, num_classes_stem = preprocess_data(df, 4)

# Split data into training and testing
# x_train_basic, x_test_basic, y_train_basic, y_test_basic = train_test_split(x_basic, y_basic, test_size=0.2)
# x_train_lem, x_test_lem, y_train_lem, y_test_lem = train_test_split(x_lem, y_lem, test_size=0.2)
# x_train_stem, x_test_stem, y_train_stem, y_test_stem = train_test_split(x_stem, y_stem, test_size=0.2)

# Train models
model_basic = model(x_basic, y_basic, vocab, length, num_classes, 2)
model_lem = model(x_basic, y_basic, vocab, length, num_classes, 3)
model_stem = model(x_basic, y_basic, vocab, length, num_classes, 4)

# Evaluate models
print("Evaluating Basic Model:")
model_basic.evaluate(x_test_basic, y_test_basic)

print("Evaluating Lemmatized Model:")
model_lem.evaluate(x_test_lem, y_test_lem)

print("Evaluating Stemmed Model:")
model_stem.evaluate(x_test_stem, y_test_stem)

# print("What would you like to do: inputs are 1-4")
# print("1: Create or train a new model")
# print("2: Load basic model")
# print("3: Load lemmatized model")
# print("4: Load stemmed model")
# task = input("Enter your choice (1-4): ")
# def model_load(choice):
#     models = {
#         "2": 'basic_model.keras',
#         "3": 'lemm_model.keras',
#         "4": 'stem_model.keras'
#     }
#     if choice in models:
#         return load_model(models[choice])
#     else:
#         print("Invalid task number for loading models.")
#         return None
#
#
# if task == "1":
#     train_new_model(df)
# else:
#     model = model_load(task)
#     if model:
#         print("Model loaded successfully.")
