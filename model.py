import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from tensorflow.keras.regularizers import l2


# Function for creating a model - Not in use currently
def create_model(vocab_size, max_length, num_classes):  # use num_classes instead of unique_labels for clarity
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=50, input_length=max_length))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Function for creating and training a NLP model based on variables defined from preprocessing.
# Splits the dataset into training and testing sets, defines conditions for early stopping and
# trains a sequential model with 9 layers 20 epochs that will be monitored by the value loss and stop the model early
# to prevent over fitting and restore the best weights. Gives a printout of the models summary, a confusion matrix and
# a classification report, saves the current model based on the model being used or trained.
def model(x_padded, y_one_hot, vocab_size, max_length, num_classes, task):

    x_train, x_test, y_train, y_test = train_test_split(x_padded, y_one_hot, test_size=0.2, random_state=42)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss',  # Monitor the model's validation loss
                                   patience=3,  # Stop after 3 epochs if the validation loss hasn't improved
                                   restore_best_weights=True)  # Restore model weights from the epoch with the best validation loss

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=50, input_length=max_length))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, callbacks=[early_stopping])

    # Setup the classifier with additional parameters
    # model = KerasClassifier(build_fn=lambda: create_model(vocab_size, max_length, num_classes),
    #                         epochs=20, batch_size=10, verbose=1)
    #
    # # Define the K-fold Cross Validator
    # kfold = StratifiedKFold(n_splits=5, shuffle=True)
    #
    # # Perform K-fold cross-validation
    # results = cross_val_score(model, x_padded, y_one_hot,
    #                           cv=kfold)  # Ensure y_encoded is your label encoded target variable

    # Predict the labels on test dataset
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    # Generate a classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes))

    # Summarize the model to check layer configurations
    model.summary()
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    # Evaluate models

    if task == 2:
        model.save('basic.keras')
        # print("Evaluating Basic Model:")
        # model.evaluate(x_test, y_test)
    if task == 3:
        model.save('lemm_model.keras')
        # print("Evaluating Lemmatized Model:")
        # model.evaluate(x_test, y_test)
    if task == 4:
        model.save('stem_model.keras')
        # print("Evaluating Stemmed Model:")
        # model.evaluate(x_test, y_test)


# Define the function to load a model based on user input
def model_load(task):
    if task == "2":
        return load_model('basic_model.keras')
    elif task == "3":
        return load_model('lemm_model.keras')
    elif task == "4":
        return load_model('stem_model.keras')
    else:
        print("Invalid task number for loading models.")
