from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from tensorflow.keras.regularizers import l2


def model(x_padded, y_one_hot, unique_labels, vocab_size, max_length, num_classes):
    x_train, x_test, y_train, y_test = train_test_split(x_padded, y_one_hot, test_size=0.2, random_state=42)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss',  # Monitor the model's validation loss
                                   patience=3,  # Stop after 3 epochs if the validation loss hasn't improved
                                   restore_best_weights=True)  # Restore model weights from the epoch with the best validation loss

    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=50, input_length=max_length),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Flatten(),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, callbacks=[early_stopping])

    # Summarize the model to check layer configurations
    model.summary()

    # model.save('stem_model.keras')
    # model.save('lemm_model.keras')


