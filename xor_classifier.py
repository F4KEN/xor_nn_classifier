import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix, accuracy_score
import datetime
import os

# Input data (XOR input combinations)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output labels (XOR results)
y = np.array([0, 1, 1, 0], dtype=np.float32)

def build_model(input_dim=2,
                hidden_layers=[8, 4],
                activation='tanh',
                optimizer='adam',
                learning_rate=0.01):
    """
    Builds a feedforward neural network model.
    """
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_shape=(input_dim,), activation=activation))

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))

    model.add(Dense(1, activation='sigmoid'))

    if optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X, y, epochs=1000, batch_size=1, use_tensorboard=False):
    """
    Trains the model with optional TensorBoard support and early stopping.
    """
    callbacks = []
    early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    callbacks.append(early_stop)

    if use_tensorboard:
        log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

    history = model.fit(X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=callbacks)
    return history


def plot_history(history):
    """
    Plots the accuracy and loss curves.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(history.history['accuracy'])
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')

    ax[1].plot(history.history['loss'])
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')

    plt.tight_layout()
    plt.show()


def evaluate_model(model, X, y):
    """
    Evaluates the model and prints predictions and accuracy.
    """
    predictions = model.predict(X)
    predictions_rounded = np.round(predictions).astype(int)

    print("\nRaw Predictions:")
    print(predictions)
    print("\nRounded Predictions:")
    print(predictions_rounded)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions_rounded))

    acc = accuracy_score(y, predictions_rounded)
    print("Accuracy Score:", acc)


def save_model(model, filename='xor_model.h5'):
    """
    Saves the trained model to disk.
    """
    model.save(filename)
    print(f"\nModel saved to: {filename}")


def load_existing_model(filename='xor_model.h5'):
    """
    Loads a saved model from disk.
    """
    model = load_model(filename)
    print(f"\nModel loaded from: {filename}")
    return model


# ================================
# MAIN SCRIPT
# ================================

model = build_model(
    input_dim=2,
    hidden_layers=[8, 4],
    activation='tanh',
    optimizer='adam',
    learning_rate=0.01
)

history = train_model(
    model,
    X,
    y,
    epochs=1000,
    batch_size=1,
    use_tensorboard=False
)

plot_history(history)
evaluate_model(model, X, y)
save_model(model, filename='xor_model.h5')

loaded_model = load_existing_model('xor_model.h5')
evaluate_model(loaded_model, X, y)
