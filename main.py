# main.py
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

import w2v
from utils import *
from preprocess import Preprocess
from model import buildModel, testing, BiLstmTuner

AUTOTUNE = tf.data.experimental.AUTOTUNE

def main():
    # Setup paths
    path_prefix = Path.cwd()
    train_with_label = os.path.join(path_prefix, 'data/training_label.txt')
    train_no_label = os.path.join(path_prefix, 'data/training_nolabel.txt')
    testing_data = os.path.join(path_prefix, 'data/testing_data.txt')
    w2v_path = path_prefix.joinpath('model/w2v_all.model') 

    # Configuration
    sen_len = 20
    batch_size = 128

    # Preprocess dataset
    ## Read 'training_label.txt' and 'training_nolabel.txt'
    print("loading training data ...")
    X_train_lable, y_train_lable = load_training_data(train_with_label)
    X_train, X_val, y_train, y_val = train_test_split(X_train_lable, 
                                                      y_train_lable, 
                                                      test_size = 0.1)

    train_x_no_label = load_training_data(train_no_label)

    print(f"Positive rate in training dataset: {np.sum(y_train) / len(y_train)}")
    print(f"Positive rate in validation dataset: {np.sum(y_val) / len(y_val)}")

    ## Build the preprocessor
    preprocessor = Preprocess(sen_len, w2v_path = str(w2v_path))
    embedding = preprocessor.make_embedding(load = True)
    X_train_idx = preprocessor.sentences_word2idx(X_train)
    X_val_idx = preprocessor.sentences_word2idx(X_val)

    print(f"Pretrained embedding matrix shape: {embedding.shape}")

    ## Preprocess training and validation datasets
    X_train_idx_dataset = tf.data.Dataset.from_tensor_slices(X_train_idx)
    y_train_dataset = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((X_train_idx_dataset, y_train_dataset))

    X_val_idx_dataset = tf.data.Dataset.from_tensor_slices(X_val_idx)
    y_val_dataset = tf.data.Dataset.from_tensor_slices(y_val)
    val_dataset = tf.data.Dataset.zip((X_val_idx_dataset, y_val_dataset))

    train_dataset = train_dataset.batch(batch_size)
    val_dataset   = val_dataset.batch(batch_size)

    train_dataset = train_dataset.cache().prefetch(AUTOTUNE)
    val_dataset   = val_dataset.cache().prefetch(AUTOTUNE)

    # Train a bidirectional LSTM model
    train_embedding = False # fix embedding during training

    ## Build the model
    hidden_dim1 = 64
    hidden_dim2 = 64
    dp_rate = 0.5
    lr = 0.001
    epochs = 1

    model = buildModel(embedding, train_embedding, sen_len, hidden_dim1, hidden_dim2, dp_rate, lr)

    model.summary()

    ## Train the model
    checkpoint_filepath = os.path.join(path_prefix, 'ckpt/')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True)

    history = model.fit(train_dataset, 
                        validation_data=val_dataset, 
                        epochs = epochs, 
                        callbacks=[model_checkpoint_callback])

    # Testing
    ## Preprocess test dataset
    print("loading testing data ...")
    X_test = load_testing_data(testing_data)
    X_test_idx = preprocessor.sentences_word2idx(X_test)

    test_dataset = tf.data.Dataset.from_tensor_slices(X_test_idx)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.cache().prefetch(AUTOTUNE)

    ## Load the best model
    print('\nload model ...')
    best_model = tf.keras.models.load_model(checkpoint_filepath)

    ## Predict
    outputs = testing(best_model, test_dataset)

    # Write the result to a CSV file
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(X_test))],"label":outputs})
    print("save csv ...")
    tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
    print("Finish Predicting")


if __name__ == '__main__':
    main()