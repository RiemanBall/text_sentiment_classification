# main.py
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import w2v
from utils import *
from preprocess import Preprocess
from model import buildModel, testing, BiLstmTuner

AUTOTUNE = tf.data.experimental.AUTOTUNE

def main():
    parser = argparse.ArgumentParser(description = "Train a bidirectional LSTM model for text sentiment classification")
    parser.add_argument("--train_mode", type = str, default = 'preset_param', choices = ['preset_param', 'kerastuner'],
                        help = "Set the training mode (preset_param/kerastuner)")
    parser.add_argument("--batch_size", type = int, default = 64, help = "Batch size")
    parser.add_argument("--sen_len", type = int, default = 20, help = "Maximum length of a sentence")
    parser.add_argument("--lstm1", type = int, default = 32, help = "Hidden dimension of first LSTM")
    parser.add_argument("--lstm2", type = int, default = 32, help = "Hidden dimension of second LSTM")
    parser.add_argument("--dp_rate", type = float, default = 0.5, help = "Dropout rate (percentage of droping)")
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning rate")
    parser.add_argument("--epochs", type = int, default = 1, help = "epochs")
    
    parser.add_argument("--epochs_before_search", type = int, default = 1, help = "epochs_before_search")
    parser.add_argument("--epochs_after_search", type = int, default = 1, help = "epochs_after_search")
    parser.add_argument("--max_trials", type = int, default = 1, help = "max_trials for kerastuner")
    parser.add_argument("--executions_per_trial", type = int, default = 1, help = "executions_per_trial for kerastuner")

    args = parser.parse_args()

    # Setup paths
    path_prefix = Path.cwd()
    train_with_label = os.path.join(path_prefix, 'data/training_label.txt')
    train_no_label = os.path.join(path_prefix, 'data/training_nolabel.txt')
    testing_data = os.path.join(path_prefix, 'data/testing_data.txt')
    w2v_path = path_prefix.joinpath('model/w2v_all.model') 

    # Configuration
    batch_size = args.batch_size
    sen_len = args.sen_len

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

    ## Method1 - preset parameters
    if args.train_mode == 'preset_param':
        ### Build the model
        hidden_dim1 = args.lstm1
        hidden_dim2 = args.lstm2
        dp_rate = args.dp_rate
        lr = args.lr
        epochs = args.epochs

        model = buildModel(embedding, train_embedding, sen_len, hidden_dim1, hidden_dim2, dp_rate, lr)

        model.summary()

        ### Train the model
        checkpoint_filepath = os.path.join(path_prefix, 'ckpt/')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True)

        history = model.fit(train_dataset, 
                            validation_data=val_dataset, 
                            epochs = epochs, 
                            callbacks=[model_checkpoint_callback])

    elif args.train_mode == 'kerastuner':
        import IPython
        from kerastuner.tuners import RandomSearch

        class ClearTrainingOutput(tf.keras.callbacks.Callback):
            def on_train_end(*args, **kwargs):
                IPython.display.clear_output(wait = True)
            
        ### Build the model
        tuner = RandomSearch(
            BiLstmTuner(embedding, train_embedding, sen_len),
            objective='val_accuracy',
            max_trials = args.max_trials,
            executions_per_trial = args.executions_per_trial,
            directory = os.path.join(path_prefix, 'tuner_dir'),
            project_name = 'tsc')

        ### Train the model
        tuner.search(train_dataset,
                     epochs = args.epochs_before_search,
                     validation_data = val_dataset,
                     verbose = 1,
                     callbacks = [ClearTrainingOutput()],)
        

    # Load the best model
    print('\nload model ...')

    ## Method1
    if args.train_mode == 'preset_param':
        best_model = tf.keras.models.load_model(checkpoint_filepath)
    ## Method2
    elif args.train_mode == 'kerastuner':
        tuner.results_summary(num_trials = min(3, args.max_trials))
        best_model = tuner.get_best_models()[0]
        best_model.summary()

        # Train again with training set and validation set
        combined_dataset = train_dataset.concatenate(val_dataset)
        best_model.fit(combined_dataset,
                       epochs = args.epochs_after_search)

    # Testing
    ## Preprocess test dataset
    print("loading testing data ...")
    X_test = load_testing_data(testing_data)
    X_test_idx = preprocessor.sentences_word2idx(X_test)

    test_dataset = tf.data.Dataset.from_tensor_slices(X_test_idx)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.cache().prefetch(AUTOTUNE)

    ## Predict
    outputs = testing(best_model, test_dataset)

    # Write the result to a CSV file
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(X_test))],"label":outputs})
    print("save csv ...")
    tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
    print("Finish Predicting")


if __name__ == '__main__':
    main()