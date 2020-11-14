# w2v.py
# 這個 block 是用來訓練 word to vector 的 word embedding
# 注意！這個 block 在訓練 word to vector 時是用 cpu，可能要花到 10 分鐘以上
import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from utils import load_training_data, load_testing_data
from gensim.models import word2vec

def train_word2vec(x):
    # 訓練 word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model

def main(data_folder_path, model_folder_path):
    if not isinstance(data_folder_path, Path):
        data_folder_path = Path(data_folder_path)

    if not isinstance(model_folder_path, Path):
        model_folder_path = Path(model_folder_path)
        
    training_label_path = data_folder_path.joinpath('training_label.txt')
    training_nolabel_path = data_folder_path.joinpath('training_nolabel.txt')
    testing_path = data_folder_path.joinpath('testing_data.txt')
    
    print("loading training data ...")
    train_x, y = load_training_data(str(training_label_path))
    train_x_no_label = load_training_data(str(training_nolabel_path))

    print("loading testing data ...")
    test_x = load_testing_data(str(testing_path))

    # model = train_word2vec(train_x[:100])
    model = train_word2vec(train_x + train_x_no_label + test_x)
    
    print("saving model ...")
    model.save(str(model_folder_path.joinpath('w2v_all.model')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train word2vec model with given input")
    parser.add_argument("data_folder_path", type = str, default = './', help = "The path to the dataset folder")
    parser.add_argument("--model_folder", type = str, default = '', 
                        help = "The path to the folder where the trained model will be stored")
    args = parser.parse_args()

    # Get data folder path
    data_folder_path = Path(args.dataset_folder_path)
    if not data_folder_path.is_absolute():
        data_folder_path = Path.cwd().joinpath(data_folder_path)

    # Get the folder path for storing trained model
    if args.model_folder == '':
        model_folder_path = data_folder_path
    else:
        model_folder_path = Path(args.model_folder)

    if not model_folder_path.is_absolute():
        model_folder_path = Path.cwd().joinpath(model_folder_path)

    print(f"data_folder_path: {data_folder_path}")
    print(f"model_folder_path: {model_folder_path}")

    main(data_folder_path, model_folder_path)