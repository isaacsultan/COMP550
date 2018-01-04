# Load files into in-memory datasets
import pickle
import os
import numpy as np
import pandas as pd
import random
import csv

glove_pretrained_file = "./data/glove.42B.300d.txt"

# We used the smaller file for testing
glove_smaller_file = "./data/globe.6B.300d.txt"

def load_csv(file):
    """Load a csv file with commas as delimiters, and no headers
    return value as a numpy array"""
    data = pd.read_csv(file, encoding="utf-8", keep_default_na=False, header=0)
    return data.values


def load_embedding_matrix(we: dict, word_indices_path: str, embedding_dimension: int):
    """
    Given word embeddings, compute the related embedding matrix
    """
    with open(word_indices_path, "rb") as f:
        word_indices = pickle.load(f)

    # Defaults to 0 if not found
    embedding = np.zeros((len(word_indices), embedding_dimension))

    for key, val in we.items():
        if key in word_indices:
            embedding[word_indices[key]] = val

    return embedding, word_indices


def breakdown(data):
    """Break down each row into context-utterance pairs.
    Each pair is labeled to indicate truth (1.0) vs distraction (0.0).
    Output is a native array with format : [context, utterance, label]"""
    output = []
    for row in data:
        context = row[0]
        ground_truth_utterance = row[1]
        output.append([list(context), ground_truth_utterance, 1.0])
        for i in range(2,11):
            output.append([list(context), row[i], 0.0])
    return output
