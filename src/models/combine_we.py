import pickle
import csv
import os
from collections import Counter

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np

from util.params import Params


def compute_enhanced(words: set, word2vec: Word2Vec, pretrained: pd.DataFrame):
    """
    Algorithm to compute the enhanced word representation
    :param dictionary: Counter object which holds all unique tokens
    :param word2vec: Word2Vec object representing the trained word2vec model.
        Can be a subclass of Word2Vec, for example FastText
    :param pretrainedGlove: Dataframe containing the pretrained glove word embeddings

    :return : A dictionary of words as key as word embeddings as value
    """
    enhanced = dict()
    vec_size = pretrained.shape[1] + word2vec.vector_size

    # Store indexing. Pretrained comes first
    u_begin = 0
    u_end = pretrained.shape[1]
    v_begin = u_end
    v_end = v_begin + word2vec.vector_size

    for word in words:
        res = np.zeros((vec_size))
        if word in pretrained.index:
            res[u_begin:u_end] = pretrained.loc[word]

        if word in word2vec:
            res[v_begin:v_end] = word2vec[word]

        if np.count_nonzero(res) > 0:
            enhanced[word] = res
    return enhanced


def load_and_compute_enhanced(word_indices_path, word2vec_model_path, glove_filtered_path, enhanced_we_path):
    print("Computing enhanced word embeddings")

    with open(word_indices_path, "rb") as f:
        word_counts = pickle.load(f)
    words = set(word_counts.keys())

    w2vmodel = Word2Vec.load(word2vec_model_path)
    pretrained = pd.read_csv(glove_filtered_path, header=None, 
                index_col=0, delim_whitespace=True, quoting=csv.QUOTE_NONE)

    res = compute_enhanced(words, w2vmodel, pretrained)

    with open(enhanced_we_path, "wb") as f:
        pickle.dump(res, f)

    print("Done.")
    return res

if __name__ == "__main__":
    
    params = Params()
    word_indices_path = os.path.join(params.dump_folder, "word_indices.pkl")
    we_model = os.path.join(params.dump_folder, params.word_embed_name)
    glove_filtered_path = os.path.join(params.data_folder, "glove_filtered.txt")
    enhanced_we_path = os.path.join(params.dump_folder, "enhanced_we.pkl")

    if os.path.isfile(enhanced_we_path):
        print("Enhanced we already exists.")
        with open(enhanced_we_path, "rb") as f:
            enhanced = pickle.load(f)
    else:
        print("Computing enhanced...")
        enhanced = load_and_compute_enhanced(word_indices_path, we_model, glove_filtered_path, enhanced_we_path)
        print("Done.")

    

