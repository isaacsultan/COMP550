import os
import pickle
import argparse

import pandas as pd
from gensim.models import (Word2Vec, KeyedVectors)
from gensim.models.fasttext import FastText

from util.params import Params

"""
    Possibly useful resources:
    https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    https://rare-technologies.com/word2vec-tutorial/
    https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/
"""


def __train__(sentences, model=None, EmbeddingModel=Word2Vec):
    """Sentence input should be in format: 
    [['first', 'sentence'], ['second', 'sentence'], ..., ['last', 'sentence']]
    """

    # initialize model
    if model is None:
        # New model
        model = EmbeddingModel(None, size=100, window=5, min_count=5, workers=4, iter=20)
        model.build_vocab(sentences)
    else:
        model.build_vocab(sentences, update=True)

    # Train
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    return model


def train(training_path="dumps/v1", save_to="dumps/v1/train_word2vec.model", word2vec=True):
    model = None
    # Load sentences by batch
    for i in range(1,11):

        print("Loading batch {}".format(i))
        with open(os.path.join(training_path, "train_{}.pkl".format(i)), "rb") as f:
            train_set = pickle.load(f)
        print("Done")

        print("Training {} on batch {}".format("Word2Vec" if word2vec else "FastText", i))
        EmbeddingModel = Word2Vec if word2vec else FastText
        df_train = pd.DataFrame(train_set)

        # Context
        model = __train__(df_train[0].values, model=model, EmbeddingModel=EmbeddingModel)
        # Hypothesis
        model = __train__(df_train[1].values, model=model, EmbeddingModel=EmbeddingModel)
        print("Done.")

    # Finally, save the model
    model.save(save_to)


if __name__ == "__main__":
    params = Params()
    we_filename = os.path.join(params.dump_folder, params.word_embed_name)

    train(params.dump_folder, we_filename, params.is_word2vec)
    print("All training sets have been trained. Result in {}".format(save_to))
