import load
import tokenizer
import pickle
import numpy as np
from collections import Counter
import pandas
import os
import argparse

from util.params import Params


tags = ["eou", "eot"]
word_counts_name = "word_counts.pkl"
word_indices_name = "word_indices.pkl"

_unk = "<UNK>"
_pad = "<PAD>"

def construct_indices_from_count(path):
    """Convert the dictionary of word counts into a dictionary of word indices"""
    word_counts_path = os.path.join(path, word_counts_name)
    word_indices_path = os.path.join(path, word_indices_name)

    with open(word_counts_path, "rb") as f:
        counts = pickle.load(f)

    vocab = list(counts.keys())
    # Account for padding and unknown words
    vocab = [_pad, _unk] + vocab
    word_indices = dict(zip(vocab, range(len(vocab))))

    with open(word_indices_path, "wb") as f:
        pickle.dump(word_indices, f)


def merge_back_test_array(context, true, distractors):
    res = []
    for i in range(len(context)):
        row = []
        row.append(context[i])
        row.append(true[i])
        for k in range(len(distractors)):
            row.append(distractors[k][i])
        
        res.append(row)

    return res

def merge_back_train_array(context, hypothesis, value):
    # Value is a numpy array, so use item() to get the value
    res = []
    for i in range(len(context)):
        row = []
        row.append(context[i])
        row.append(hypothesis[i])
        row.append(value[i])
        res.append(row)
    return res


def split_training_dataset(file, nb_splits, output_format_file):
    # Output format file is expected to be in the form "filename_{}.csv", where the brackets will be replaced by the split number
    train = load.load_csv(file)

    subtrains = np.split(train, nb_splits, 0)

    for i in range(len(subtrains)):
        df = pandas.DataFrame(subtrains[i])
        df.to_csv(output_format_file.format(i+1), header=["Context", "Utterance" , "Label"], index=False, encoding="utf-8")


def tokenize_dataset(file_in, is_training, dump_folder, file_out, add_tag, domain_tags):
    
    path_file_out = os.path.join(dump_folder, file_out)
    word_counts_path = os.path.join(dump_folder, word_counts_name)

    if os.path.isfile(path_file_out):
        return

    # Load dict from pickle
    if os.path.isfile(word_counts_path):
        with open(word_counts_path, "rb") as f:
            words = pickle.load(f)
    else:
        words = Counter()

    # Load all data, tokenize, fetch all unique words
    print("Loading file...")
    dataset = load.load_csv(file_in)
    print("Done.")

    print("Preprocess sentences...")
    col_range = range(2) if is_training else range(11)
    results = []

    for i in col_range:
        res = tokenizer.tokenize_all(dataset[:,i], verbose_at=1000, add_tag=add_tag, domain_tag=domain_tags)
        
        for sentence in res:
            words.update(sentence)

        results.append(res)
    print("Done.")

    print("Dumping dictionary of words and tokenized dataset...")
    # Dump word dictionary
    with open(word_counts_path, "wb") as f:
        pickle.dump(words, f)

    #Merge back in correct form
    if is_training:
        results = merge_back_train_array(results[0], results[1], dataset[:,2])
    else:
        results = merge_back_test_array(results[0], results[1], results[2:11])
        results = load.breakdown(results)
    with open(path_file_out, "wb") as f:
        pickle.dump(results, f)


def main():
    params = Params()

    print("Started version {}...".format(params.dump_folder))

    for i in range(1, 11):
        tokenize_dataset("data/ubuntu_train_{}.csv".format(i), True, params.dump_folder, "train_{}.pkl".format(i), add_tag, domain_tag)

    tokenize_dataset("data/ubuntu_valid.csv", False, params.dump_folder, "valid_expanded.pkl", params.add_tag, params.domain_tag)
    tokenize_dataset("data/ubuntu_test.csv", False, params.dump_folder, "test_expanded.pkl", params.add_tag, params.domain_tag)

    # Store word indices
    construct_indices_from_count(params.dump_folder)

    print("Done version {}.".format(params.dump_folder))


if __name__ == "__main__":
    main()








