import pickle
import csv
import os

import pandas as pd
import numpy as np

from util.params import Params


def filter_glove(word_indices_path, filtered_output):

    print("Read files...")
    iterator = pd.read_csv('data/glove.42B.300d.txt', header=None, index_col=0, 
                            delim_whitespace=True, quoting=csv.QUOTE_NONE, dtype="str", chunksize=100000)

    with open(word_indices_path, 'rb') as f:
        word_indices = pickle.load(f)
    print("Done.")

    df = pd.DataFrame()

    words = set(word_indices.keys())

    total = 0
    in_glove = 0
    total_ubuntu = len(words)

    print("Iterating through chunks...")
    done = 0
    # Iterate chunk by chunk
    for i in iterator:
        total += i.shape[0]
        unique_toks = set(i.index.values)
        in_glove += len(unique_toks.intersection(words))

        remain = unique_toks - words
        df = df.append(i.drop(remain, axis=0))
        done += 1
        print("Batch {} done".format(done))
    print("Done.")

    # Print compression percentage
    filtered = df.shape[0]
    print("Kept {0:.4f}% of the rows".format((filtered/total) * 100))
    print("{0:.4f}% of tokens were in glove".format(in_glove/total_ubuntu))

    df.to_csv(filtered_output, sep=" ", header=False, index=True, quoting=csv.QUOTE_NONE)


def main():
    params = Params()
    indices_path = os.path.join(params.dump_folder, "word_indices.pkl")
    output_path = os.path.join(params.data_folder, "glove_filtered.txt")

    filter_glove(indices_path, output_path)

if __name__ == "__main__":
    main()