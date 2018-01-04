import os
import pickle
import argparse

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def show_accuracy_graph(data):
    # Indices 1 and 2 refers to val. acc. and val. loss
    # Indices 3 and 4 refers to train. acc. and train. cost

    # Accuracy
    plt.plot(data[:,0], data[:,1], "g-", data[:,0], data[:, 3], "r-")
    #plt.title("Accuracy over training")
    plt.ylim([0.65, 0.95])
    plt.xlabel("Global step")
    #plt.ylabel("Accuracy")
    plt.legend(["Validation set", "Mini-training set"])
    plt.show()

    # Loss
    plt.plot(data[:,0], data[:,2], "g-", data[:,0], data[:, 4], "r-")
    #plt.title("Loss")
    plt.ylim([0, 0.008])
    plt.xlabel("Global step")
    #plt.ylabel("Loss")
    plt.legend(["Validation set", "Mini-training set"])
    plt.show()

    # Recall @ 1 and MRR
    plt.plot(data[:,0], data[:, 5], "b-", data[:,0], data[:, 6], "y-")
    #plt.ylim([])
    plt.xlabel("Global step")
    plt.legend(["Recall @ 1", "M.R.R."])
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    args = parser.parse_args()

    history_file = os.path.join(args.folder, "esim.pkl")

    with open(history_file, "rb") as f:
        data = pickle.load(f)

    show_accuracy_graph(np.array(data))
