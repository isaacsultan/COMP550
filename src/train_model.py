import pickle
import os
import random
import math

import tensorflow as tf
import pandas as pd
import numpy as np

from models.tf_esim import TfEsim
import evaluation as evl
import load
from util.params import Params

# This code was adapted from https://github.com/nyu-mll/multiNLI
# See NOTICE.txt for modification details

_unk = "<UNK>"
_pad = "<PAD>"

modname = "esim"

def get_embed_indices(dataset, word_indices, sequence_length):

    # We modify the values in place
    for i in range(len(dataset)):
        # one row contains the premise, the hypothesis and the value
        for j in range(2):
            # Store old values in temp
            temp = dataset[i][j]
            sent_length = len(temp)
            dataset[i][j] = [None]*sequence_length

            for k in range(sequence_length):
                if k >= sent_length:
                    val = word_indices[_pad]
                else:
                    if temp[k] in word_indices:
                        val = word_indices[temp[k]]
                    else:
                        val = word_indices[_unk]

                dataset[i][j][k] = val


class modelClassifier:
    def __init__(self, embedding_matrix, checkpoint_path):
        ## Define hyperparameters

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = 0.001

        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.evaluate_val_freq = 500
        self.embedding_dim = 400
        self.dim = 200  # Word Bilstm dimension
        self.batch_size = 128
        self.emb_train = False
        self.keep_rate = 1.0    # Dropout rate (or the complementary of dropout)
        self.sequence_length = 100
        # self.sequence_length = 120    # To train the extended v3 version, comment the previous
                                        # Line and uncomment this one.

        self.model = TfEsim(seq_length=self.sequence_length, emb_dim=self.embedding_dim,  
                            hidden_dim=self.dim, embeddings=embedding_matrix, emb_train=self.emb_train)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.model.total_cost, global_step=self.global_step)

        # Boolean stating that training has not been completed, 
        self.completed = False 

        # tf things: initialize variables and create placeholder for session
        print("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

        # Store checkpoint paths in object
        self.ckpt_path = checkpoint_path
        self.ckpt_file = ckpt_file = os.path.join(self.ckpt_path, modname) + ".ckpt"


    def get_minibatch(self, dataset, start_index, end_index):
        # We stop at the end of the dataset in case the end_index is bigger
        indices = range(start_index, min(end_index, len(dataset)))
        
        # We're flattening the sentences to a list of token (regardless of the sentences) ? 
        premise_vectors = np.vstack([dataset[i][0] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i][1] for i in indices])
        labels = [dataset[i][2] for i in indices]
        return premise_vectors, hypothesis_vectors, labels


    def train(self, train, valid):        
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.step = 0
        self.epoch = 0
        self.best_val_acc = 0.
        self.best_mrr = 0.
        self.best_rec1 = 0.
        self.best_train_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0

        # Restore most recent checkpoint if it exists. 
        # Also restore values for best test MRR result
        best_ckpt_file = self.ckpt_file + "_best"
        res_dump_files = os.path.join(self.ckpt_path, modname) + ".pkl"

        if os.path.isfile(res_dump_files):
            with open(res_dump_files, "rb") as f:
                log_res = pickle.load(f)
        else:
            log_res = []

        if os.path.isfile(self.ckpt_file + ".meta"):

            if os.path.isfile(best_ckpt_file + ".meta"):
                self.saver.restore(self.sess, best_ckpt_file)
                self.best_val_acc, val_cost, self.best_rec1, self.best_mrr = evl.evaluate_classifier(self.classify, valid[0:10000], self.batch_size, True)
                self.best_train_acc, train_cost, _1, _2 = evl.evaluate_classifier(self.classify, train[0:5000], self.batch_size)
                    
                print("Restored best MRR: {}\n Restored best Recall@1: {}".format(self.best_mrr, self.best_rec1))

            self.saver.restore(self.sess, self.ckpt_file)
            print("Model restored from file: {}".format(self.ckpt_file))

        training_data = train

        ### Training cycle
        print("Training...")

        while True:
            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)

            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels = self.get_minibatch(
                    training_data, self.batch_size * i, self.batch_size * (i + 1))
                
                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: self.keep_rate}
                _, c, glob = self.sess.run([self.optimizer, self.model.total_cost, self.global_step], feed_dict)


                self.step += 1

                
                if self.step % self.display_step_freq == 0:
                    print("Batch step {}".format(self.step), end="\r")

                # we evaluate the model only at 500 steps, because this takes a while
                if self.step % self.evaluate_val_freq == 0:
                    print("\nEvaluating accuracy...")
                    val_acc, val_cost, rec1, mrr = evl.evaluate_classifier(self.classify, valid[0:10000], self.batch_size, True)
                    train_acc, train_cost, _1, _2 = evl.evaluate_classifier(self.classify, train[0:5000], self.batch_size)

                    print("Step: {}\t Validation Recall@1: {}\t Validation MRR: {}".format(glob, rec1, mrr))
                    print("Step: {}\t Validation acc: {}\t Train acc: {}".format(glob, val_acc, train_acc))
                    print("Step: {}\t Validation cost: {}\t Train cost: {}".format(glob, val_cost, train_cost))

                    # Add it to result
                    log_res.append([glob, val_acc, val_cost, train_acc, train_cost, rec1, mrr])

                    # Dump it
                    with open(res_dump_files, "wb") as f:
                        pickle.dump(log_res, f)

                    # Save the model
                    self.saver.save(self.sess, self.ckpt_file)
                    best_test = 100 * (1 - self.best_mrr / mrr)
                    if self.step > 100 and best_test > 0.04:
                        self.saver.save(self.sess, best_ckpt_file)
                        self.best_mrr = mrr
                        self.best_train_acc = train_acc
                        self.best_step = self.step
                        print("Checkpointing with new best validation MRR: {}".format(self.best_mrr))

                # Compute average loss
                avg_cost += c / (total_batch * self.batch_size)
                                
            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                print("Epoch: {}\t Avg. Cost: {}".format(self.epoch+1, avg_cost))
            
            self.epoch += 1 
            self.last_train_acc[(self.epoch % 5) - 1] = train_acc

            # Early stopping
            # We commented early stopping because it caused us problems (i.e. some model was stopped when it shouldn't be)
            #progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1) 

            #if (progress < 0.1) or (self.step > self.best_step + 30000):
            #    print("Best MRR: {}".format(self.best_mrr))
            #    print("Train accuracy: {}".format(self.best_train_acc))
            #    self.completed = True
            #    break

    def restore(self, best=True):
        if best:
            path = self.ckpt_file + "_best"
        else:
            path = self.ckpt_file

        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver.restore(self.sess, path)
        print("Model restored from file: {}".format(path))

    def classify(self, examples):
        # This classifies a list of examples
        total_batch = math.ceil(len(examples) / self.batch_size)
        logits = np.empty(0)
        logit_probs = np.empty(0)
        for i in range(total_batch):
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels = self.get_minibatch(
                examples, self.batch_size * i, self.batch_size * (i + 1))

            feed_dict = {self.model.premise_x: minibatch_premise_vectors, 
                                self.model.hypothesis_x: minibatch_hypothesis_vectors,
                                self.model.y: minibatch_labels, 
                                self.model.keep_rate_ph: 1.0}

            logit, logit_prob, cost = self.sess.run([self.model.logits, self.model.logit_probs, self.model.total_cost], feed_dict)
            logits = np.concatenate([logits, logit])
            logit_probs = np.concatenate([logit_probs, logit_prob])

        return logits, logit_probs, cost



if __name__ == "__main__":

    params = Params()

    train_file_format = os.path.join(params.dump_folder, "train_{}.pkl")
    valid_file = os.path.join(params.dump_folder, "valid_expanded.pkl")
    test_file = os.path.join(params.dump_folder, "test_expanded.pkl")

    word_indices_path = os.path.join(params.dump_folder, "word_indices.pkl")
    word_embeddings_file = os.path.join(params.dump_folder, "enhanced_we.pkl")
    

    print("Loading dump dataset files...")
    # Load data before initializng classifier (i.e. training and test set, embedding matrix
    train = []
    for i in range(1,11):
        with open(train_file_format.format(i), "rb") as f:
            train.extend(pickle.load(f))

    with open(valid_file, "rb") as f:
        valid = pickle.load(f)

    with open(test_file, "rb") as f:
        test_set = pickle.load(f)
    print("Done.\n")

    print("Loading word embeddings...")
    with open(word_embeddings_file, "rb") as f:
        word_embeddings = pickle.load(f)

    embedding_mat, word_indices = load.load_embedding_matrix(word_embeddings, word_indices_path, 400)
    print("Done.\n")

    print("Instanciating classifier.")
    classifier = modelClassifier(embedding_mat, params.checkpoint_folder)
    print("Done.\n")

    print("Converting datasets into indices")
    # Resulting train and valid will have three columns : 1 is premise, 2 is hypothesis and 3 is label (1 or 0)
    get_embed_indices(train, word_indices, classifier.sequence_length)
    get_embed_indices(valid, word_indices, classifier.sequence_length)
    get_embed_indices(test_set, word_indices, classifier.sequence_length)
    print("Done.\n")

    """
    Either train the model and then run it on the test-sets or 
    load the best checkpoint and get accuracy on the test set. Default setting is to train the model.
    """
    test = False

    print("Training...")
    if test == False:
        classifier.train(train, test_set)
        # Since we removed early stopping, the following lines are never executed
        print("MRR on test set: {}".format(evl.evaluate_classifier(classifier.classify, test_set, classifier.batch_size, True)[3]))
        print("MRR on validation set: {}".format(evl.evaluate_classifier(classifier.classify, valid, classifier.batch_size, True)[3]))
        print("Acc on training set: {}".format(evl.evaluate_classifier(classifier.classify, train, classifier.batch_size)[0]))
    else: 
        result = evl.evaluate_final(classifier.restore, classifier.classify, valid, classifier.batch_size)
        print("MRR on Test set: {}".format(result))
    print("Done.\n")