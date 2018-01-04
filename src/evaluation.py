import os
import argparse

import pickle
import numpy as np
import pandas as pd

import load
import train_model as model
from util.params import Params
from util.filter_indices import get_filtered_indices

# This code was adapted from https://github.com/nyu-mll/multiNLI
# See NOTICE.txt for modification details

result_file = "./result/results.csv"
data_colums = [
    'Name', 'Global Step', 'R@1', 'R@2', 'R@5', 'MRR', '(Filtered) R@1',
    '(Filtered) R@2', '(Filtered) R@5', '(Filtered) MRR'
]


def indices_gen(array_length):
    index = 0
    indices = []
    for i in range(array_length):
        indices.append(index)
        if (i + 1) % 10 == 0:
            index += 1
    return indices


def join_rank_predictions(y_true, y_predict, indices):
    df = pd.DataFrame({
        'context_indices': indices,
        'response': y_true[:, 1],
        'likelihood': y_predict,
        'label': y_true[:, 2]
    })
    sorted_y = df.sort_values(
        by=['context_indices', 'likelihood'], ascending=[True, False])
    return sorted_y


def rank_predictions(X_matrix, predictions, labels, label_col=2):
    df = pd.DataFrame({
        'context': X_matrix[0],
        'response': X_matrix[1],
        'prediction': predictions
    })
    sorted_df = df.sort_values(
        by=['context', 'prediction'], ascending=[True, False])
    return sorted_df


def recall_at_k(predictions, K=1, ranks=10, params=None, isFiltered=True):
    """Recall = TP/(TP+FN)"""
    tp, fn = 0.0, 0.0
    itervals = predictions.values
    label_idx = predictions.columns.get_loc("label")
    N = itervals.shape[0]

    recall_vector = []

    for p in range(0, N, ranks):
        found_true = False
        for k in range(0, K):
            if itervals[p + k][label_idx] == 1:
                tp += 1.0
                found_true = True
                recall_vector.append(1)
                break
        if not found_true:
            fn += 1.0
            recall_vector.append(0)

    if K == 1 and params is not None:
        name = os.path.join(params.dump_folder, "recall_vector")
        if isFiltered:
            name += "_filtered.pkl"
        else:
            name += ".pkl"
        with open(name, "wb") as f:
            pickle.dump(recall_vector, f)
    return tp / (tp + fn)


def precision_at_1(predictions, K=1, ranks=10):
    """Precision = TP/(TP+FP)"""
    tp, fp = 0, 0
    itervals = predictions.values
    label_idx = predictions.columns.get_loc("label")
    N = itervals.shape[0]
    for p in range(0, N, ranks):
        found_true = False
        for k in range(0, K):
            if itervals[p + k][label_idx] == 1:
                tp += 1.0
                found_true = True
                break
        if not found_true:
            fp += 1.0

    return tp / (tp + fp)


def mean_reciprocal_rank(predictions, ranks=10):
    """MMR = (1/Q)*SUM(1/rank_i)"""
    itervals = predictions.values
    label_idx = predictions.columns.get_loc("label")
    N = itervals.shape[0]
    Q = N / ranks
    reciprical_rank = 0.0
    for p in range(0, N, ranks):
        for rank in range(0, ranks):
            if itervals[p + rank][label_idx] == 1:
                reciprical_rank += (1.0 / (rank + 1.0))
    return reciprical_rank / Q


def mcnemar_table(x, y):
    table = np.zeros((2, 2), dtype=np.int)

    name = os.path.join("dumps", x, "recall_vector.pkl")
    with open(name, "rb") as f:
        vector_x = pickle.load(f)

    name = os.path.join("dumps", y, "recall_vector.pkl")
    with open(name, "rb") as f:
        vector_y = pickle.load(f)

    for i in range(len(vector_x)):
        table[vector_x[i],vector_y[i]] += 1

    file_name = x + "_" + y + "mcnemar_table.csv"
    name = os.path.join("dumps", x, file_name)
    np.savetxt(name, table, delimiter=",")


def evaluate_classifier(classifier, eval_set, batch_size, is_validation=False):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    recall_at_1 = -1
    mrr = -1
    hypotheses, probs, cost = classifier(eval_set)

    if is_validation:
        # We add more information when it's the validation set
        indices = indices_gen(len(eval_set))
        y_true = np.array(eval_set)
        # Join in one dataframe
        sorted_res = join_rank_predictions(y_true, probs, indices)

        recall_at_1 = recall_at_k(sorted_res, 1, 10)
        mrr = mean_reciprocal_rank(sorted_res, 10)

    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        if hypothesis == eval_set[i][2]:
            correct += 1
    return correct / float(len(eval_set)), cost, recall_at_1, mrr


def evaluate_final(restore, classifier, eval_set, batch_size):
    """
    Function to get the MRR of the model, evaluated on a set of chosen datasets.
    
    restore: a function to restore a stored checkpoint
    classifier: the model's classfier, it should return logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    restore(best=True)

    hypotheses, probs, cost = classifier(eval_set)

    # We add more information when it's the validation set
    indices = indices_gen(len(eval_set))
    y_true = np.array(eval_set)

    # Join in one dataframe
    sorted_res = join_rank_predictions(y_true, probs, indices)

    return mean_reciprocal_rank(sorted_res, 10)


def getconfmatrix(predicted_y, true_y):
    res = np.zeros((2,2))

    for i in range(predicted_y.shape[0]):
        res[predicted_y[i], true_y[i]] += 1

    return res


def main():
    """
    Order of operations:
        1. append column of labels to predictions
            s.t. predictions = <context, response, likelihood> AND <label> ==> labeled_predictions = <context, response, likelihood, label>
        2. rank responses
            s.t. ranked_labeled_predictions = [<context_1, response_1a, likelihood_1a, label_1a>, <context_1, response_1b, likelihood_1b, label_1b>, ..., <context_N, response_Nj, likelihood_Nj, label_1j>]
        3. evaluate predictions
            ex. recall_at_k(labeled_ranked_preditions, 10)
            ex. mean_reciprical_rank(labeled_ranked_predictions)

    for the Ubuntu set, ranks should always be 10 so it's ok to always use the default value
    'predictions' is expecting = <context, response, likelihood, label>
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str)
    params = Params(parser)

    # Prepare classifier
    validation_file = os.path.join(params.dump_folder, "valid_expanded.pkl")
    word_indices_path = os.path.join(params.dump_folder, "word_indices.pkl")
    word_embedding_file = os.path.join(params.dump_folder, "enhanced_we.pkl")

    print("Loading validation file...")
    with open(validation_file, "rb") as f:
        valid_set = pickle.load(f)
    print("Done.")

    print("Loading word embeddings...")
    with open(word_embedding_file, "rb") as f:
        word_embeddings = pickle.load(f)

    embedding_mat, word_indices = load.load_embedding_matrix(
        word_embeddings, word_indices_path, 400)
    print("Done.\n")

    print("Instanciating classifier.")
    classifier = model.modelClassifier(embedding_mat, params.checkpoint_folder)
    classifier.restore(True)
    print("Done.\n")

    print("Converting datasets into indices")
    model.get_embed_indices(valid_set, word_indices,
                            classifier.sequence_length)
    # We also append the index for unique contexts
    indices = indices_gen(len(valid_set))
    print("Done.")

    # Train
    result = []
    result.append(params.args.name)

    print("Training")
    hypotheses, probs, cost = classifier.classify(valid_set)
    print("Done.")

    print("Calculating metrics")
    y_true = np.array(valid_set)
    sorted_res = join_rank_predictions(y_true, probs, indices)

    # Global step
    global_step = classifier.sess.run(classifier.global_step)
    result.append(global_step)

    # Get confusion matrix
    labels = sorted_res["label"].values.astype('i')
    hypotheses = np.array(hypotheses, dtype='i')
    conf_matr = getconfmatrix(hypotheses, labels)
    # Print confusion matrix in dump folder
    np.savetxt(os.path.join(params.dump_folder, "conf_matrix.csv"), conf_matr, delimiter=",")

    # Evaluate first set of results
    print("Step : {}".format(global_step))
    for i in [1, 2, 5]:
        recall = recall_at_k(sorted_res, i, 10, params, False)
        result.append(recall)
        print("Recall @ {}: {}".format(i, recall))

    mrr = mean_reciprocal_rank(sorted_res, 10)
    result.append(mrr)
    print("Mean Reciprocal Rank: {}".format(mrr))

    # Filter out only data that contains urls and paths
    filtered_indices = get_filtered_indices()
    new_res = sorted_res[sorted_res.index.isin(filtered_indices)]
    print("Dataframe shape: {}",format(new_res.shape))

    # Get filtered confusion matrix
    labels = labels[filtered_indices]
    hypotheses = hypotheses[filtered_indices]
    conf_matr = getconfmatrix(hypotheses, labels)
    # Print filtered confusion matrix in dump folder
    np.savetxt(os.path.join(params.dump_folder, "conf_matrix_filtered.csv"), conf_matr, delimiter=",")
    

    # Recalculate metrics
    for i in [1, 2, 5]:
        recall = recall_at_k(new_res, i, 10, params, True)
        result.append(recall)
        print("Filtered Recall @ {}: {}".format(i, recall))

    mrr = mean_reciprocal_rank(new_res, 10)
    result.append(mrr)
    print("Filtered MRR: {}".format(mrr))

    # Add to result file
    if os.path.isfile(result_file):
        data = pd.read_csv(
            result_file, encoding="utf-8", keep_default_na=False, header=0)
    else:
        data = pd.DataFrame(columns=data_colums)

    data_dict = dict(zip(data_colums, result))
    data = data.append(data_dict, ignore_index=True)

    # Write back to file
    data.to_csv(
        result_file, columns=data_colums, index=False, encoding='utf-8')


if __name__ == '__main__':
    main()


