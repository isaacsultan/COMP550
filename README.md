# Natural Language Project (COMP550) #

COMP 550 Natural Language Programming Final Project

Students: 

- Hugo Scurti
- Isaac Sultan

Presented to Prof. Jackie Cheung

## Prerequisites ##

This project runs with __Python 3.6__. The project depends on the following libraries : 

- numpy
- pandas
- matplotlib
- Tensorflow
- NLTK
- gensim

In addition, the following steps should be done before running any preprocessing methods : 

### GloVe Pre-trained word embeddings ###

We used pre-trained word embeddings to combine with the word embeddings that are trained on the corpus. We used file [glove.42B.300d.zip](http://nlp.stanford.edu/data/glove.42B.300d.zip) as the pre-trained word embedding vectors. 

Download the zip file and extract the `.txt` file to the folder `data/`.

### Datasets ###

We generated training, test, and validation datasets from [the ubuntu dialogue corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator). Since the training file is somewhat large and cumbersome to load in memory, we split it into 10 subsets, and iterate through them sequentially. Generated files are stored in this [google drive folder](https://drive.google.com/drive/folders/1yIBjXgTuKPPw7SS_TVFHUpeNeZQ14-PU?usp=sharing). 

Download all the files and store them in the `data/` folder.

### Folder structure ###

Most of our functions write in folders dumps, data, checkpoints and result. In order for the write to succeed, the following folders must be created : 

- dumps
    - v1
    - v2
    - v3
- data
    - v1
    - v2
    - v3
- checkpoints
    - v1
    - v2
    - v3

## Code ##

Below we explain how to run every step of our methodology.

**NOTE**: All program executions should be done at the project's root folder.

### Preprocessing ###

#### 1. Tokenizing and lemmatization

This is done by calling the following command:
```
python data_processing.py -v i
``` 
,where i should be replaced by the version number associated with the model that you want to train. (1, 2 or 3)

This will create dump fils which will be put in `dumps/v[i]`


#### 2. Train word embeddings on corpus

This is done by calling the following command:

```
python models/train_we.py - v i
```

This command uses the dumped file generated in the previous step and train them using either word2vec (for versions 1 and 2) or fasttext (version 3). Results will be put in the appropriate dump folder.

**NOTE**: Training fasttext takes a considerable amount of time. Be aware of it.

#### 3. Filter pre-trained word embeddings

Before combining word embeddings, we filter the pre-trained glove embedding vectors, so that it takes less place when loading it in memory. This is done by calling the following command : 

```
python models/glove_filter.py -v i
```

This will take the appropriate word indices generated in the dump folder for the selected model version and filter the pre-trained glove dataset, outuputting the result in the appropriate data sub folder (e.g. `data/v1` for version 1).


#### 4. Combine word embeddings

This step uses files generated from the previous 3 steps to combine pre-trained word embeddings and trained word embeddings into 400 dimension word-embedding vectors. Use the following command to generate them : 
```
python models/train_we.py -v i
```


### Training ###

To train the model, use the following command: 
```
python train_model.py -v i
```

We recommend using tensorflow-gpu, as the model is significantly large and takes a while to train.

Since we've removed early stopping, the model should run until you stop it (using CTRL-C, or closing the program). Checkpoints for the last evaluated batch step and the best batch step are saved during training, so that we can use them afterwards to evaluate the model using the validation set.


### Evaluating ###

To evaluate a final model, use the following command:
```
python evaluation.py -v i -n model_name
```
, where `model_name` is an arbitrary name that will appear in the result table.

This evaluates the chosen model on the validation set and prints the result in the file `result/results.csv`. If the file doesn't exist, it creates it, otherwise it appends the result to the end of the table.
This method also generates in `dumps/v[i]/` a confusion matrix for the whole validation set and a confusion matrix for the filtered version of the validation set (with rows that contains urls and paths). 
This method also store the probabilities of each prediction in a file stored in `dumps/v[i]/`, which is used to calculate the mcnemar table shown in the report.

To print the graphs showing the evaluated metrics during the training phase, use the following command : 
```
python util/plotgraphs.py -f dumps/[i]
```

This function only needs the `esim.pkl` file generated during training.
We pass it as a folder so that it's easy to copy different versions of one model to different folders and print their graph one after the other. Per example, if we'd want to train the extended version 3 (with max. sequence length of 120), we could store its `esim.pkl` file in folder `dumps/v3_extended` and call the plotgraph file using this folder.

This function prints 3 graphs : 

1. Graph showing Test Accuracy and Training Accuracy;
2. Graph showing Test cost and training cost;
3. Graph showing Recall@1 and M.R.R. on test set;
