"""
EE219 Winter 2017
Project 2 - Task f to Task j

Zeyu Li
lizeyu_cs@foxmail.com
2017-02-07

This file provides some toolkit functions for the sub-tasks from f to j
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD



# Set the datasets names
comp_categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
rec_categories =  ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

def preprocess_train_set(dataset, k=20):
    """
    Pre-processing pipeline for the training data:
     (1) Do the Count Vectorization on the data set
     (2) Do the Tf-Idf transformation
     (3) Do the LSI (Latent Semantic Indexing or LSA Latent Semantic Analysis) transformation

    :param dataset: the training part of a certain data set
    :return: the fetched, Counted, Tf-Idf transformed, and LSI transformed matrix
    """

    # Construct an CountVectorizer and do count vectorization

    count_vector   = CountVectorizer()
    print "Counting Vectorization..."
    dataset_counts = count_vector.fit_transform(dataset)
    print "Done!"


    # Construct the Tf-Idf object and do tf-idf transformation
    tfidf_transformer = TfidfTransformer()
    print "Computing Tf-Idf Matrix..."
    dataset_tfidf     = tfidf_transformer.fit_transform(dataset_counts)
    print "Done!"

    # Apply LSI (Latent Semantic Indexing) on Tf-Idf matrices
    # Reduce the dimension of "comp_train_tfidf" and "rec_train_tfidf" to "n_component=50"
    # to get the matrix "dataset_dr". "dr" stands for "dimensional reduced"
    lsi_transformer = TruncatedSVD(n_components=k)
    print "Computing LSI with k={}".format(k)
    dataset_dr      = lsi_transformer.fit_transform(dataset_tfidf)
    print "Done!"
    return dataset_dr

def preprocess(categories, subset="train", k=20):
    """
    Fetch the data set and return the pre-processed training data and the target
    :param categories: The categories of the data set to play with
    :param subset: "train" or "test", default as "train"
    :return: the training part of the data, the target set, and the target name
    """
    dataset             = fetch_20newsgroups(subset=subset, categories=categories, shuffle=True, random_state=42)
    data, target, names = dataset.data, dataset.target, dataset.target_names
    print "Processing data from {}".format(str(categories))
    return preprocess_train_set(data, k=k), target, names

