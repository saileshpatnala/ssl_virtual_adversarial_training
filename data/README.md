# Data File Description

* `dbpedia.wordembeddings` & `dbpedia.wordembeddings.vectors.npy` 
These are the binary files for the Word2Vec model that is trained on the entire dataset. We use the utility function in the `gensim` library to load the model and it requires both of these binary.

* `dbpedia_test_wv.pkl`
Testing dataset with vectorized text. No class labels
Object - Pandas DataFrame

* `dbpedia_train_label_split1.pkl`
Labelled dataset with vectorized text and 2000 data points per class (5% of dataset)
Object - Pandas DataFrame

* `dbpedia_train_label_split2.pkl`
Labelled dataset with vectorized text and 500 data points per class (1.25% of dataset)
Object - Pandas DataFrame

* `dbpedia_train_wv.pkl`
Full training dataset with vectorized text and class labels.
Object - Pandas DataFrame