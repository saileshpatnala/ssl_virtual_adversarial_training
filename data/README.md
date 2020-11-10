# Data File Description

* `dbpedia.wordembeddings` & `dbpedia.wordembeddings.vectors.npy` 
These are the binary files for the Word2Vec model that is trained on the entire dataset. We use the utility function in the `gensim` library to load the model and it requires both of these binary.

**NOTE:** All the pkl files contain Pandas DataFrame objects

* `dbpedia_test_wv.pkl` - Testing dataset containing vectorized text. No class labels.

* `dbpedia_train_all_x.pkl` - DataFrame containing vectorized text for entire training dataset. No class labels.

* `dbpedia_train_all_y.pkl` - DataFrame containing class labels for entire training dataset.

* `dbpedia_train_wv.pkl` - Full training dataset with vectorized text and class labels.

* `dbpedia_train_x_split1.pkl` - DataFrame containing vectorized text for 28000 data points (2000 data points per class - 5% of entire dataset)

* `dbpedia_train_x_split2.pkl` - DataFrame containing vectorized text for 7000 data points (500 data points per class - 1.25% of entire dataset)

* `dbpedia_train_y_split1.pkl` - DataFrame containing class labels for 28000 data points (2000 data points per class - 5% of entire dataset)

* `dbpedia_train_y_split2.pkl` - DataFrame containing class labels for 7000 data points (500 data points per class - 1.25% of entire dataset)