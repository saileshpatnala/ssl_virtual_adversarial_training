import os
import pandas as pd
import numpy as np
import nltk
import re
import random
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec, KeyedVectors

# downloading necessary nltk
nltk.download('wordnet')
nltk.download('stopwords')

DATA_DIR = 'dbpedia_csv'

"""Loading the data"""
def load_data():
    train_data_path = os.path.join(data_dir, 'train.csv')
    test_data_path = os.path.join(data_dir, 'test.csv')

    train_df = pd.read_csv(train_data_path, header=None, names=['class', 'title', 'text'])
    test_df = pd.read_csv(test_data_path, header=None, names=['class', 'title', 'text'])

    return (train_df, test_df)

"""Preprocessing text"""
def preprocess_text(text):
    # removing numbers
    text = re.sub('[0-9]+', '', text)
    
    # removing urls
    text = re.sub(r'http\S+', '', text)
    
    # removing punctuation and special characters
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # convert to lowercase and lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token.lower(), pos='v') for token in tokens]
    
    # remove stop words
    keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
    
    # remove small words
    keywords = [word for word in keywords if len(word) > 2]
    
    return keywords

def preprocess_data(df):
    df['preprocess_text'] = df.text.apply(preprocess_text)
    return df

"""Generating word embeddings"""
def train_word_embedding_model(data):
    w2c_model = Word2Vec(sentences=data, size=300, min_count=1, window=5, workers=4, sg=1)
    return w2c_model
sentences = pd.concat([train_df.preprocess_text, test_df.preprocess_text], axis=0)
w2c_model = Word2Vec(sentences=sentences, size=300, min_count=1, window=5, workers=4, sg=1)
w2c_model.wv.vectors.shape

"""Saving the word embeddings"""
def save_word_embeddings(model):
    model.wv.save('dbpedia.wordembeddings')

"""Loading word embeddings"""
def load_word_embeddings():
    word_vectors = KeyedVectors.load('dbpedia.wordembeddings', mmap='r')
    return word_vectors

"""Generating word vectors for text"""
def vectorize_text(text, wv):
    vec = np.zeros((1, 300))
    for w in text:
        vec += wv.get_vector(w)
    return vec / len(text)

def vectorize_data(df, word_vectors):
    df['text_vec'] = df.preprocess_text.apply(vectorize_text, args=(word_vectors,))
    return df

def save_text_vec_to_pickle():
    pd.DataFrame(train_df[['text_vec', 'class']]).to_pickle('dbpedia_train_wv.pkl')
    pd.DataFrame(test_df.text_vec).to_pickle('dbpedia_test_wv.pkl')

def generate_label_splits(df):
    num_per_class_label_split1 = 2000
    num_per_class_label_split2 = 500

    label_split1_df = pd.DataFrame(columns=['text_vec', 'class'])
    label_split2_df = pd.DataFrame(columns=['text_vec', 'class'])

    for c in classes:
        c_idx = np.where(df['class'] == c)[0]
        split1_idx = random.sample(c_idx.tolist(), num_per_class_label_split1)
        label_split1_df = label_split1_df.append(df[['text_vec', 'class']].iloc[split1_idx])

        split2_idx = random.sample(c_idx.tolist(), num_per_class_label_split2)
        label_split2_df = label_split2_df.append(df[['text_vec', 'class']].iloc[split2_idx])

    return label_split1_df, label_split2_df

def save_label_splits_to_pickle(label_split1_df, label_split2_df):
    label_split1_df.to_pickle('dbpedia_train_label_split1.pkl')
    label_split2_df.to_pickle('dbpedia_train_label_split2.pkl')

def main():
    train_df, test_df = load_data()
    train_df = preprocess_data(train_df)
    sentences = pd.concat([train_df.preprocess_text, test_df.preprocess_text], axis=0)
    model = train_word_embedding_model(sentences)
    save_word_embeddings(model)
    word_vectors = load_word_embeddings()
    train_df = vectorize_data(train_df, word_vectors)
    test_df = vectorize_data(test_df, word_vectors)
    label_split1_df, label_split2_df = generate_label_splits(train_df)
    save_label_splits_to_pickle(label_split1_df, label_split2_df)


if __name__ == '__main__':
    main()
