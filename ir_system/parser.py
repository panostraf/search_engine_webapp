from typing import final
from bs4 import BeautifulSoup
import nltk
from nltk.corpus.reader.chasen import test
from nltk.util import pr
import pandas as pd
from collections import defaultdict
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas.io.formats.format import return_docstring
import numpy as np
from nltk import word_tokenize,sent_tokenize
from sentence_transformers import SentenceTransformer
from ir_system import functions
import numpy as np
pd.set_option('display.max_columns', None)
# nltk.download('stopwords')
from ir_system.functions import *
import pickle
from scipy.sparse import csr_matrix
import scipy


path = "./data/scrapped_articles_new.xml"
query = "phone with large ram"


def calculate_similarity(X, vectorizor, query, top_k=20):
    """ Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of
    the `query` and `X` (all the documents) and returns the `top_k` similar documents."""

    # Vectorize the query to the same length as documents
    query_vec = vectorizor.transform(query)
    # Compute the cosine similarity between query_vec and all the documents
    cosine_similarities = cosine_similarity(X,query_vec).flatten()
    # Sort the similar documents from the most similar to less similar and return the indices
    most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
    return (most_similar_doc_indices, cosine_similarities)

def show_similar_documents(df, cosine_similarities, similar_doc_indices):
    """ Prints the most similar documents using indices in the `similar_doc_indices` vector."""
    counter = 1
    indexes = []
    for index in similar_doc_indices:
        # print()
        indexes.append(index)
        counter += 1
    return indexes

def show_similar_documents_bert(df, cosine_similarities, similar_doc_indices):
    """ Prints the most similar documents using indices in the `similar_doc_indices` vector."""
    counter = 1
    indexes = []
    for index in similar_doc_indices:
        # print('Top-{}, Similarity = {}'.format(counter, cosine_similarities[index]))
        # print('body: {}, '.format(df[index]))
        # print(df['article'].iloc[index])
        # print(df.columns)
        # print()
        indexes.append(index)
        # print(index)
        counter += 1
    return indexes

def calculate_similarity_bert(X, query, top_k=20):
    """ Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of
    the `query` and `X` (all the documents) and returns the `top_k` similar documents."""
    # X = X.value
    # X = X.flatten()
    X = X.to_list()
    # Vectorize the query to the same length as documents
    query_embed = np.array(sbert_model.encode(query))
    # Compute the cosine similarity between query_vec and all the documents
    cosine_similarities = cosine_similarity(X,query_embed).flatten()
    # Sort the similar documents from the most similar to less similar and return the indices
    most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
    return (most_similar_doc_indices, cosine_similarities)


class TFIDF:
    def __init__(self,df,query):
        self.df = df
        self.query = [query]


    def tfidf_vectorizer_fast(self,X,vectors):
        sim_vecs,cosine_similarities = self.calculate_similarity(X,vectors,self.query)
        output = self.df[self.df.index.isin(sim_vecs)]
        output['content'] = output['content'].apply(lambda x: x.replace("\n", " ").replace("-",' ').replace("_"," "))
        output['similarities'] = cosine_similarities
        output = output.sort_values(by='similarities',ascending=False)
        print(output)
        return output


    @staticmethod
    def vec_creator(dataset):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(dataset)
        return X,vectorizer


    @staticmethod
    def calculate_similarity(X, vectorizor, query, top_k=50):
        """ Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of
        the `query` and `X` (all the documents) and returns the `top_k` similar documents."""

        query_vec = vectorizor.transform(query)
        cosine_similarities = cosine_similarity(X,query_vec).flatten()
        most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
        cosine_sims = [cosine_similarities[item] for item in np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]]
        return (most_similar_doc_indices, cosine_sims)


    @staticmethod
    def get_links(df):
        return df['url'].tolist()

    @staticmethod
    def get_titles(df):
        return df['title'].tolist()

    @staticmethod
    def get_similarities(df):
        return df['similarities'].tolist()


def ir_tfidf(df,query,X,vectors):
    tfidf = TFIDF(df,query)
    new_df = tfidf.tfidf_vectorizer_fast(X,vectors)
    links = tfidf.get_links(new_df)
    titles = tfidf.get_titles(new_df)
    similarities = tfidf.get_similarities(new_df)
    return links, titles, similarities


    

def main(query,documents,X,vectors):
    links,titles,similarities = ir_tfidf(documents,query,X,vectors)
    similarities = [round(sim,2) for sim in similarities]

    results = dict(zip(links,list(zip(titles,similarities))))

    average_similarity = round(np.mean(similarities),2)

    text = ""
    for link,data in results.items():
        title = data[0]
        similarity = data[1]
        text = text + f'{query};{title};{link};{similarity}\n'
    with open("dataset_removed_stops_tfidf.csv","a") as f:
        f.write(text)

    return results,query,similarities,average_similarity


def bert_similarities(query,df,bert_embedings,sbert_model,top_k=50):
    bert_query = sbert_model.encode(query)
    query_sm = csr_matrix(bert_query)
    sim_sparse = cosine_similarity(bert_embedings, query_sm)

    most_similar_doc_indices = np.argsort(sim_sparse, axis=0)[:-top_k-1:-1]
    best_articles = [article for article in most_similar_doc_indices.flatten()]
    df['similarities'] = sim_sparse
    output = df[df.index.isin(best_articles)]
    
    output = output.sort_values(by='similarities',ascending=False)
    print(output)
    sentences = output['sentences'].tolist()
    similarities = output['similarities'].tolist()
    links = output['url'].tolist()
    return links, sentences, similarities




def main_bert(query,documents,bert_embedings,sbert_model):
    links,sentences,similarities = bert_similarities(query,documents,bert_embedings,sbert_model)
    similarities = [round(sim,2) for sim in similarities]
    results = dict(zip(links,list(zip(sentences,similarities))))


    text = ""
    for link,data in results.items():
        sentence = data[0]
        similarity = data[1]
        text = text + f'{query};{sentence};{link};{similarity}\n'
    with open("train_dataset_bert.csv","a") as f:
        f.write(text)

    return results,query,similarities



if __name__ == "__main__":

    path = "./corpus/scrapped_articles_new.xml"
    query = "phone with large ram"

    ir_tfidf(path,query)
    exit(-1)