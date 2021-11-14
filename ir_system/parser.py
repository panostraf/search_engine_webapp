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

path = "./data/scrapped_articles_new.xml"
query = "phone with large ram"


class Preprocessing:
    def __init__(self,file):
        try:
            self.file = open(file).read()
        except UnicodeDecodeError:
            try:
                self.file = open(file,encoding="cp1252").read()
        
            except UnicodeDecodeError:
                self.file = open(file,encoding="utf-8").read()
            self.documents = defaultdict(lambda: defaultdict(dict()))
        
        self.documents = defaultdict(lambda: defaultdict(dict()))

    
    
    def parser(self):
        soup = BeautifulSoup(self.file,'lxml')
        articles = soup.find_all("article")
        
        i = 0
        for article in articles:
            i+=1
            article_name = "article" + str(i)
            self.documents[article_name] = dict(url = article.find_all("url")[0].get_text(),
                                            title = article.find_all("title")[0].get_text(),
                                            content = article.find_all("content")[0].get_text())

    @staticmethod
    def transform_to_df(documents):
        df = pd.DataFrame.from_dict(dict(documents),orient='index')
        df.drop_duplicates(subset=['url'],inplace=True,keep="first")
        df.drop_duplicates(subset=['content'],inplace=True,keep="first")
        df.drop_duplicates(subset=['title'],inplace=True,keep="first")
        df.reset_index(inplace=True)
        return df

            

def words(row):

    # print(row['article'])
    # print(row['content'])
    words = [word.lower() for word in word_tokenize(row['content']) if (word.lower() not in stopwords.words('english')) and (word.lower().isalnum())]
    words = []
    
    return words

def vectorizer(dataset):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(dataset)
    return X,vectorizer

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

        

    def tfidf_vectorizer(self):
        X,vectors = self.vec_creator(self.df['content'])
        sim_vecs,cosine_similarities = self.calculate_similarity(X,vectors,self.query)
        output = self.df[self.df.index.isin(sim_vecs)]
        output['content'] = output['content'].apply(lambda x: x.replace("\n", " ").replace("-",' ').replace("_"," "))
        return output



    @staticmethod
    def vec_creator(dataset):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(dataset)
        return X,vectorizer

    @staticmethod
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

    @staticmethod
    def get_links(df):
        return df['url'].tolist()

    
def load_docs(full_file_path):
    file = Preprocessing(full_file_path)
    file.parser()
    df = file.transform_to_df(file.documents)
    return df

def ir_tfidf(df,query):
    # file = Preprocessing(full_file_path)
    # file.parser()
    # df = file.transform_to_df(file.documents)
    tfidf = TFIDF(df,query)
    new_df = tfidf.tfidf_vectorizer()
    links = tfidf.get_links(new_df)
    print(links)
    return links


if __name__ == "__main__":

    path = "./corpus/scrapped_articles_new.xml"
    query = "phone with large ram"

    ir_tfidf(path,query)
    exit(-1)
    print('reading documents')
    file = Preprocessing("./corpus/scrapped_articles_new.xml")
    file.parser()
    df = file.transform_to_df(file.documents)
    query = "test query"
    tfidf = TFIDF(df,query)
    new_df = tfidf.tfidf_vectorizer()
    print(new_df)
    exit(-1)
    
    # df = pd.DataFrame.from_dict(dict(file.documents),orient='index')
    # df.drop_duplicates(subset=['url'],inplace=True,keep="first")
    # df.drop_duplicates(subset=['content'],inplace=True,keep="first")
    # df.drop_duplicates(subset=['title'],inplace=True,keep="first")
    # df.reset_index(inplace=True)
    # print(df)
    X,vectors = vectorizer(df['content'])
    
    print('reading query')

    test_query = ["macbook pro m1"]

    sim_vecs,cosine_similarities = calculate_similarity(X,vectors,test_query)
    print(sim_vecs)
    
    indexes = show_similar_documents(df['content'],cosine_similarities,sim_vecs)
    new_df = df[df.index.isin(sim_vecs)]
    print(new_df)
    new_df['content'] = new_df['content'].apply(lambda x: x.replace("\n", " ").replace("-",' ').replace("_"," "))
    new_df['sentences'] = new_df['content'].apply(lambda x: sent_tokenize(x))
    print('df sent tokenize')
    print(new_df)
    new_df = new_df.explode('sentences').reset_index(drop=True)
    new_df.drop_duplicates(inplace=True,subset=['sentences'])
    print('df sent expanded')
    print(new_df)
    new_df['embendings'] = new_df['sentences'].apply(bert_embendings)
    print(new_df)
    sim_vecs,cosine_similarities = calculate_similarity_bert(new_df['embendings'], test_query, top_k=3)
    exit(-1)



