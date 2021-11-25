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




sbert_model = SentenceTransformer('bert-base-nli-max-tokens')
path = "./data/scrapped_articles_new.xml"
query = "phone with large ram"

'''This file has 2 main classes,

Class Preprocessing
-------------------
    which reads the XML files and convert them to csv
    returns DataFrame


Class TFIDF
-----------
    Reads a dataframe and stores 2 files
    1 file for the transformer (which will be useed later to encode the query with 
    the same weights as the content.

    1 file with a scipy sparse matrix, which contains the weight for each word
    
'''


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

    




class TFIDF:
    def __init__(self,df):
        self.df = df


    def tfidf_vectorizer(self):
        X,vectors = self.vec_creator(self.df['content'])
        scipy.sparse.save_npz('sparse_matrix.npz', X)
        print(vectors)


    @staticmethod
    def vec_creator(dataset):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(dataset)
        with open('vectorizer.pk', 'wb') as f:
            pickle.dump(vectorizer, f)
        f.close()
        return X,vectorizer


class BertEmbendings:
    def __init__(self,df):
        self.df = df
        
      
    def sent_tokenize(self):
        self.df['content']= self.df['content'].apply(lambda x: x.replace("\n", " ").replace("-",' ').replace("_"," "))
        self.df['sentences'] = self.df['content'].apply(lambda x: sent_tokenize(x))
        self.df = self.df.explode('sentences').reset_index(drop=True)
        self.df.drop_duplicates(inplace=True,subset=['sentences'])
        self.df['embendings'] = self.df['sentences'].apply(self.bert_embendings)
        self.df.to_csv("content_with_embendings.csv",sep=';',index=False)

    @staticmethod
    def bert_embendings(content):
        sentence_embeddings = sbert_model.encode(content)
        return sentence_embeddings

if __name__=='__main__':

    # Parse XML to DF
    path = "./data/scrapped_articles_new.xml"
    file = Preprocessing(path)
    file.parser()
    df = file.transform_to_df(file.documents)

    # Save as csv
    df.to_csv("all_articles.csv",sep=';',index=False)
    
    # Use the content of the articles and calculate TFIDF
    # Store vectors and transformer as files
    tfidf = TFIDF(df)
    tfidf.tfidf_vectorizer()

    # Create Embendings and store to csv
    bert = BertEmbendings(df)
    bert.sent_tokenize()