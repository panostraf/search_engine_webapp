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
import itertools

lemmatizer = WordNetLemmatizer()
path = "./data/scrapped_articles_new.xml"
# query = "phone with large ram"

class TFIDF:
    # Class to calculate the cosine similarity using tf-idf
    def __init__(self,df,query):
        self.df = df
        self.query = [query]

    def tfidf_vectorizer_fast(self,X,vectors):
        sim_vecs,cosine_similarities = self.calculate_similarity(X,vectors,self.query)
        output = self.df[self.df.index.isin(sim_vecs)]
        output['content'] = output['content'].apply(lambda x: x.replace("\n", " ").replace("-",' ').replace("_"," "))
        output['similarities'] = cosine_similarities
        output = output.sort_values(by='similarities',ascending=False)
        print("TFIDF DOC RANKING")
        print(output)
        return output

    @staticmethod
    def vec_creator(dataset):
        # this is an old function and not in use anymore
        # It will calculate tfidf for the corpus
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(dataset)
        return X,vectorizer

    @staticmethod
    def calculate_similarity(X, vectorizor, query, top_k=10):
        # vectorize query using the model and calculate the similarity
        # returns the top n articles
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
    # It uses the tfidf class from above
    # this function is meant to be called from main
    tfidf = TFIDF(df,query)
    new_df = tfidf.tfidf_vectorizer_fast(X,vectors)
    links = tfidf.get_links(new_df)
    titles = tfidf.get_titles(new_df)
    similarities = tfidf.get_similarities(new_df)
    return links, titles, similarities   

def main(query,documents,X,vectors):
    # This is the class that the flask app will call 
    # given a query it expanded and search using tfidf 
    # for a list of queries that it creates
    # Then it will sort all those queries and 
    # keep the most similar
    queries = functions.synonyms_production(query)
    links,titles,similarities = [],[],[]
    for q in queries:
        print( f"searching for {q}\n\n\n")
        q = lemmatizer.lemmatize(q)
        l,t,s = ir_tfidf(documents,q,X,vectors)
        links.append(l)
        titles.append(t)
        similarities.append(s)
    links = list(itertools.chain(*links))
    titles = list(itertools.chain(*titles))
    similarities = list(itertools.chain(*similarities))
    temp_df = pd.DataFrame(list(zip(links, titles)),
               columns =['links', 'titles'])
    temp_df['similarities'] = similarities
    temp_df = temp_df.sort_values(by='similarities',ascending=False)
    temp_df.drop_duplicates(subset='links',inplace=True)
    link = temp_df['links'].tolist()
    title = temp_df['titles'].tolist()
    similarities = temp_df['similarities'].tolist()
    similarities = [round(sim,2) for sim in similarities]
    results = dict(zip(links,list(zip(titles,similarities))))
    text = ""
    i = 0
    for link,data in results.items():
        title = data[0]
        similarity = data[1]
        text = text + f'{query};{title};{link};{similarity}\n'
        i+=1
        if i >10:
            break
    # with open("all_data/test_set_tfidf.csv","a") as f:
    #     f.write(text)
    return results,query,similarities,links


def bert_similarities(query,df,bert_embedings,sbert_model,tfidf_sims,results_classifier,doc_ids,top_k=5):
    # It encodes the query and calculates similarity with sents
    # returns the articles with the most relevant sentences
    # Then it uses the classifier to predict the relevance
    # and excludes non relevant documents
    # Returns list of links sentences and similarities of the n most similar docs
    bert_query = sbert_model.encode(query)
    query_sm = csr_matrix(bert_query).toarray()
    bert_embedings = csr_matrix(bert_embedings).toarray()
    sim_sparse = cosine_similarity(bert_embedings, query_sm) # calc similarity
    most_similar_doc_indices = np.argsort(sim_sparse, axis=0)[::-1]#[:-top_k-1:-1]
    best_articles = [article for article in most_similar_doc_indices.flatten()]
    df['similarities'] = sim_sparse
    df.reset_index(inplace=True)
    output = df[df.index.isin(best_articles)]
    output = output.sort_values(by='similarities',ascending=False)
    print("BERT DOC RANKING with duplicates")
    print(output)
    output.drop_duplicates('url',inplace=True,keep='first')
    output = output[:top_k]
    output = output.merge(tfidf_sims,on='url',how='left')
    output['query'] = query
    output.to_csv('classifier_dataset.csv', mode='a', sep=';', header=False)
    output = output.merge(doc_ids, how='left', on='url')
    
    output['tfidf_similarity'] =output['tfidf_similarity'].fillna((output['tfidf_similarity'].mean()))
    output['similarities'] =output['similarities'].fillna((output['similarities'].mean()))
    output['similarities'] = output['similarities'].astype("float32")
    output['tfidf_similarity'] = output['tfidf_similarity'].astype("float32")
    output['Doc_ID'] = output['Doc_ID'].astype("int")
    print("BERT DOC RANKING unique")
    print(output)
    data = results_classifier.predict(output[["Doc_ID","similarities","tfidf_similarity"]].values)
    output['relevance'] = data
    
    print("articles that will be excluded:")
    print(output[output['relevance'] == 0])
    output = output[output['relevance'] != 0]
    # output = output[output['similarities']>0.7]
    sentences = output['sentences'].tolist()
    sents = []
    for sent in sentences:
        if len(sent)>150:
            sent = sent[:150] + "..."
        sents.append(sent)
    similarities = output['similarities'].tolist()
    links = output['url'].tolist()
    return links, sents, similarities




def main_bert(query,documents,bert_embedings,sbert_model,tfidf_sims,results_classifier,doc_ids):
    # this is the function that flask app calls to calculate similarity using bert
    # It uses the function bert_similarities (for organisation puproses)
    # returns results(dict of sentences[value] and similarities[value] for each like[key])
    links,sentences,similarities = bert_similarities(query,documents,bert_embedings,sbert_model,tfidf_sims,results_classifier,doc_ids)
    similarities = [round(sim,2) for sim in similarities]
    results = dict(zip(links,list(zip(sentences,similarities))))
    text = ""
    for link,data in results.items():
        sentence = data[0]
        similarity = data[1]
        text = text + f'{query};{sentence};{link};{similarity}\n'
    with open("all_data/test_set_bert.csv.csv","a") as f:
        f.write(text)
    f.close()

    return results,query,similarities

def bert_similarities_umap(query,df,bert_embedings,sbert_model,reducer,tfidf_sims,top_k=5):
    # Variation of simple bert function but uses embedings with less dimensions
    # Non in use and the classifier has is not present here
    # Also recommend not to use this function, provides poor results
    bert_query = [sbert_model.encode(query)]
    bert_query = reducer.transform(bert_query)
    query_sm = csr_matrix(bert_query).toarray()
    bert_embedings = csr_matrix(bert_embedings).toarray()
    sim_sparse = cosine_similarity(bert_embedings, query_sm)
    most_similar_doc_indices = np.argsort(sim_sparse, axis=0)#[:-top_k-1:-1]
    best_articles = [article for article in most_similar_doc_indices.flatten()]
    df['similarities'] = sim_sparse
    df.reset_index(inplace=True)
    output = df[df.index.isin(best_articles)]
    
    output = output.sort_values(by='similarities',ascending=False)
    output.drop_duplicates('url',inplace=True,keep='first')

    output = output[:top_k]
    output = output.merge(tfidf_sims,on='url',how='left')
    sentences = output['sentences'].tolist()
    sents = []
    for sent in sentences:
        if len(sent)>150:
            sent = sent[:150] + "..."
        sents.append(sent)
    similarities = output['similarities'].tolist()
    links = output['url'].tolist()
    return links, sents, similarities

def main_bert_umap(query,documents,bert_embedings,sbert_model,reducer,tfidf_sims):
    # Variation of simple bert function but uses embedings with less dimensions
    # Non in use and the classifier has is not present here
    # Also recommend not to use this function, provides poor results
    links,sentences,similarities = bert_similarities_umap(query,documents,bert_embedings,sbert_model,reducer,tfidf_sims)
    similarities = [round(sim,2) for sim in similarities]
    results = dict(zip(links,list(zip(sentences,similarities))))


    text = ""
    for link,data in results.items():
        sentence = data[0]
        similarity = data[1]
        text = text + f'{query};{sentence};{link};{similarity}\n'
    with open("all_data/test_set_bert_umap350d.csv","a") as f:
        f.write(text)
    f.close()

    return results,query,similarities

if __name__ == "__main__":

    path = "./corpus/scrapped_articles_new.xml"
    query = "phone with large ram"

    ir_tfidf(path,query)
    exit(-1)