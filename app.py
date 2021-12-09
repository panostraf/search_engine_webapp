from flask import Flask, render_template, request
from numpy.lib.function_base import average
from pandas.core.frame import DataFrame
from ir_system import parser
from ir_system import functions
import scipy
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import ir_system
from sentence_transformers import SentenceTransformer
from scipy import sparse
import umap
import pickle
import time

app = Flask(__name__)

# read all files to avoid calculations later on
path_file = "data/scrapped_articles_new.xml"
# documents = parser.load_docs(path_file)
documents = pd.read_csv('all_data/big_df.csv',sep=';')
documents_loaded = ""
X = scipy.sparse.load_npz('data/tfidf_vectors_removed_stops.npz')
vectors = pickle.load(open("data/tfidf_model_removed_stops.pk", 'rb'))
print("files have been loaded")
# nltk.download('stopwords')
stops = stopwords.words()
sbert = sbert_model = SentenceTransformer('bert-base-nli-max-tokens')
bert_embedings = sparse.load_npz('all_data/bert_embedings.npz')
# bert_embedings = sparse.load_npz('all_data/cp_embedings.npz')
document_sents = pd.read_csv('all_data/big_df_sents.csv',sep=';')
corpus_list = list(pickle.load(open("all_data/words.pkl","rb")))





reducer = umap.UMAP(n_components=10,random_state=42)
# with open("all_data/reducer_big.pickle","rb") as f:
#     reducer = pickle.load(f)
# f.close()

print("\n\n\n")
print("packages loaded")

@app.route('/')
@app.route('/search')
def search():
    return render_template("search_bar.html")


@app.route('/', methods=['POST',"GET"])
@app.route('/search', methods=['POST',"GET"])
def search_bar():
    a = time.time()
    query = request.form['q']
    query_visual = query
    query = functions.query_preprocess(query,stops)
    other_q = functions.spell_checking(query,corpus_list)
    

    btn = "on"

    if request.method == "POST":
        if request.form.get("do_you_mean"):
            # print("\n\n\n ----------------  1")
            btn = "off"
            query = request.form['do_you_mean']
            query_visual = query
        else:
            if query != other_q: 
                # print("\n\n\n ----------------  2")
                btn = "on"
            else:
                # print("\n\n\n ----------------  3")
                btn = "off"
    else:
        if query != other_q: 
            # print("\n\n\n ----------------  2")
            btn = "on"
        else:
            # print("\n\n\n ----------------  3")
            btn = "off"

    # Run this for TFIDF ranking only
    results,query,similarities,links = parser.main(str(query).lower(),documents,X,vectors)
    
    #only bert
    # results_bert,query,similarities_bert = parser.main_bert(query,document_sents,bert_embedings.toarray(),sbert_model,reducer,tfidf_df)



    # bert filtered by tfidf (Hierarchical method)
    tfidf_df = pd.DataFrame(list(zip(links, similarities)),
               columns =['url', 'tfidf_similarity'])

    
    
    filtered_embedings = bert_embedings.toarray()
    filtered_embedings = filtered_embedings[document_sents[document_sents['url'].isin(links)].index]
    
    filtered_sents = document_sents[document_sents['url'].isin(links)]
    results_bert,query,similarities_bert = parser.main_bert(query,filtered_sents,filtered_embedings,sbert_model,reducer,tfidf_df)

    results = results_bert
    similarities = similarities_bert
    b= time.time()
    total_time = str(round(b-a,2))



    if len(similarities) > 1:
        return render_template("search_results.html",
                                btn = btn, 
                                results = results,
                                query = query_visual, 
                                similarity = similarities, 
                                # av_sim = average_similarity,
                                other_query=other_q,
                                retreival_time = total_time,
                                n_results=str(len(similarities)),
                                n_articles= len(documents))
    else:
        return render_template("search_bar_no_results_found.html",
        other_query = other_q,
        btn = btn,
        query = query,
        n_articles= len(documents))
    



if __name__=="__main__":
    import cProfile
    app.run(debug=True)

    # app.run(debug=True)#,host="0.0.0.0",port="8080")
    # app.run(debug=True,host="0.0.0.0",port="8080")