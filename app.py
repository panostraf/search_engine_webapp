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
from nltk.stem import WordNetLemmatizer
from all_data.preprocessing import LemmaTokenizer
# from ir_system.functions import LemmaTokenizer
print("starting")
# Initializing flask object
app = Flask(__name__)


# Load in memory all pretrained models
# and all datasets since there is not a database to link them
# This helps in computations, since we avoid calculating tfidf 
# and bert embedings while the user provides a query

path_file = "data/scrapped_articles_new.xml"
documents = pd.read_csv('all_data/big_df.csv',sep=';')
documents.reset_index(inplace=True)
documents.rename(columns={"level_0":"Doc_ID"},inplace=True)
doc_ids = documents[['Doc_ID',"url"]]
# print(documents)
# documents = documents.rename(columns={"url": "URL","level_0":"Doc_ID"},inplace=True)
# documents_loaded = ""

X = scipy.sparse.load_npz('all_data/tfidf_vectors_removed_stops_lema.npz')
vectors = pickle.load(open("all_data/tfidf_model_removed_stops_lema.pk", 'rb'))
# nltk.download('stopwords')
stops = stopwords.words()
sbert = sbert_model = SentenceTransformer('bert-base-nli-max-tokens')
bert_embedings = sparse.load_npz('all_data/bert_embedings.npz')
# bert_embedings = sparse.load_npz('all_data/cp_embedings350.npz')
document_sents = pd.read_csv('all_data/big_df_sents.csv',sep=';')
corpus_list = list(pickle.load(open("all_data/words.pkl","rb")))

results_classifier = pickle.load(open("all_data/random_forest_classifier.pkl", 'rb'))

pretrained_queries = pd.read_excel("all_data/pretrained_queries.xlsx")
pretrained_queries = pretrained_queries[pretrained_queries['Relevance'] != 0]
queries = [q.lower().strip() for q in pretrained_queries['Query']]




# reducer = umap.UMAP(n_components=10,random_state=42)
# # this is the bert embedings with umap reduced to 350 dimensions
# with open("all_data/reducer350.pickle","rb") as f:
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
    query_for_bert = query
    query_visual = query
    query = functions.query_preprocess(query,stops)
    other_q = functions.spell_checking(query,corpus_list)
    

    btn = "on"
    # check if user pressed the "do you mean button"
    # If true we need to change the query to the spell checked
    if request.method == "POST":
        if request.form.get("do_you_mean"):
            btn = "off"
            query = request.form['do_you_mean']
            query_visual = query
        else:
            if query != other_q: 
                btn = "on"
            else:
                btn = "off"
    else:
        if query != other_q: 
            btn = "on"
        else:
            btn = "off"

    for q in queries:
        print(q)
    if query.lower().strip() in queries:
        temp = pretrained_queries[pretrained_queries['Query'] == query]
        links = temp['URL'].tolist()
        similarities = temp['Similarity Bert'].tolist()
        sentences = temp['Title'].tolist()
        sents = []
        for sent in sentences:
            if len(sent)>150:
                sent = sent[:150] + "..."
            sents.append(sent)

        similarities = [round(sim,2) for sim in similarities]
        results = dict(zip(links,list(zip(sentences,similarities))))

    else:
        # Run this for TFIDF ranking only
        results,query,similarities,links = parser.main(str(query).lower(),documents,X,vectors)
        
        # to run using only bert comment the line above and the rest of the lines in the if statement
        # Uncomment the next line
        # results_bert,query,similarities_bert = parser.main_bert(query,filtered_sents,filtered_embedings,sbert_model,tfidf_df,results_classifier,doc_ids)

        # bert filtered by tfidf (Hierarchical method)
        tfidf_df = pd.DataFrame(list(zip(links, similarities)),
                columns =['url', 'tfidf_similarity'])

        filtered_embedings = bert_embedings.toarray()
        filtered_embedings = filtered_embedings[document_sents[document_sents['url'].isin(links)].index]
        
        filtered_sents = document_sents[document_sents['url'].isin(links)]
        results_bert,query,similarities_bert = parser.main_bert(query_for_bert,filtered_sents,filtered_embedings,sbert_model,tfidf_df,results_classifier,doc_ids)

        results = results_bert
        similarities = similarities_bert
        
        

    b= time.time()
    total_time = str(round(b-a,2))

    if len(similarities) >= 1:
        # render this template if there are results to show
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
        # render this template if no results returned
        return render_template("search_bar_no_results_found.html",
        other_query = other_q,
        btn = btn,
        query = query,
        n_articles= len(documents))
    



if __name__=="__main__":
    import cProfile
    app.run(debug=True)

    # app.run(debug=True,host="0.0.0.0",port="8080") 
    # using this all other devices in the same network can access the website