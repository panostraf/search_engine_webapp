from flask import Flask, render_template, request
from numpy.lib.function_base import average
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

app = Flask(__name__)

# read all files to avoid calculations later on
path_file = "data/scrapped_articles_new.xml"
# documents = parser.load_docs(path_file)
documents = pd.read_csv('all_data/big_df.csv',sep=';')
documents_loaded = ""
X = scipy.sparse.load_npz('data/tfidf_vectors_removed_stops.npz')
vectors = pickle.load(open("data/tfidf_model_removed_stops.pk", 'rb'))
print("files have been loaded")
nltk.download('stopwords')
stops = stopwords.words()
sbert = sbert_model = SentenceTransformer('bert-base-nli-max-tokens')
bert_embedings = sparse.load_npz('all_data/bert_embedings.npz')
document_sents = pd.read_csv('all_data/big_df_sents.csv',sep=';')
print("\n\n\n")
print("packages loaded")

@app.route('/')
@app.route('/search')
def search():
    return render_template("search_bar.html")


@app.route('/', methods=['POST',"GET"])
@app.route('/search', methods=['POST',"GET"])
def search_bar():
    query = request.form['q']
    query_visual = query
    query = functions.query_preprocess(query,stops)
    other_q = functions.spell_checking(query)

    btn = "on"

    if request.method == "POST":
        if request.form.get("do_you_mean"):
            print("\n\n\n ----------------  1")
            btn = "off"
            query = request.form['do_you_mean']
        else:
            if query != other_q: 
                print("\n\n\n ----------------  2")
                btn = "on"
            else:
                print("\n\n\n ----------------  3")
                btn = "off"
    else:
        if query != other_q: 
            print("\n\n\n ----------------  2")
            btn = "on"
        else:
            print("\n\n\n ----------------  3")
            btn = "off"

    # Run this for TFIDF ranking only
    # results,query,similarities,average_similarity = parser.main(query,documents,X,vectors)

    # Run this for bert embendings ranking only
    results,query,similarities = parser.main_bert(query,document_sents,bert_embedings,sbert)
    return render_template("search_results.html",
                            btn = btn, 
                            results = results,
                            query = query_visual, 
                            similarity = similarities, 
                            # av_sim = average_similarity,
                            other_query=other_q)
    

# <!-- <a>Average Similarity: {{av_sim * 100}}%</a>  -->

if __name__=="__main__":
    import cProfile
    app.run(debug=True)

    # app.run(debug=True)#,host="0.0.0.0",port="8080")