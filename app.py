from flask import Flask, render_template, request
from numpy.lib.function_base import average
from ir_system import parser
from ir_system import functions
import scipy
import pickle
import pandas as pd
import numpy as np

import ir_system

app = Flask(__name__)

# read all files to avoid calculations later on
path_file = "data/scrapped_articles_new.xml"
# documents = parser.load_docs(path_file)
documents = pd.read_csv('all_data/all_articles.csv',sep=';')
documents_loaded = ""
X = scipy.sparse.load_npz('data/sparse_matrix.npz')
vectors = pickle.load(open("vectorizer.pk", 'rb'))
print("files have been loaded")



@app.route('/')
@app.route('/search')
def search():
    return render_template("search_bar.html")


@app.route('/', methods=['POST',"GET"])
@app.route('/search', methods=['POST',"GET"])
def search_bar():
    query = request.form['q']
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
    results,query,similarities,average_similarity = parser.main(query,documents,X,vectors)
    
    return render_template("search_results.html",
                            btn = btn, 
                            results = results,
                            query = query, 
                            similarity = similarities, 
                            av_sim = average_similarity,
                            other_query=other_q)
    



if __name__=="__main__":
    import cProfile
    app.run(debug=True)

    # app.run(debug=True)#,host="0.0.0.0",port="8080")