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
    
    
    

    if request.method == "POST":
        if request.form.get("do_you_mean"):

            query = request.form['do_you_mean']

            links,titles,similarities = parser.ir_tfidf(documents,query,X,vectors)
            similarities = [round(sim,2) for sim in similarities]

            results = dict(zip(links,list(zip(titles,similarities))))

            average_similarity = round(np.mean(similarities),2)

            text = ""
            for link,data in results.items():
                title = data[0]
                similarity = data[1]
                text = text + f'{query};{title};{link};{similarity}\n'
            with open("train_dataset.csv","a") as f:
                f.write(text)

            return render_template("search_results_no_button.html",results = results,query = query, similarity = similarities, av_sim = average_similarity)

    
    
    links,titles,similarities = parser.ir_tfidf(documents,query,X,vectors)
    similarities = [round(sim,2) for sim in similarities]

    results = dict(zip(links,list(zip(titles,similarities))))

    average_similarity = round(np.mean(similarities),2)

    text = ""
    for link,data in results.items():
        title = data[0]
        similarity = data[1]
        text = text + f'{query};{title};{link};{similarity}\n'
    with open("train_dataset.csv","a") as f:
        f.write(text)
    
    if query != other_q:  
        return render_template("search_results.html",results = results,query = query, similarity = similarities, av_sim = average_similarity,other_query=other_q)
    else:
        return render_template("search_results_no_button.html",results = results,query = query, similarity = similarities, av_sim = average_similarity)
    
        


if __name__=="__main__":
    import cProfile
    app.run(debug=True)

    # app.run(debug=True)#,host="0.0.0.0",port="8080")