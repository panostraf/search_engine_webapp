from flask import Flask, render_template, request
from ir_system import parser
from ir_system import functions
import scipy
import pickle
import pandas as pd

app = Flask(__name__)

path_file = "data/scrapped_articles_new.xml"
# documents = parser.load_docs(path_file)
documents = pd.read_csv('all_articles.csv',sep=';')
documents_loaded = ""
X = scipy.sparse.load_npz('data/sparse_matrix.npz')
vectors = pickle.load(open("vectorizer.pk", 'rb'))
print("files have been loaded")

@app.route('/')
def hello():
    name = "home page"
    return render_template("index.html",value = name)

@app.route('/search')
def search():
    name = "thodoris"
    return render_template("search_bar.html",value = name)

@app.route('/search', methods=['POST',"GET"])
def search_bar():
    query = request.form['q']
    
    links,titles = parser.ir_tfidf(documents,query,X,vectors)
    
    results = dict(zip(links,titles))
    text = ""
    for link,title in results.items():
        text = text + f'{query};{title};{link}\n'
    with open("train_dataset.csv","a") as f:
        f.write(text)
        
    return render_template("search_results.html",results = results,query = query)



if __name__=="__main__":
    import cProfile
    app.run(debug=True)

    # app.run(debug=True)#,host="0.0.0.0",port="8080")