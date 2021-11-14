from flask import Flask, render_template, request
from ir_system import parser
# from ir_system import functions
# from ir_system.functions import load_docs
from ir_system import functions

app = Flask(__name__)

path_file = "data/scrapped_articles_new.xml"
documents = parser.load_docs(path_file)


@app.route('/')
def hello():
    name = "thodoris"
    return render_template("index.html",value = name)

@app.route('/search')
def search():
    name = "thodoris"
    return render_template("search_bar.html",value = name)

@app.route('/search', methods=['POST',"GET"])
def search_bar():
    name = "thodoris"
    query = request.form['q']
    
    results = parser.ir_tfidf(documents,query)
    return render_template("search_results.html",mylist = results)

if __name__=="__main__":
    app.run(debug=True)#,host="0.0.0.0",port="8080")