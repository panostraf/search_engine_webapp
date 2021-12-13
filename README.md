Installation
------------------------------
Download the code from github:
git clone https://github.com/panostraf/search_engine_webapp.git

cd in main directory
mkdir all_data/
make sure that all pretrained models and datasets are in all_data/ directory
download them and unzipp the data in the new directory
download from https://www.googledrive.com/

create a virtual enviroment and activate it:
python -m venv venv
venv\Scripts\activate (for windows)
source venv/bin/activate (mac/linux)
pip install -r requirements.txt
pip install -U sentence-transformers


How to run the code
----------------------------
python app.py
when executed it will start to load in memory all the required documents
It will display the ip:port of the local host to run it on your browser
Open the browser and paste the ip:port
Now you can start making queries


Notes
----------------------------
Inside all_data/ you will also find:

preprocessing.py - standalone file to preprocess the documents and train the classifiers
persicion_recall_tfidf.py - standalone file which is used for our metrics
classifier.ipynb - a jupyter notebook file with the training of our classifier




