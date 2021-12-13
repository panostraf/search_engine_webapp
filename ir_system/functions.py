import re
from spellchecker import SpellChecker
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize


# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')


# from sentence_transformers import SentenceTransformer
# sbert_model = SentenceTransformer('bert-base-nli-max-tokens')
# sbert_model = SentenceTransformer()

def replace_dot(content):
    # takes as input an article and returns the same content,
    # but with one space character added if needed between sentences
    # in order to help nltk.tokenize produce better results
    criterion = r'[A-Za-z0-9][A-Za-z0-9][\.\!\?][A-Za-z][A-Za-z]*'
    
    outcome = re.findall(criterion,content)
    if len(outcome) > 0:
        for item in outcome:
            new_item1 = item.replace( ".", ". " )
            new_item2 = item.replace( "?", "? " )
            new_item3 = item.replace( "!", "! " )
            new_item4 = item.replace( "-", " " )
            new_item5 = item.replace( "_", " " )
            new_item6 = item.replace("  "," ")

            content = content.replace(item, new_item1)
            content = content.replace(item, new_item2)
            content = content.replace(item, new_item3)
            content = content.replace(item, new_item4)
            content = content.replace(item, new_item5)
    return content


def replace_contractions(content):
    # takes as input a string and returns the 
    # string with replaced contractions
    replacements = {
    "won't":"will not",
    "can't":"can not",
    "i'm":"i am",
    "he's":"he is",
    "she's":"she is",
    "it's":"it is",
    "that's":"that is",
    "here's":"here is",
    "there's":"there is",
    "i've": "I have",
    "won't": "will not",
    "could've": "could have",
    "wouldn't": "would not",
    "it's": "It is",
    "i'll": "I will",
    "haven't": "have not",
    "can't": "can not",
    "that's": "that is",
    "they'r": "they are",
    "doesn't": "does not",
    "don't": "do not",
    "i'm": "I am",
    "story's": "story s",
    "souldn't've": "sould not have",
    "n't":" not",
    "n't":" not",
    "'ll":" will",
    "'ve": " have",
    "'re":" are",
    "'s":" s",
    "’s":" s",
    "'":" "
        
    }
    for key,value in replacements.items():
        content = content.replace(key,value)
    return content 



def spell_checking(text,extra_corpus):
    # spell checking 
    # inputs the query (text) and bag of words (extra_corpus)
    # returns a corrected query
    spell = SpellChecker(distance=3,)
    spell.word_frequency.load_words(extra_corpus)
    corrected=[]
    tokens=word_tokenize(text)
    spell.word_frequency.remove_words(['samson','hawe'])
    for word in tokens:
        if spell[word] == False:
            corrected.append(spell.correction(word))
            print("correcting query")
            print(word,"---",spell.correction(word))
        else:
            corrected.append(word)
        #print(spell.candidates(word))
    strcorrected = ' '.join(map(str, corrected))
    print("Output of spell checking:",strcorrected)
    return strcorrected


def query_preprocess(text,stops):
    # remove stopwords from query
    # returns proccesed query
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stops]
    filtered_sentence = (" ").join(tokens_without_sw)
    return filtered_sentence



replacements = {
"television":["tv","smart-tv","smart tv","television","tvs","televisions"],
"tv":["television","smart-tv","smart tv","television","tvs","televisions"],
"televisions":["television","smart-tv","smart tv","television","tvs","televisions","tv"],
"smartphone":["phone","cell phones","cell phone","smart phone","smart-phone","android phone"],
"phone":["smartphone","cell phones","cell phone","smart phone","smart-phone","android phone"],
"cellphone":["smartphone","cell phones","cell phone","smart phone","smart-phone","android phone"],
"mobile phone":["smartphone","cell phones","cell phone","smart phone","smart-phone","android phone"],
"mobile phones":["smartphone","cell phones","cell phone","smart phone","smart-phone","android phone"],
"applications": ["apps","applications","app"],
"apps": ["applications","app",'application'],
"app":["applications","app",'application'],
"application":["applications","app",'application'],
"laptop":["notebook"],
"notebook":["laptop"],
"laptops":["notebooks"],
"notebooks":["laptops"],
"os":["operation system"],
"operation system":["os"],
"hdd":["hard drive disk"],
"hard drive disk":["hdd"],
"ssd":["solid-state drive","ssd","solid state drive"],
"solid-state drive":["ssd"],
"modem":["router"],
"router":["modem"],
"modems":["routers"],
"routers":["modems"],
"vr":["virtual reality"],
"virtual reality":["vr"],
"wireless":["bluetooth"],
"computer":["pc"],
"pc":["desktop"],
"motherboard":["mobo"],
"mobo":["motherboard"],
"processor":["cpu"],
"cpu":["processor"],
"graphic card":["gpu"],
"graphics card":["gpu"],
"gpu":"graphics card",
"navigator":["gps","navigation system","geo loc system","navigation"],
"gps":["navigator","navigation system","geo loc system","navigation"],
"smartwatch":["smart watch","watch os","smart-watch"],
"smart watch":["smartwatch"],
"handsfree":["earphones","wireless headphones"],
"camera":["videocamera","ip camera"],
"desktop":["pc"],
"printer":["3d printer","printers","scanner","scanners"],
"scanner":["3d printer","printer","printers","scanner","scanners"],
"3d printer":["3d-printer","scanner","printers","scanner","scanners"]

}



def synonyms_production_old(query):
    query_words = word_tokenize(query)
    queries = []
    for word in query_words:
        
        if word in replacements.keys():
            queries.append(re.sub(word,replacements[word],query))
            continue
        
        
    queries.append(query)
    return queries

def synonyms_production(query):
    # Query expansion
    # creates a list of queries by replacacing every word it founds
    # on the custom replacements keys with every value
    query_words = word_tokenize(query)
    queries = []
    for word in query_words:
        if word in replacements.keys():
            for item in replacements[word]:
                queries.append(re.sub(word,item,query))
                continue
    queries.append(query)
    return queries
    






class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]



if __name__=="__main__":
    query = "the best"
    print(synonyms_production(query))