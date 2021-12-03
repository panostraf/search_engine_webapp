import re
from spellchecker import SpellChecker
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
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



def spell_checking(text):
    spell = SpellChecker()
    # To add words in spell checker
    #spell.word_frequency.load_words(list of tokens of our content)
    corrected=[]
    tokens=word_tokenize(text)
    for word in tokens:
        #print(spell.candidates(word))
        corrected.append(spell.correction(word))
    strcorrected = ' '.join(map(str, corrected))
    return strcorrected


def query_preprocess(text,stops):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stops]
    filtered_sentence = (" ").join(tokens_without_sw)
    return filtered_sentence
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize




# Take misspleled text and return corrected text
# def spell_checking(text):
#     spell = SpellChecker()
#     # To add words in spell checker
#     #spell.word_frequency.load_words(list of tokens of our content)
#     corrected=[]
#     tokens=word_tokenize(text)
#     for word in tokens:
#         #print(spell.candidates(word))
#         corrected.append(spell.correction(word))
#     strcorrected = ' '.join(map(str, corrected))
#     return strcorrected


# def bert_embendings(content):
    
#     # Input sentence
#     # output vectorized sentence [array with bert embendings]  
#     # print(content) 
#     sentence_embeddings = sbert_model.encode(content)
#     # print(sentence_embeddings)
#     return sentence_embeddings