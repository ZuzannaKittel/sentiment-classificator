import re
import spacy

from spacy.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')

from spacy.lang.pl import Polish
from spacy.lang.en import English
pl = spacy.load('pl_core_news_sm')
nlp = English()

# Process text
def process_text(text):

    print (text)
    
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', str(text))

    # tokenize text
    text_tokens = word_tokenize(text)
    print("text_tokens: ")
    
    print(text_tokens)

    # import the polish stopwords from spacy
    stopwords = pl.Defaults.stop_words 

    text_clean = []
    
    for word in text_tokens: # go through every word in your tokens list
        if (word not in stopwords):  # remove stopwords
            text_clean.append(word)

    #print(text_clean)

    text_lem = ''
    doc = pl(str(text_clean))
    #text_lem = ''.join(token.lemma_ for token in doc)
    for token in doc:
        text_lem += ''+token.lemma_  # append to the phrase

    #print("text_lem: ")
    #print(text_lem)

    return text_lem

# Use process_text on the whole data, row by row
# Ignoring the "nan" and "NaN" records
# Creates a dictionary
def processDataDict(data):
    counter = 0
    result = {}
    for row in data:
        if str(row) != "nan" and str(row) != "NaN":
            result.update({counter :process_text(row)})
            counter = counter + 1
    return result

# Use process_text on the whole data, row by row
# Ignoring the "nan" and "NaN" records
# Creates a list
def processDataList(data):
    result = []
    for row in data:
        if str(row) != "nan" and str(row) != "NaN":
            result.append(process_text(row))
    return result

# Use process_text on the whole data, row by row
# Ignoring the "nan" and "NaN" records
# Creates a dictionary
def processSentimentsDict(data):
    counter = 0
    result = {}
    for row in data:
        if str(row) != "nan" and str(row) != "NaN":
            if row == '- s' or row == '- m':
                row = 0
            if row == '+ s' or row == '+ m':
                row = 1
            counter = counter + 1
            result.append(row)
    return result

# Ignoring the "nan" and "NaN" records
# Creates a list
def processSentimentsList(data):
    result = []
    for row in data:
        if str(row) != "nan" and str(row) != "NaN":
            if row == '- s' or row == '- m':
                row = 0
            if row == '+ s' or row == '+ m':
                row = 1
            result.append(row)
    return result

def mergeDictionaries(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = [value , dict_1[key]]
   return dict_3

