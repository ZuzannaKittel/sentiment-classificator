import re
import spacy

from spacy.tokenizer import Tokenizer

from spacy.lang.pl import Polish
pl = spacy.load('pl_core_news_sm')

# Process text
def process_text(text):

    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', str(text))

    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', str(text))

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', str(text))

    # remove "nan" (so records without examples)
    # text = re.sub(r'nan', '', str(text))

    # instantiate tokenizer class
    tokenizer = Tokenizer(pl.vocab)

    # tokenize text
    text_tokens = tokenizer(text)

    # import the polish stopwords from spacy
    stopwords = pl.Defaults.stop_words 

    text_clean = []
    
    for word in text_tokens: # go through every word in your tokens list
        if (word not in stopwords):  # remove stopwords
            text_clean.append(word)

    #print(text_clean)

    texts_lem = ''

    doc = pl(str(text_clean))

    texts_lem = ' '.join(token.lemma_ for token in doc)

    #print(texts_lem)

    return texts_lem

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


def deleteRecordsWithNeutralSentiment(dict):
    

    return dict
