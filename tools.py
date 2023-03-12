import re
import spacy

from spacy.tokenizer import Tokenizer
from spacy.lang.pl import Polish
pl = spacy.load('pl_core_news_sm', disable = ['parser','ner'])

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

    texts_lem = ''
    text_clean_string = ''
    text_clean_string = ''.join(str(text_clean))

    doc2 = pl(text_clean_string)

    texts_lem = ' '.join([token.lemma_ for token in doc2])

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
    print(counter)
    return result

# Use process_text on the whole data, row by row
# Ignoring the "nan" and "NaN" records
# Creates a list
def processDataList(data):
    counter = 0
    result = []
    for row in data:
        if str(row) != "nan" and str(row) != "NaN":
            result.append(process_text(row))
            counter = counter + 1
    print(counter)
    return result