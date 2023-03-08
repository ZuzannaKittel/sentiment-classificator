import spacy

import re

import pandas as pd

from spacy.tokenizer import Tokenizer
from spacy.lang.pl import Polish
nlp = Polish()

pl = spacy.load('pl_core_news_sm', disable = ['parser','ner'])

# Import database
data = pd.read_csv('słownik_anotacji_emocjonalnej.csv')

# Nice order of the data | insert into a table
data = data.iloc[:10,:]
print(data)

# Extraction of the examples (column 7)
examples = data[data.columns[7]]
print(examples)

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

    # print(stopwords)

    text_clean = []
    
    for word in text_tokens: # go through every word in your tokens list
        if (word not in stopwords):  # remove stopwords
            text_clean.append(word)
            #print(word)

    #print(text_clean)

    texts_lem = ''

    text_clean_string = ''
    text_clean_string = ''.join(str(text_clean))
    #print(text_clean_string)

    doc2 = pl(text_clean_string)

    texts_lem = ' '.join([token.lemma_ for token in doc2])

    return texts_lem

# Use process_text on the whole data, row by row
# Ignoring the "nan" and "NaN" records
def processAllTexts(data):
    counter = 0
    result = {}
    for row in data:
        if str(row) != "nan" and str(row) != "NaN":
            print("nie jest nan")
            result.update({counter :process_text(row)})
            counter = counter + 1
    print(counter)
    return result


examples = processAllTexts(examples)

print(examples)

"""
# List the tokens including their lemmas and POS tags
doc = pl("Granice mojego języka oznaczają granice mojego świata") # ~Wittgenstein
for token in doc:
    print(token.text, token.lemma_, token.tag_)
"""