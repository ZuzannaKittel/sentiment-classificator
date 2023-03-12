import pandas as pd
import tools
import gensim

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors

# Import database
data = pd.read_csv('słownik_anotacji_emocjonalnej.csv')

# Nice order of the data | insert into a table
data = data.iloc[:10,:]
print(data)

# Extraction of the examples (column 7)
examples = data[data.columns[7]]
sentiments = data[data.columns[6]]
print(examples)


examples_dict = tools.processDataDict(examples)
examples_list = tools.processDataList(examples)
print(examples_list)

sentiments_dict = tools.processDataDict(sentiments)
sentiments_list = tools.processDataList(sentiments)

"""
def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = [value , dict_1[key]]
   return dict_3

examples_and_sentiments = mergeDictionary(examples_dict, sentiments_dict)
print(examples_and_sentiments)

print(examples_dict)
#print(sentiments_dict)
"""

"""
# List the tokens including their lemmas and POS tags
doc = pl("Granice mojego języka oznaczają granice mojego świata") # ~Wittgenstein
for token in doc:
    print(token.text, token.lemma_, token.tag_)
"""

# Initialise the TfidfVectorizer
tfidf = TfidfVectorizer()

# Vectorise the text column of the data
vectorized_examples = tfidf.fit_transform(examples_list)
print(vectorized_examples)
