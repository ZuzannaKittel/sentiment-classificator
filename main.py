import pandas as pd
import tools
import gensim

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# Classifiers
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

# Import database
data = pd.read_csv('słownik_anotacji_emocjonalnej.csv')

# Nice order of the data | insert into a table
data = data.iloc[:7500,:]
print(data)

# Extraction of the examples (column 7)
examples = data[data.columns[7]]
# Extraction of the sentiments (column 6)
sentiments = data[data.columns[6]]
print(sentiments)
#print(examples)

examples_list = []
sentiments_list = []

# Create the lists of examples and sentiments with only positive or negative sentiments
# Ignore the neutral sentiment
examples_list = []
sentiments_list = []
for example, sentiment in zip(data[data.columns[7]], data[data.columns[6]]):
    #(str(sentiment) != "amb" and str(sentiment) != "nan" and str(sentiment) != "NaN" and str(example) != "NaN" and str(example) != "nan")
    if ((str(sentiment) == "- s" or str(sentiment) == "- m" or str(sentiment) == "+ s"  or str(sentiment) == "+ m") and (str(example) != "NaN" and str(example) != "nan")):
        #print(sentiment, example)
        examples_list.append(tools.process_text(example))
        sentiments_list.append(sentiment)


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
#print(vectorized_examples)

sentiments_list = tools.processSentimentsList(sentiments_list)
print(sentiments_list)

"""
# Word2Vec
sentences = [['i', 'like', 'apple', 'pie', 'for', 'dessert'],
           ['i', 'dont', 'drive', 'fast', 'cars'],
           ['data', 'science', 'is', 'fun'],
           ['chocolate', 'is', 'my', 'favorite'],
           ['my', 'favorite', 'movie', 'is', 'predator']]
w2v = Word2Vec(examples_list, min_count = 1,
                              vector_size = 100, window = 5)
print(w2v)

wordsInW2V = list(w2v.wv.key_to_index)
print(wordsInW2V)
"""

X_train, X_test, y_train, y_test = train_test_split(vectorized_examples, sentiments_list, test_size=0.3, random_state=123)
classifier = BernoulliNB().fit(X_train, y_train)
predicted = classifier.predict(X_test)
print("BernoulliNB Accuracy: ", metrics.accuracy_score(y_test,predicted))
print("BernoulliNB Precision: ", metrics.precision_score(y_test,predicted))
print("BernoulliNB Recall: ", metrics.recall_score(y_test,predicted))
print("BernoulliNB F1: ", metrics.f1_score(y_test,predicted))
print("\n\n")

classifier = LogisticRegression(solver='liblinear').fit(X_train, y_train)
predicted = classifier.predict(X_test)
print("LogisticRegression Accuracy: ", metrics.accuracy_score(y_test,predicted))
print("LogisticRegression Precision: ", metrics.precision_score(y_test,predicted))
print("LogisticRegression Recall: ", metrics.recall_score(y_test,predicted))
print("LogisticRegression F1: ", metrics.f1_score(y_test,predicted))
print("\n\n")
