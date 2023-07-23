import pandas as pd
import tools
import gensim
import spacy
import numpy

pl = spacy.load('pl_core_news_sm')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec, FastText
from spacy.pipeline import Tok2Vec
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import plotly.graph_objects as go

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# Classifiers
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

# Import database
data = pd.read_csv('słownik_anotacji_emocjonalnej.csv')

# Nice order of the data | insert into a table
data = data.iloc[:100000,:]
print(data)

# Extraction of the examples (column 7)
examples = data[data.columns[7]]
# Extraction of the sentiments (column 6)
sentiments = data[data.columns[6]]
#print(sentiments)
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
#print(examples_list)

# Vectorise the text examples
def textRepresentation():
    
    # TF-IDF representation
    # Initialise the TfidfVectorizer
    tfidf = TfidfVectorizer()
    vectorized_examples = tfidf.fit_transform(examples_list)
    #print(vectorized_examples)
    #print(vectorized_examples.shape)

    """# Bag of Words representation
    BoW = CountVectorizer()
    vectorized_examples = BoW.fit_transform(examples_list)
    #print(vectorized_examples)
    #print(vectorized_examples.shape)"""
    
    return vectorized_examples

vectorized_examples = textRepresentation()

# Initialise the Tok2Vec vectorizer from spacy
#tok2vec = Tok2Vec(pl.vocab, DEFAULT_TOK2VEC_MODEL)
#doc = "This is a sentence."
# This usually happens under the hood
#processed = tok2vec(examples_list)
#print(processed)
"""tok2vec = Tok2Vec(pl, DEFAULT_TOK2VEC_MODEL)
doc = pl("This is a sentence.")
vectorized_examples = tok2vec(doc)
print(vectorized_examples)"""

"""tok2vec = Tok2Vec(pl, DEFAULT_TOK2VEC_MODEL)
vectorized_examples = tok2vec(examples_list)
print(vectorized_examples)"""

sentiments_list = tools.processSentimentsList(sentiments_list)
#print(sentiments_list)


"""# Word2Vec
sentences = [['i', 'like', 'apple', 'pie', 'for', 'dessert'],
           ['i', 'dont', 'drive', 'fast', 'cars'],
           ['data', 'science', 'is', 'fun'],
           ['chocolate', 'is', 'my', 'favorite'],
           ['my', 'favorite', 'movie', 'is', 'predator']]
w2v = Word2Vec(sentences, min_count = 1,
                              vector_size = 100, window = 5)
print(w2v)

wordsInW2V = list(w2v.wv.vectors)
print(wordsInW2V)"""

#print(examples_list)

"""
word2vec = gensim.models.Word2Vec(examples_list, min_count=1, window=5)
vectorized_examples = list(word2vec.wv.vectors)
"""
"""w2vSkipGram = gensim.models.Word2Vec(examples_list, min_count=1, vector_size=100, window=5, sg=1)
vectorized_examples = list(w2vSkipGram.wv.vectors)
print(vectorized_examples)"""

#vectorized_examples = vectorized_examples.reshape(vectorized_examples.shape[1:])
#vectorized_examples = vectorized_examples.transpose()

def basicTraining():
    # Splitting the data to train and test sets
    x_train, x_test, y_train, y_test = train_test_split(vectorized_examples, sentiments_list, test_size=0.3, random_state=42) # random_state? (bez ustawienia random _state - kneighbors sa lepsze, ale NB gorzej)

    # Usage of classifiers

    # Bernoulli
    classifier = BernoulliNB().fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    print("BernoulliNB Accuracy: ", metrics.accuracy_score(y_test,predicted))
    print("BernoulliNB Precision: ", metrics.precision_score(y_test,predicted))
    print("BernoulliNB Recall: ", metrics.recall_score(y_test,predicted))
    print("BernoulliNB F1: ", metrics.f1_score(y_test,predicted))
    print("\n")

    """classifier = GaussianNB().fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    print("GaussianNB Accuracy: ", metrics.accuracy_score(y_test,predicted))
    print("GaussianNB Precision: ", metrics.precision_score(y_test,predicted))
    print("GaussianNB Recall: ", metrics.recall_score(y_test,predicted))
    print("GaussianNB F1: ", metrics.f1_score(y_test,predicted))
    print("\n")"""

    # Logistic Regression Liblinear
    classifier = LogisticRegression(solver='liblinear').fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    print("LogisticRegression Accuracy: ", metrics.accuracy_score(y_test,predicted))
    print("LogisticRegression Precision: ", metrics.precision_score(y_test,predicted))
    print("LogisticRegression Recall: ", metrics.recall_score(y_test,predicted))
    print("LogisticRegression F1: ", metrics.f1_score(y_test,predicted))
    print("\n")

    # Logistic Regression Sigmoid
    classifier = LogisticRegression(solver='sag').fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    print("LogisticRegression Accuracy: ", metrics.accuracy_score(y_test,predicted))
    print("LogisticRegression Precision: ", metrics.precision_score(y_test,predicted))
    print("LogisticRegression Recall: ", metrics.recall_score(y_test,predicted))
    print("LogisticRegression F1: ", metrics.f1_score(y_test,predicted))
    print("\n")

    # Support Vector Machines
    classifier = svm.SVC(kernel='linear').fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    print("Support Vector Machines Accuracy: ", metrics.accuracy_score(y_test,predicted))
    print("Support Vector Machines Precision: ", metrics.precision_score(y_test,predicted))
    print("Support Vector Machines Recall: ", metrics.recall_score(y_test,predicted))
    print("Support Vector Machines F1: ", metrics.f1_score(y_test,predicted))
    print("\n")

    classifier = svm.SVC(kernel='sigmoid').fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    print("Support Vector Machines Accuracy: ", metrics.accuracy_score(y_test,predicted))
    print("Support Vector Machines Precision: ", metrics.precision_score(y_test,predicted))
    print("Support Vector Machines Recall: ", metrics.recall_score(y_test,predicted))
    print("Support Vector Machines F1: ", metrics.f1_score(y_test,predicted))
    print("\n")

    # K-nearest Neighbors
    # Looks for the k with the best accuracy
    def bestKNeighbors(limit = 30):
        best_k = 1
        best_accuracy = 0
        for k in range(1,limit):
            classifier = KNeighborsClassifier(k, weights='distance').fit(x_train, y_train)
            predicted = classifier.predict(x_test)
            #print(k)
            #print("Kneighbors Accuracy: ", metrics.accuracy_score(y_test,predicted))
            if (metrics.accuracy_score(y_test,predicted) > best_accuracy):
                best_accuracy = metrics.accuracy_score(y_test,predicted)
                best_k = k

        classifier = KNeighborsClassifier(best_k, weights='distance').fit(x_train, y_train)
        predicted = classifier.predict(x_test)
        print("K-nearest Neighbors Accuracy: ", metrics.accuracy_score(y_test,predicted))
        print("K-nearest Neighbors Precision: ", metrics.precision_score(y_test,predicted))
        print("K-nearest Neighbors Recall: ", metrics.recall_score(y_test,predicted))
        print("K-nearest Neighbors F1: ", metrics.f1_score(y_test,predicted))
        print("\n")

    bestKNeighbors()
    
def crossValidaitionTraining(kFolds):
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    # Bernoulli
    classifier = BernoulliNB()
    results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
    print("Bernoulli Accuracy scores: ", results['train_accuracy'])
    print("Bernoulli Mean Training Accuracy: ", results['train_accuracy'].mean())
    print("Bernoulli Mean Accuracy: ", results['test_accuracy'].mean())
    print("Bernoulli Mean Precision: ", results['test_precision'].mean())
    print("Bernoulli Mean Recall: ", results['test_recall'].mean())
    print("Bernoulli Mean F1: ", results['test_f1'].mean())
    print("\n")
    
    # Logistic Regression Liblinear
    classifier = LogisticRegression(solver='liblinear')
    results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
    print("Logistic Regression liblinear Accuracy scores: ", results['train_accuracy'])
    print("Logistic Regression liblinear Mean Training Accuracy: ", results['train_accuracy'].mean())
    print("Logistic Regression liblinear Mean Accuracy: ", results['test_accuracy'].mean())
    print("Logistic Regression liblinear Mean Precision: ", results['test_precision'].mean())
    print("Logistic Regression liblinear Mean Recall: ", results['test_recall'].mean())
    print("Logistic Regression liblinear Mean F1: ", results['test_f1'].mean())
    print("\n")
    
    # Logistic Regression Sigmoid
    classifier = LogisticRegression(solver='sag')
    results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
    print("Logistic Regression sag Accuracy scores: ", results['train_accuracy'])
    print("Logistic Regression sag Mean Training Accuracy: ", results['train_accuracy'].mean())
    print("Logistic Regression sag Mean Accuracy: ", results['test_accuracy'].mean())
    print("Logistic Regression sag Mean Precision: ", results['test_precision'].mean())
    print("Logistic Regression sag Mean Recall: ", results['test_recall'].mean())
    print("Logistic Regression sag Mean F1: ", results['test_f1'].mean())
    print("\n")
    
    # Support Vector Machines Linear
    classifier = svm.SVC(kernel='linear')
    results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
    print("SVM linear Accuracy scores: ", results['train_accuracy'])
    print("SVM linear Mean Training Accuracy: ", results['train_accuracy'].mean())
    print("SVM linear Mean Accuracy: ", results['test_accuracy'].mean())
    print("SVM linear Mean Precision: ", results['test_precision'].mean())
    print("SVM linear Mean Recall: ", results['test_recall'].mean())
    print("SVM linear Mean F1: ", results['test_f1'].mean())
    print("\n")
    
    # Support Vector Machines Sigmoid
    classifier = svm.SVC(kernel='sigmoid')
    results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
    print("SVM sigmoid Accuracy scores: ", results['train_accuracy'])
    print("SVM sigmoid Mean Training Accuracy: ", results['train_accuracy'].mean())
    print("SVM sigmoid Mean Accuracy: ", results['test_accuracy'].mean())
    print("SVM sigmoid Mean Precision: ", results['test_precision'].mean())
    print("SVM sigmoid Mean Recall: ", results['test_recall'].mean())
    print("SVM sigmoid Mean F1: ", results['test_f1'].mean())
    print("\n")
    
    # K-nearest Neighbors
    # Looks for the k with the best accuracy
    def bestKNeighbors(limit = 20):
        best_k = 1
        best_accuracy = 0
        best_result = 0
        for k in range(1,limit):
            classifier = KNeighborsClassifier(k, weights='distance')
            results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
            result = (results['test_accuracy'].mean()+results['test_precision'].mean()+results['test_recall'].mean()+results['test_f1'].mean())/4
            """if (results['test_accuracy'].mean() > best_accuracy):
                best_accuracy = results['test_accuracy'].mean()
                best_k = k"""
            if (result > best_result):
                best_result = (results['test_accuracy'].mean()+results['test_precision'].mean()+results['test_recall'].mean()+results['test_f1'].mean())/4
                best_k = k

        classifier = KNeighborsClassifier(best_k, weights='distance')
        results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
        print("K-nearest Neighbors Accuracy scores: ", results['train_accuracy'])
        print("K-nearest Neighbors Mean Training Accuracy: ", results['train_accuracy'].mean())
        print("K-nearest Neighbors Mean Accuracy: ", results['test_accuracy'].mean())
        print("K-nearest Neighbors Mean Precision: ", results['test_precision'].mean())
        print("K-nearest Neighbors Mean Recall: ", results['test_recall'].mean())
        print("K-nearest Neighbors Mean F1: ", results['test_f1'].mean())
        print("\n")

    bestKNeighbors()
    
def crossValidaitionTrainingMaxScores(kFolds):
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    # Bernoulli
    classifier = BernoulliNB()
    results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
    print("Bernoulli Accuracy scores: ", results['train_accuracy'])
    print("Bernoulli Mean Training Accuracy: ", results['train_accuracy'].max())
    print("Bernoulli Mean Accuracy: ", results['test_accuracy'].max())
    print("Bernoulli Mean Precision: ", results['test_precision'].max())
    print("Bernoulli Mean Recall: ", results['test_recall'].max())
    print("Bernoulli Mean F1: ", results['test_f1'].max())
    print("\n")
    
    # Logistic Regression Liblinear
    classifier = LogisticRegression(solver='liblinear')
    results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
    print("Logistic Regression liblinear Accuracy scores: ", results['train_accuracy'])
    print("Logistic Regression liblinear Mean Training Accuracy: ", results['train_accuracy'].max())
    print("Logistic Regression liblinear Mean Accuracy: ", results['test_accuracy'].max())
    print("Logistic Regression liblinear Mean Precision: ", results['test_precision'].max())
    print("Logistic Regression liblinear Mean Recall: ", results['test_recall'].max())
    print("Logistic Regression liblinear Mean F1: ", results['test_f1'].max())
    print("\n")
    
    # Logistic Regression Sigmoid
    classifier = LogisticRegression(solver='sag')
    results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
    print("Logistic Regression sag Accuracy scores: ", results['train_accuracy'])
    print("Logistic Regression sag Mean Training Accuracy: ", results['train_accuracy'].max())
    print("Logistic Regression sag Mean Accuracy: ", results['test_accuracy'].max())
    print("Logistic Regression sag Mean Precision: ", results['test_precision'].max())
    print("Logistic Regression sag Mean Recall: ", results['test_recall'].max())
    print("Logistic Regression sag Mean F1: ", results['test_f1'].max())
    print("\n")
    
    # Support Vector Machines Linear
    classifier = svm.SVC(kernel='linear')
    results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
    print("SVM linear Accuracy scores: ", results['train_accuracy'])
    print("SVM linear Mean Training Accuracy: ", results['train_accuracy'].max())
    print("SVM linear Mean Accuracy: ", results['test_accuracy'].max())
    print("SVM linear Mean Precision: ", results['test_precision'].max())
    print("SVM linear Mean Recall: ", results['test_recall'].max())
    print("SVM linear Mean F1: ", results['test_f1'].max())
    print("\n")
    
    # Support Vector Machines Sigmoid
    classifier = svm.SVC(kernel='sigmoid')
    results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
    print("SVM sigmoid Accuracy scores: ", results['train_accuracy'])
    print("SVM sigmoid Mean Training Accuracy: ", results['train_accuracy'].max())
    print("SVM sigmoid Mean Accuracy: ", results['test_accuracy'].max())
    print("SVM sigmoid Mean Precision: ", results['test_precision'].max())
    print("SVM sigmoid Mean Recall: ", results['test_recall'].max())
    print("SVM sigmoid Mean F1: ", results['test_f1'].max())
    print("\n")
    
    # K-nearest Neighbors
    # Looks for the k with the best accuracy
    def bestKNeighbors(limit = 20):
        best_k = 1
        best_accuracy = 0
        best_result = 0
        for k in range(1,limit):
            classifier = KNeighborsClassifier(k, weights='distance')
            results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
            result = (results['test_accuracy'].max()+results['test_precision'].max()+results['test_recall'].max()+results['test_f1'].max())/4
            """if (results['test_accuracy'].mean() > best_accuracy):
                best_accuracy = results['test_accuracy'].mean()
                best_k = k"""
            if (result > best_result):
                best_result = (results['test_accuracy'].max()+results['test_precision'].max()+results['test_recall'].max()+results['test_f1'].max())/4
                best_k = k

        classifier = KNeighborsClassifier(best_k, weights='distance')
        results = cross_validate(classifier, vectorized_examples, sentiments_list, cv = kFolds, scoring = scoring, return_train_score=True)
        print("K-nearest Neighbors Accuracy scores: ", results['train_accuracy'])
        print("K-nearest Neighbors Mean Training Accuracy: ", results['train_accuracy'].max())
        print("K-nearest Neighbors Mean Accuracy: ", results['test_accuracy'].max())
        print("K-nearest Neighbors Mean Precision: ", results['test_precision'].max())
        print("K-nearest Neighbors Mean Recall: ", results['test_recall'].max())
        print("K-nearest Neighbors Mean F1: ", results['test_f1'].max())
        print("\n")

    bestKNeighbors()
    
#basicTraining()
#crossValidaitionTraining(kFolds = 3)
#crossValidaitionTraining(kFolds = 5)
#crossValidaitionTraining(kFolds = 10)

crossValidaitionTrainingMaxScores(kFolds = 3)