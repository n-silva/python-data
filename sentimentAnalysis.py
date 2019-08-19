import json
import re
import csv
import string

from sklearn import linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from nltk import TweetTokenizer, PorterStemmer, RegexpTokenizer
from nltk.sentiment.util import mark_negation

from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from twokenize import tokenizeRawTweetText

url = 'data/Tweets_EN_sentiment.json'
emolex = 'data/NCR-lexicon.csv'

#
text_data = []
category = []
data = []

english_words  = []
positive_vocab = []
negative_vocab = []
negation_pos = []
negation_neg = []

dict_word_polarity = {}

#cleaning function
def clean_text(text):
    # Happy Emoticons
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'
    ])

    # Sad Emoticons
    emoticons_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('
    ])
    # all emoticons (happy + sad)
    emoticons = emoticons_happy.union(emoticons_sad)

    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)

    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)

    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)

    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(text)
    stemmer = PorterStemmer()

    text_clean = []
    for word in tweet_tokens:
        if (word not in emoticons and  # remove emoticons
            word not in string.punctuation):  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            text_clean.append(stem_word)

    return ' '.join(text_clean)

#read file
def readfile(file):
    dataset = open(file, encoding="utf-8")
    for i, line in enumerate(dataset.readlines()):
        review = json.loads(line)
        text = review["text"]
        text_data.append(text)
        category.append(review["class"])
        data.append((text,review["class"]))
    dataset.close()

#read dict lexicon
def read_dict_lexicon(file):
    with open(emolex, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for i, row in enumerate(reader):
            word = row["English"]
            if (row["Positive"] == '1'):
                positive_vocab.append(word)
                dict_word_polarity[word] = 1
            elif (row["Negative"] == '1'):
                negative_vocab.append(word)
                dict_word_polarity[word] = -1

readfile(url)
read_dict_lexicon(emolex)
len_doc = len(text_data) - 1

#remove stop_words function
def stop_words(text):
    stop = set(stopwords.words('english'))
    return " ".join([i for i in text.lower().split() if i not in stop])

#remove pontuation function
def remove_pontuation(text):
    exclude = set(string.punctuation)
    return ' '.join(ch for ch in text if ch not in exclude)


lex_pos = []
lex_neg = []

#sentiment analysis with textblob
def sentiment_analysis_textblob(clean=False):
    pos_correct = 0
    pos_count = 0
    neg_correct = 0
    neg_count = 0
    for i in range(0,len_doc):
        sentence = clean_text(text_data[i]) if clean == True else text_data[i]
        if category[i] == 'pos':
            if TextBlob(sentence).sentiment.polarity >= 0:
                pos_correct += 1
            pos_count += 1
        else:
            if TextBlob(sentence).sentiment.polarity <= 0:
                neg_correct += 1
            neg_count += 1
    # =============================================================================================
    # - Count positive/negative polarity
    # - Calculate positive accurary by: correct positive / total positive
    # - Calculate negative accurary by: correct negative / total negative
    print("#TextBlob sentiment clean text: ",clean)
    print("#================================================================")
    print("Positive accuracy = {:.2f}% of {} pos tweets".format(pos_correct / pos_count * 100.0, pos_count))
    print("Negative accuracy = {:.2f}% of {} neg tweets".format(neg_correct / neg_count * 100.0, neg_count))
    print("#================================================================")

#lexicon function
def lexicon (clean=True):
    pos_correct = 0
    pos_count = 0
    neg_correct = 0
    neg_count = 0
    neg_positive = 0
    pos_negative = 0
    for i in range(0, len_doc):
            sentence = clean_text(text_data[i]) if clean == True else text_data[i]
            if category[i] == 'pos':
                for word in sentence.split():
                    pos_correct += 1 if word in positive_vocab else 0
                pos_count += 1
            else:
                for word in sentence.split():
                    neg_correct += 1 if word in negative_vocab else 0
                neg_count += 1


    print("#Lexicon sentiment analysis #clean text: ", clean)
    print("#================================================================")
    print("Positive accuracy = {:.2f}% of {} pos tweets".format((pos_correct + pos_negative) / pos_count * 100.0, pos_count))
    print("Negative accuracy = {:.2f}% of {} neg tweets".format((neg_correct + neg_positive) / neg_count * 100.0, neg_count))
    print("#================================================================")


#machine learning model
def algoritmo_aprendizagem_automatica(method='tfidf',clean=False):
    clean_text_data = [clean_text(text) for text in text_data]
    # tokenizer
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    ftidf = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), tokenizer=token.tokenize)
    vectorizer = ftidf if method == 'tfidf' else CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), tokenizer=token.tokenize)
    text_counts =  vectorizer.fit_transform(clean_text_data) if clean else vectorizer.fit_transform(text_data)
    X_train, X_test, y_train, y_test = train_test_split(text_counts, category, test_size=0.2, random_state=1)

    # Classification
    model = LinearSVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print("================================================================")
    print("Accuracy LinearSVC: {:.2f}".format(acc * 100))
    print("Precision:  {:.2f}".format(metrics.precision_score(y_test, y_pred, average="macro") * 100))
    print("================================================================")

    model = linear_model.LogisticRegression(solver='lbfgs', C=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print("================================================================")
    print("Accuracy LogisticRegression: {:.2f}".format(acc * 100))
    print("================================================================")

    model = linear_model.SGDClassifier(max_iter=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print("================================================================")
    print("Accuracy SGDClassifier: {:.2f}".format(acc * 100))
    print("================================================================")

    model = BernoulliNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print("================================================================")
    print("Accuracy BernoulliNB: {:.2f}".format(acc * 100))
    print("================================================================")

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print("================================================================")
    print("Accuracy MultinomialNB: {:.2f}".format(acc * 100))
    print("================================================================")



#without text cleaning
#sentiment_analysis_textblob()

#text cleaning
sentiment_analysis_textblob(clean=True)

#lexicon

lexicon(clean=True)

#method tf-idf by default
algoritmo_aprendizagem_automatica(clean=True)

#algoritmo_aprendizagem_automatica(method='frequency',clean=False)