import re

import gensim
import json
import string
from pprint import pprint
import pickle

import nltk
import six
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaMulticore, HdpModel, LsiModel
import subprocess

from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
from nltk import word_tokenize, WordNetLemmatizer
import nltk.corpus, nltk.tag

from gensim.models.ldamodel import  LdaModel
from gensim import similarities

from gensim.models.tfidfmodel import TfidfModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pandas.plotting import table
from matplotlib.patches import Rectangle

import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import numpy as np

json_url = 'data/SFU_Review_Corpus.json'

stop = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemma = WordNetLemmatizer()

def apply_re(doc):
    line = ''
    for sent in doc:
        sent = re.sub('\S*@\S*\s?', ' ', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", " ", sent)  # remove single quotes
        line =line + " " + sent
    return line


def removePunctuation(doc):
    return ''.join(word for word in doc if word not in punctuation)

def min_char(doc,min=3):
   return  ' '.join([i for i in doc if (len(i) > min)])

def cleanAll(doc):
    stop_free = ' '.join([i for i in doc.split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in punctuation)
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def pos_tag(doc, allowed_postags=['NNP','NN']):
    tag_doc = nltk.pos_tag(doc, lang='eng')
    text_out = []
    for word, tag in tag_doc:
        if tag in allowed_postags:
            text_out.append(word)
    return min_char(text_out).split()

def make_bigrams(docs):
    return [bigram_mod[doc] for doc in docs]

doc_clean_gram_postag = []

def make_trigrams(docs):
    docTrigram = []
    for doc in docs:
        trigram = trigram_mod[bigram_mod[doc]]
        doc_clean_gram_postag.append(pos_tag(trigram))
        docTrigram.append(trigram)
    return docTrigram

# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

def format_topics_sentences(model, corpus, texts):
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row_list in enumerate(model[corpus]):
        try:
            row = row_list[0] if model.per_word_topics else row_list
        except:
            row =  row_list

        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0 :  # => dominant topic
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    texts = [" ".join(t)[0:200] + " ..."  for t in texts]

    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def print_topics(dataframe,fname,n_line=8):
    dataframe.head(n_line).to_csv(fname+'.csv', sep='|')

def dominant_topic(dataframe,description):
    # Format
    df_dominant_topic = dataframe.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    print_topics(df_dominant_topic,description+'_dominant_topic')

def topic_sents(dataframe,description):
    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100
    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = dataframe.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat(
            [sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], axis=0)
    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    # Show
    print_topics(sent_topics_sorteddf_mallet, description+'_dominant_sentence_topic')

def wordCount_perDocument(model,docs,n_topics,description):
    topics = model.show_topics(num_topics=n_topics,formatted=False)
    data_flat = [w for w_list in docs for w in w_list]
    counter = Counter(data_flat)
    out = []
    indx_topic = []
    for i, topic in topics:
        indx_topic.append(i)
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        if i > 7:
            ax.axis('off')
            break
        id = indx_topic[i]
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == id, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == id, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030);
        ax.set_ylim(0, 3500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=10)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == id, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=18, y=1.05)
    plt.savefig(description)
    plt.close()

def WordCLoud(model,description):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    cloud = WordCloud(stopwords=stop,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=1000,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)


    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        if i == n_topics:
            fig.add_subplot(ax)
            plt.gca().axis('off')
            break

        fig.add_subplot(ax)
        cloud.fit_words(dict(model.show_topic(i, 50)))
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig(description)
    plt.close()

def print_result(model,data,corpus,n_topics=8,description='LDA Model'):
    df_topic_sents_keywords = format_topics_sentences(model=model, corpus=corpus, texts=data)
    dominant_topic(df_topic_sents_keywords,description) #dominant topic representation
    topic_sents(df_topic_sents_keywords,description)#representative sentence per topic

    WordCLoud(model, description+"word_cloud")
    wordCount_perDocument(model, data, n_topics,description+"wordCount_document")

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3,description='LDA Coherence score'):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    limit = 40
    start = 2
    step = 6
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig(description)
    plt.close()

def ldamodel(doc_clean,n_topics,n_words,description,tfidfmodel=False,unseen_docs=None):
    doc_clean = [min_char(doc).split() for doc in doc_clean]

    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
    compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=doc_clean, start=2, limit=40, step=6)
    if tfidfmodel:
       tfidf = TfidfModel(corpus,id2word=dictionary,smartirs='ntc')
       corpus = tfidf[corpus]

    ldamodel = LdaModel(corpus, num_topics=16, id2word=dictionary,random_state=1,passes=50,per_word_topics=True)
    print("#Tópicos LDA")
    for i in range(0, n_topics):
        temp = ldamodel.show_topic(i, n_words)
        terms = []
        for term in temp:
            terms.append(term)
        print("Topic #" + str(i) + ": ", ", ".join([t + '*' + str(i) for t, i in terms]))
    print('Bound: ',ldamodel.bound(corpus))
    # Compute Perplexity
    print('Perplexity: ',ldamodel.log_perplexity(corpus))
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    if unseen_docs:
        corpus_new = [dictionary.doc2bow(doc) for doc in unseen_docs]
        for i, unseen_doc in enumerate(corpus_new):
            topic = None
            score = 0
            inference_doc = ldamodel[unseen_doc]
            print(unseen_docs[i])
            for index,tmpScore in inference_doc[0]:
                if tmpScore > score:
                    score = tmpScore
                    topic = ldamodel.print_topic(index, 5)
            print ("Score: {}\t Topic: {}".format(score, topic))
        print("Log perplexity for new corpus is", ldamodel.log_perplexity(corpus_new))

    print_result(ldamodel, doc_clean, corpus, n_topics, description)
    pickle.dump(corpus, open(description+'.pkl', 'wb'))
    dictionary.save(description+'dictionary.gensim')
    ldamodel.save(description+'_ldamodel.gensim')

def lsamodel(doc_clean,n_topics,n_words,description,tfidfmodel=False):
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    corpus = [dictionary.doc2bow(doc) for doc in doc_clean]

    if tfidfmodel:
        tfidf = gensim.models.TfidfModel(corpus)  # Generate tfidf matrix (tf-idf model)
        corpus = tfidf[corpus]  # use the model to transform vectors
    #======================LSA ==============================
    lsa = LsiModel
    lsamodel = lsa(corpus, num_topics=n_topics, id2word=dictionary)
    print("#Tópicos LSA")
    for i in range(0, n_topics):
        temp = lsamodel.show_topic(i, n_words)
        terms = []
        for term in temp:
            terms.append(term)
        print("Topic #" + str(i) + ": ", ", ".join([t+'*'+str(i) for t, i in terms]))

    print(lsamodel.show_topic(n_topics,n_words))
    # Compute Coherence Score
    coherence_model_lsa = CoherenceModel(model=lsamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
    df_topic_sents_keywords = format_topics_sentences(model=lsamodel, corpus=corpus, texts=doc_clean)
    dominant_topic(df_topic_sents_keywords,description)  # dominant topic representation
    topic_sents(df_topic_sents_keywords,description)  # representative sentence per topic

    WordCLoud(lsamodel, description + "word_cloud")
    wordCount_perDocument(lsamodel, doc_clean, n_topics, description + "wordCount_document")
    coherence_lsa = coherence_model_lsa.get_coherence()
    print('\nLSA Coherence Score: ', coherence_lsa)
    fig = plt.figure()
    ## word clouds
    for i in range(n_topics):
       ax = fig.add_subplot(3, 3, i + 1)
       wordcloud = WordCloud(font_path="LiberationMono-Regular.ttf", background_color="white",max_words=50,max_font_size=300).fit_words(dict(lsamodel.show_topic(i, 50)))
       ax.imshow(wordcloud, interpolation="bilinear")
       ax.axis('off')
       plt.title("Topic #" + str(i), fontsize=20)
    plt.savefig(description)
    plt.close(fig=fig)

    coherence_values = []
    for num_topics in range(2, 40, 6):
        model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        coherence = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v').get_coherence()
        coherence_values.append(coherence)
    limit = 40
    start = 2
    step = 6
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig(description+"_Coherence")
    plt.close()

    pickle.dump(corpus, open(description + '.pkl', 'wb'))
    dictionary.save(description + 'dictionary.gensim')
    lsamodel.save(description + '_lsamodel.gensim')

documents = []
dictionaty = None
corpus = None
inference = []
doc_clean = []
doc_clean_all = []
doc_clean_minchar = []
doc_clean_pos_tag = []
dataset = open(json_url, encoding="utf-8")
for i, line in enumerate(dataset.readlines()):
     doc = json.loads(line)['text']
     documents.append(doc)
     remPont = removePunctuation(doc).split()
     cleanWithMinchar = min_char(remPont).split()
     docCleanAll = cleanAll(doc).split()
     if i < 390:
         doc_clean.append(remPont)
         doc_clean.append(min_char(remPont).split())
         doc_clean_all.append(cleanAll(doc).split())
         doc_clean_minchar.append(min_char(docCleanAll).split())
         doc_clean_pos_tag.append(pos_tag(docCleanAll))
     else:
         inference.append(pos_tag(docCleanAll))
dataset.close()

# Build the bigram and trigram models
bigram_phrases = gensim.models.Phrases(doc_clean_all,min_count=5, threshold=300)
bigram_mod = gensim.models.phrases.Phraser(bigram_phrases)
trigram_phrases =  gensim.models.Phrases(bigram_mod[doc_clean_all], threshold=300)
trigram_mod = gensim.models.phrases.Phraser(trigram_phrases)

doc_clean_gram = make_trigrams(doc_clean_all)
n_topics = 8
n_words = 10

#Experiências finais efectuadas
#remover somente pontuação
#ldamodel(doc_clean,n_topics,n_words,description='LDA_Model')
#lsamodel(doc_clean,n_topics,n_words,description='LSA_Model')

#remover stopwords,pontuacao,normalizacao
#ldamodel(doc_clean_all,n_topics,n_words,description='LDA_Model_clean_all')
#lsamodel(doc_clean_all,n_topics,n_words,description='LSA_Model_clean_all')

#remover stopwords,pontuacao,normalizacao,bigram
#ldamodel(doc_clean_gram,n_topics,n_words,description='LDA_Model_gram')
#lsamodel(doc_clean_gram,n_topics,n_words,description='LSA_Model_gram')

#remover stopwords,pontuacao,normalizacao,postag
#ldamodel(doc_clean_pos_tag,n_topics,n_words,description='LDA_Model_postag')
#lsamodel(doc_clean_pos_tag,n_topics,n_words,description='LSA_Model_postag')

#remover stopwords,pontuacao,normalizacao,bigram,postag
#ldamodel(doc_clean_gram_postag,n_topics,n_words,description='LDA_Model_gram_postag',tfidfmodel=False)
#lsamodel(doc_clean_gram_postag,n_topics,n_words,description='LSA_Model_gram_postag',tfidfmodel=False)

doc_inference_gram_postag = [ trigram_mod[bigram_mod[doc]] for doc in inference]
ldamodel(doc_clean_gram_postag,n_topics,n_words,description='LDA_Model_gram_postag_inference',unseen_docs=doc_inference_gram_postag)
lsamodel(doc_clean_gram_postag,n_topics,n_words,description='LSA_Model_gram_postag_inference')
