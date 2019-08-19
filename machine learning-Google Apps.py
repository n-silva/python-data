from datetime import datetime
from collections import Counter
import time

import numpy as np
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, average_precision_score, precision_score, \
    jaccard_similarity_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import label_binarize, LabelBinarizer, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.externals import joblib

from scipy import interp

import pandas as pd  # Data processing, CSV file I/O
import numpy as np  # Linear algebra
import seaborn as sns
from math import sqrt

from sklearn import preprocessing #model processing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier #model classifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb

from sklearn.ensemble import VotingClassifier

from sklearn.tree import export_graphviz
import statsmodels.api as sm
from sklearn import metrics
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report

from matplotlib import pyplot as plt
import pydot
import plotly as pl
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

import math
import re
from re import match as re_match

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#Clean and Convert all too KB
def size_format(n):
    try:
        size = float(re.findall(r"[-+]?\d*\.\d+|\d+", n)[0])
        s = '{:.0f}'.format(size * 1024) if n[-1] == 'M' else size
    except:
        s = 0
    return int(s)

def modify_rating(r):
    new_rating = 0
    if r >= 0 and r < 0.5:
        new_rating = 0
    elif r >= 0.5 and r < 1.5:
        new_rating = 1
    elif r >= 1.5 and r < 2.5:
        new_rating = 2
    elif r >= 2.5 and r < 3.5:
        new_rating = 3
    elif r >= 3.5 and r < 4.5:
        new_rating = 4
    elif r >= 4.5 and r <= 5.0:
        new_rating = 5
    else:
        #print("rating desconforme: ", r)
        new_rating = 0
    return int(new_rating)

def modify_rating_1_5_scale(r):
    new_rating = 0
    if r >= 1.0 and r < 2.0:
        new_rating = 1
    elif r >= 2.0 and r < 3.0:
        new_rating = 2
    if r >= 3.0 and r < 4.0:
        new_rating = 3
    elif r >= 4.0 and r <= 4.9:
        new_rating = 4
    elif r > 4.9:
        new_rating = 5
        # print("rating desconforme: ", r)
        #new_rating = 0
    return int(new_rating)

def pre_process_data(dataset):
    #print(dataset['Category'].value_counts().count())
    # Map 'Category' fields to numbers
    dataset['Category'] = dataset['Category'].map({
        'ART_AND_DESIGN': 0, 'AUTO_AND_VEHICLES': 1, 'BEAUTY': 2,
        'BOOKS_AND_REFERENCE': 3, 'BUSINESS': 4, 'COMICS': 5,
        'COMMUNICATION': 6, 'DATING': 7, 'EDUCATION': 8, 'ENTERTAINMENT': 9, 'EVENTS': 10, 'FINANCE': 11,
        'FOOD_AND_DRINK': 12, 'HEALTH_AND_FITNESS': 13, 'HOUSE_AND_HOME': 14,
        'LIBRARIES_AND_DEMO': 15, 'LIFESTYLE': 16, 'GAME': 17, 'FAMILY': 18, 'MEDICAL': 19,
        'SOCIAL': 20, 'SHOPPING': 21, 'PHOTOGRAPHY': 22, 'SPORTS': 23, 'TRAVEL_AND_LOCAL': 24,
        'TOOLS': 25, 'PERSONALIZATION': 26, 'PRODUCTIVITY': 27, 'PARENTING': 28, 'WEATHER': 29,
        'VIDEO_PLAYERS': 30, 'NEWS_AND_MAGAZINES': 31, 'MAPS_AND_NAVIGATION': 32}).astype(int)

    # Map 'Content Rating' fields to numbers
    dataset['Content Rating'] = dataset['Content Rating'].map({'Everyone': 0, 'Teen': 1, 'Mature 17+': 2,'Everyone 10+': 3, 'Adults only 18+': 4,'Unrated':5}).astype(int)

    # Convert 'Rating' values to int
    dataset['Rating_new'] = dataset['Rating'].apply(lambda n: modify_rating(n))
    # Convert 'Rating' values  float to int concat
    dataset['Rating_f2i'] = dataset['Rating'].apply(lambda n: modify_rating_1_5_scale(n))
    # Cleaning of genres
    Genres_list = dataset.Genres.unique()

    GenresDict = {}
    for i in range(len(Genres_list)):
        GenresDict[Genres_list[i]] = i
    dataset['Genres_clean'] = dataset['Genres'].map(GenresDict).astype(int)

    playstore = dataset.iloc[np.random.permutation(len(dataset))]
    return playstore

def data_clean (dataset):
    # Remove duplicates
    dataset = dataset.drop_duplicates(subset='App', keep="last").reset_index(drop=True)
    # Remove '$' in 'Price' and  Convert string to floate in 'Price'
    dataset['Price'] = dataset['Price'].apply(lambda p: int('{:.0f}'.format(float(p.replace('$', '')))))
    # dropping of unrelated and unnecessary items
    dataset.drop(labels=['Last Updated', 'Current Ver', 'Android Ver', 'App', 'Price'], axis=1, inplace=True)
    # Remove NaN
    dataset = dataset.dropna()
    print('Number of Applications in the Data Set after drop duplicates:', len(dataset))
    # Remove '+' and ',' in 'Installs' and Convert string to int in 'Installs'
    dataset['Installs'] = dataset['Installs'].apply(lambda p: int('{:.0f}'.format(float(p.replace('+', '').replace(',', '')))))
    # Pass 'Free' to '0' and 'Paid' to '1'
    dataset['Type'] = dataset['Type'].map({'Free': 0, 'Paid': 1})
    # Convert string to int in 'Reviews'
    dataset['Reviews'] = dataset['Reviews'].apply(lambda n: int(n))
    dataset['Size'] = dataset['Size'].apply(lambda n: size_format(n))

    dataset =  pre_process_data(dataset)
    return dataset

def correlationMatrix(dataset,filename):
    plt.clf()
    plt.figure(figsize=(12, 10), dpi=80)
    dataset = dataset.loc[:,['Installs', 'Content Rating', 'Genres_clean', 'Category', 'Reviews', 'Size','Type']]

    sns.heatmap(dataset.corr(), xticklabels=dataset.corr().columns, yticklabels=dataset.corr().columns, cmap='RdYlGn', center=0, annot=True)
    # Decorations
    plt.title('Correlogram Playstore dataset', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("playstore-matrix de correlacao_"+filename)
    #plt.show()
    plt.close()

def plotTable(columns,lines,columnWith,filename,auto_open=False):
    trace = go.Table(header=dict(values=columns,
                                 line=dict(color=['#506784']),
                                 fill=dict(color=['#119DFF']),
                                 ),
                     cells=dict(values=lines,
                                line=dict(color=['#506784']),
                                fill=dict(color=["lightgrey", '#F5F8FF'])
                                ),
                     columnwidth=columnWith)

    layout = go.Layout(dict(title=filename))
    figure = go.Figure(data=[trace], layout=layout)
    print(filename)
    pl.io.write_image(figure, 'anexo/TAB/'+filename+'.png')
    if auto_open: py.offline.plot(figure, output_type='file', filename='anexo/html/'+filename+'.html',auto_open=True)

def statistics(dataset,target,filename):
    plt.clf()
    size = dataset.shape[0]
    dataGroup = dataset.groupby([target]).size().reset_index(name='Total')
    val_lst = []
    for d in dataGroup.values:
        val_lst.append(['{:.2f}'.format(d[1]/size*100)+'%'])
    #print(val_lst)
    columns = ['Rating','Total','Percentage']
    lines = [dataGroup[target], dataGroup.Total,val_lst]
    filename = 'Total by Rating_table'+filename
    columnWith = [60, 100, 60]
    plotTable(columns, lines, columnWith, filename, False)
    dataset.groupby(target).size().plot(kind='bar')
    plt.savefig(filename)
    #plt.show()
    plt.close()

def data_analizes(dataset,target,filename):
    #py.offline.init_notebook_mode(connected=True)
    paid = dataset[dataset["Type"] == 1]
    free = dataset[dataset["Type"] == 0]

    # Separating catagorical and numerical columns
    summary = (dataset[[i for i in dataset.columns]].describe().transpose().reset_index())

    summary = summary.rename(columns={"index": "feature"})
    summary = np.around(summary, 3)

    val_lst = [summary['feature'], summary['count'],
               summary['mean'], summary['std'],
               summary['min'], summary['25%'],
               summary['50%'], summary['75%'], summary['max']]
    columns = summary.columns.tolist()
    lines = val_lst
    desc = 'Summary statistic table_'+filename
    columnWith = [200, 80, 80, 80, 80, 80, 80, 80, 80]
    plotTable(columns, lines, columnWith, desc,auto_open=False)

    #Rating por app 0 - free 1 - paid
    group_free = free[target].value_counts().reset_index()
    group_free.columns = [target, "count"]
    group_paid = paid[target].value_counts().reset_index()
    group_paid.columns = [target, "count"]

    # bar - free
    trace1 = go.Bar(x=group_free[target], y=group_free["count"],
                    name= target +" App free",
                    marker=dict(line=dict(width=.5, color="black")),
                    opacity=.9)

    # bar - paid
    trace2 = go.Bar(x=group_paid[target], y=group_paid["count"],
                    name= target + " App Paid",
                    marker=dict(line=dict(width=.5, color="black")),
                    opacity=.9)

    layout = go.Layout(dict(title="App rating by Type",
                            plot_bgcolor="rgb(243,243,243)",
                            paper_bgcolor="rgb(243,243,243)",
                            xaxis=dict(gridcolor='rgb(255, 255, 255)',
                                       title="tenure group",
                                       zerolinewidth=1, ticklen=5, gridwidth=2),
                            yaxis=dict(gridcolor='rgb(255, 255, 255)',
                                       title="count",
                                       zerolinewidth=1, ticklen=5, gridwidth=2),
                            )
                       )
    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    pl.io.write_image(fig,'App rating by Type_'+filename+'.png')

def roc_auc_score_multiclass(actual_class, pred_class,unique_class, filename):
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)
  i = 0

  for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        fpr, tpr, thresholds = roc_curve(new_actual_class, new_pred_class)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        i +=1
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(filename)
  plt.legend(loc="lower right")
  plt.savefig('anexo/ROC/'+filename+'.png')
  #plt.show()
  plt.close()

def compute_neighbors_values(X_train,X_test,y_train,y_test,limit, start=1, step=2,weights='uniform',filename='_uniform'):
    coherence_values = []
    cv_scores = []
    best_score = 0
    best_neighbors = 0
    minscore = 1.0
    k = 0
    for n_neighbors in range(start, limit, step):
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        score = float('{:.2f}'.format(accuracy_score(y_test, y_pred)))
        coherence_values.append(score)
        if best_score < score:
            best_score = score
            best_neighbors = n_neighbors
        """
        #print("precision: %.3f" % metrics.classification.precision_score(y_test, y_pred,average='micro'))
        #print("Area under ROC curve: %.3f" % metrics.roc_auc_score(y_test, y_pred[:1],average='micro'))
        scores2 = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')

        #print("Accuracy (cross validation score): %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

        if ((1-scores2.mean()) < minscore):
            minscore = 1-scores2.mean()
            k = n_neighbors
        cv_scores.append((1-scores2.mean()))
        """

    #print('Best score: ',best_score, 'with best neighbors: %d' %best_neighbors)
    #print('min Error: ', minscore, 'with k neighbors: %d' % k)
    x = range(start, limit, step)

    plt.clf()
    plt.plot(x, coherence_values)
    plt.title(filename)
    plt.xlabel('n_neighbors  (Best n_neighbors = %d score = %0.2f)' % (best_neighbors,best_score))
    plt.ylabel('Score')
    plt.legend(("coherence_values"), loc='best')
    plt.savefig("anexo/CV/Bestfit_"+filename)
    #plt.show()
    plt.close()
    """
    plt.plot(x, cv_scores)
    plt.xlabel('Min score error (K = %d min_err = %f)' % (k, minscore))
    plt.ylabel("Misclassification Error")
    plt.legend(("cv_scores"), loc='best')
    plt.savefig("anexo/CV/Min_score_error_"+filename)
    #plt.show()
    plt.close()
    """
    return best_neighbors

def getModel(models,X_train,X_test,y_train,y_test,unique_class, filename):

    lst_classifier, lst_accuracy, lst_precision, lst_recall,lst_f1_score = [],[],[],[],[]
    for model in models:
        name, classifier = list(model.keys())[0] ,list(model.values())[0]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = float('{:.2f}'.format(accuracy_score(y_test, y_pred)))
        #percentage = str('{:.2f}'.format(accuracy_score(y_test, y_pred,normalize=True)*100)+'%')
        precision_score = '{:.2f}'.format(metrics.precision_score(y_test, y_pred, average='weighted')) #str('{:.2f}'.format(metrics.precision_score(y_test, y_pred, average='micro')) * 100) + '%'
        f1_score = '{:.2f}'.format(metrics.f1_score(y_test, y_pred, average='weighted'))
        recall_score ='{:.2f}'.format(metrics.recall_score(y_test, y_pred, average='weighted'))

        print(name + ' Accuracy: ' + str('{:.2f}'.format(accuracy * 100) + '%'), ' Precision: ' +  str(precision_score),' F1 Score: ' + str(f1_score), ' Recall: ' +  str(recall_score))
        print("     %s"%name)
        print(" ")
        print("--------------------------------- Confusion matrix ---------------------------------")
        print(metrics.confusion_matrix(y_test, y_pred))
        print(" ")
        print("--------------------------------- Analyze ---------------------------------")
        print("     Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("     Misclassification Rate:", 1 - metrics.accuracy_score(y_test, y_pred))
        print(" ")
        print(classification_report(y_test, y_pred))
        roc_auc_score_multiclass(y_test, y_pred, unique_class,name+'_'+filename)

        lst_classifier.append(name)
        lst_accuracy.append(accuracy)
        lst_precision.append(precision_score)
        lst_recall.append(recall_score)
        lst_f1_score.append(f1_score)

    #print('#=================================================================================')
    columns = ['Classifiers', 'Precision','Recall','F1-score', 'Accuracy']
    lines = [lst_classifier,lst_precision,lst_recall,lst_f1_score, lst_accuracy]
    #filename = "Model Classifier Score_"+filename
    columnWith = [150, 60, 60, 60, 60]
    plotTable(columns, lines, columnWith, filename, auto_open=False)

def outliers_projection(data):
    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    print(mean, sd)
    print((mean - 2 * sd), (mean + 2 * sd))
    newdata = []
    for x in data:
        if (x < (mean - 2 * sd)):
            newdata.append(x)
        elif (x > mean + 2 * sd):
            newdata.append(x)

    print(len(newdata))
    sns.set(style="whitegrid", color_codes=True)
    sns.boxplot(data=data, showmeans=True, meanline=True)

    plt.yscale("log")
    #plt.show()
    plt.close()
    return newdata

def KNN_Classifier(X_train,X_test,y_train,y_test, unique_class,filename):
    # ['Installs', 'Content Rating', 'Genres_clean','Category','Reviews', 'Size'] 'Reviews', 'Size',
    k = compute_neighbors_values(X_train, X_test, y_train, y_test, 200,filename=filename)
    print(" =========================== K-Nearest Neighbor(KNN) optimized k = " + str(k) + "=========================== ")
    knn = KNeighborsClassifier(n_neighbors=k,weights=weights)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = float('{:.2f}'.format(accuracy_score(y_test, y_pred)))
    precision_score = '{:.2f}'.format(metrics.precision_score(y_test, y_pred, average='weighted'))  # str('{:.2f}'.format(metrics.precision_score(y_test, y_pred, average='micro')) * 100) + '%'
    f1_score = '{:.2f}'.format(metrics.f1_score(y_test, y_pred, average='weighted'))
    recall_score = '{:.2f}'.format(metrics.recall_score(y_test, y_pred, average='weighted'))
    
    print(' Accuracy: ' + str('{:.2f}'.format(accuracy * 100) + '%'), ' Precision: ' + str(precision_score), ' F1 Score: ' + str(f1_score), ' Recall: ' + str(recall_score))
    print("--------------------------------- Confusion matrix ---------------------------------")
    print(metrics.confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    roc_auc_score_multiclass(y_test, y_pred, unique_class, filename)
    return precision_score,recall_score,f1_score, accuracy

#Leitura do dataset
playstore = pd.read_csv(r'data/googleplaystore.csv')

print(playstore.isnull().sum().sort_values(ascending=False))
print('Number of Applications in the Data Set:', len(playstore))

Dataclean = data_clean(playstore)
print('Number of Applications in the Data Set before :', Dataclean.shape)
print(Dataclean.shape)
print(Dataclean.head())

target = 'Rating_new'
#target = 'Rating_f2i'
timestamp ='_'+str(int(time.mktime(datetime.now().timetuple())))

correlationMatrix(Dataclean,'Correlation matrix')
data_analizes(Dataclean,target,target+timestamp)
print(Dataclean.isnull().sum().sort_values(ascending=False))

Dataclean = Dataclean.iloc[np.random.permutation(len(Dataclean))]
random_state = 42
weights='uniform'
n_neighbors = 89


models = (
        {'Naive Bayes - GaussianNB':GaussianNB()},
        #{'Naive Bayes - BernoulliNB': BernoulliNB()},
        {'SVM': svm.SVC()},  # kernel='rbf', gamma='scale', C=1.1, probability = False,random_state=random_state
        {'K-Nearest Neighbor(KNN)':KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights)},
        {'Neural Network':MLPClassifier(random_state=random_state)},
        {'Logistic Regression':LogisticRegression(solver='lbfgs',random_state=random_state)},  #C = 0.1, max_iter = 100, fit_intercept = True, n_jobs = 3, solver = 'lbfgs'
        {'AdaBoost':AdaBoostClassifier(random_state=random_state)},
        {'Extra Trees':ExtraTreesClassifier(random_state=random_state)},
)

lst_features = (
        {'ALL features':['Installs', 'Content Rating', 'Genres_clean','Category','Reviews', 'Size']},
        {'Install_Reviews': ['Installs', 'Reviews']},
        {'Category_Genres':['Genres_clean','Category']},
        {'Install_Reviews_Category_Genres':['Installs', 'Genres_clean','Category','Reviews']},
)

y = Dataclean.loc[:, target].values
# creating a set of all the unique classes using the actual class list
unique_class = []
[unique_class.append(x) for x in y if x not in unique_class]
knn_features,knn_precision,knn_recall,knn_f1_score, knn_accuracy = [], [], [], [], []

for features in lst_features:
    name, feature = list(features.keys())[0], list(features.values())[0]
    X = Dataclean.loc[:, feature].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, shuffle=True)
    print(" =========================== " + name + " =========================== ")
    getModel(models,X_train,X_test,y_train,y_test,unique_class,name+timestamp)
    precision, recall, f1_score, accuracy = KNN_Classifier(X_train, X_test, y_train, y_test, unique_class,name+timestamp)
    knn_features.append(name)
    knn_precision.append(precision)
    knn_recall.append(recall)
    knn_f1_score.append(f1_score)
    knn_accuracy.append(accuracy)

dataset_genres_column = pd.get_dummies(Dataclean.loc[:, ['Installs', 'Content Rating','Category','Reviews', 'Size','Genres_clean']], columns = ['Genres_clean'])
dataset_genres_values = dataset_genres_column.loc[:, dataset_genres_column.columns.difference(['Rating_new', 'Rating_f2i'])].values

dataset_category_column = pd.get_dummies(Dataclean.loc[:, ['Installs', 'Content Rating', 'Genres_clean','Category','Reviews', 'Size']], columns = ['Category'])
dataset_category_values = dataset_category_column.loc[:, dataset_category_column.columns.difference(['Rating_new', 'Rating_f2i'])].values

#row to column features
print(" =========================== Genres as column =========================== ")
X = dataset_genres_values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, shuffle=True)
getModel(models,X_train,X_test,y_train,y_test,unique_class,"Genres as column")
precision, recall, f1_score, accuracy = KNN_Classifier(X_train, X_test, y_train, y_test,unique_class, "Genres as column")
knn_features.append("Genres as column")
knn_precision.append(precision)
knn_recall.append(recall)
knn_f1_score.append(f1_score)
knn_accuracy.append(accuracy)

print(" =========================== Category as column =========================== ")
X = dataset_category_values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, shuffle=True)
getModel(models,X_train,X_test,y_train,y_test,unique_class,"Category as column")
precision, recall, f1_score, accuracy = KNN_Classifier(X_train, X_test, y_train, y_test,unique_class, "Category as column")
knn_features.append("Category as column")
knn_precision.append(precision)
knn_recall.append(recall)
knn_f1_score.append(f1_score)
knn_accuracy.append(accuracy)

features = ['Installs','Reviews']
print(" =========================== KNN - Optimized with Outliers removed =========================== ")
outliers_Install = outliers_projection(np.array(Dataclean.loc[:,['Installs']]))
outliers_Reviews = outliers_projection(np.array(Dataclean.loc[:,['Reviews']]))

Dataclean_new = Dataclean[~Dataclean.Installs.isin(outliers_Install) & ~Dataclean.Reviews.isin(outliers_Reviews)]
print(Dataclean_new.shape)

X = Dataclean_new.loc[:, features].values
y = Dataclean_new.loc[:, target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, shuffle=True)
# creating a set of all the unique classes using the actual class list
unique_class = []
[unique_class.append(x) for x in y if x not in unique_class]

precision, recall, f1_score, accuracy = KNN_Classifier(X_train, X_test, y_train, y_test, unique_class,' knn teste')
print(precision, recall, f1_score, accuracy)

knn_features.append("Outliers removed")
knn_precision.append(precision)
knn_recall.append(recall)
knn_f1_score.append(f1_score)
knn_accuracy.append(accuracy)

#print table with KNN results
columns = ['Features', 'Precision','Recall','F1-score', 'Accuracy']
lines = [knn_features,knn_precision,knn_recall,knn_f1_score, knn_accuracy]
filename = "KNN optmized score"+timestamp
columnWith = [200, 60, 60, 60, 60]
plotTable(columns, lines, columnWith, filename, auto_open=False)
