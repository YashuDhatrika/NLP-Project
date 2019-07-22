# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:37:48 2019

@author: Yashu Dhatrika
"""
#%%
#Loading the required packages
import glob
import os
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')

import nltk
nltk.download('wordnet')

from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tag.stanford import StanfordPOSTagger
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import os 
cwd = os.getcwd()
#os.chdir(r"C:\Users\Yashu Dhatrika\Desktop") 
from sklearn_crfsuite import metrics  
print(cwd)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
#%%

#oading the datasets, input and sql query
with open('new_train.nl','r') as f:
#    next(f) # skip 1 line
#    next(f)
    input_text=[]
    for lines in f:
        x=lines.split()
        input_text.append(x)
with open('new_train.nl','r') as f:
#    next(f) # skip 1 line
#    next(f)
    input_sen=[]
    for lines in f:
        x=lines
        input_sen.append(x)
with open('new_train.sql','r') as f:
#    next(f) # skip 1 line
#    next(f)
    output_sql=[]
    aggregate=[]
    for lines in f:
        x=lines.split()
        output_sql.append(x)   
        for i in range(0,len(x)):
            if (x[i]=='DISTINCT' or  x[i]=='count'):
                aggregate.append(x[i])

#%%
    
#Doing the lemmatization on the text
lem = WordNetLemmatizer()
lem_data=[]
for one_text in input_text:
    sublst=[]
    for i in range(0,len(one_text)):
        sublst.append(lem.lemmatize(one_text[i]))
    lem_data.append(sublst)
    

lower_sql=[]
for one_sql in output_sql:
    sublst1=[x.lower() for x in one_sql]
    lower_sql.append(sublst1)    

#Tagging the word into select clause, having clause etc. in natural language input by match words in sql query
select_tag=[]
from_tag=[]
where_at_tag=[]
where_val_tag=[]
group_by_tag=[]

for line in lower_sql:
    group_index=0
    from_index=line.index('from')
    where_index=line.index('where')
    if line.count('group') >= 1:
        group_index=line.index('group')
        
    if line.count('=') >= 1:
        where_value=[i for i, x in enumerate(line) if x == "="]
        val=[]
        at=[]
        for some in where_value:
            val.append(line[some+1])
            at.append(line[some-1])
    where_val_tag.append(val)
    where_at_tag.append(at)        
    select_tag.append(line[2:from_index])
    from_tag.append(line[from_index+1:where_index])
    if group_index==0:
        group_by_tag.append('')
    else:
        group_by_tag.append(line[group_index+1:len(line)])
#%%
sql_tag=[]
important_tag=[]
for j in range(len(select_tag)):
    sublst=lem_data[j]
    tag=['O']*len(sublst)
    imp_tag=[0]*len(sublst)
    for i in range(0,len(sublst)):    
         if next((s for s in select_tag[j] if sublst[i] in s), "None") !="None":
             tag[i]='select-tag'
             imp_tag[i]=1
         if next((s for s in where_val_tag[j] if sublst[i] in s), "None") !="None":
             tag[i]='where_val'
             imp_tag[i]=1
         elif next((s for s in where_at_tag[j] if sublst[i] in s), "None") !="None":
             tag[i]='where_at'
             imp_tag[i]=1
         elif next((s for s in group_by_tag[j] if sublst[i] in s), "None") !="None":
             tag[i]='group-by'
             imp_tag[i]=1
    sql_tag.append(tag) 
    important_tag.append(imp_tag)        
    

#%%

#Generating the templates from sql query
main_template=[]
for each in output_sql:
    sub_lst=[]
    for i in range(len(each)):
        if (each[i]=="SELECT" or each[i]=="DISTINCT" or each[i]=="count" or each[i]=="FROM" or each[i]=="WHERE" or each[i]=="GROUP" or each[i]=="BY" or each[i]=="MIN" or each[i]=="MAX"):
            sub_lst.append(each[i])
    if sub_lst not in main_template:
        main_template.append(sub_lst)

key=list(range(len(main_template)))
template_dict = dict(zip(key,main_template))


label_template=[]
for each in output_sql:
    sub_lst=[]
    for i in range(len(each)):
        if (each[i]=="SELECT" or each[i]=="DISTINCT" or each[i]=="count" or each[i]=="FROM" or each[i]=="WHERE" or each[i]=="GROUP" or each[i]=="BY" or each[i]=="MIN" or each[i]=="MAX"):
            sub_lst.append(each[i])
    for k, v in template_dict.items():
#        print(v)
#        print(sub_lst)# for name, age in dictionary.iteritems():  (for Python 2.x)
        if v == sub_lst:
            label_template.append(k)       


#extracting pos tag for the words in the sentence.

pos_tag=[]
for review in input_text:
#        print(nltk.pos_tag(review))
    pos_tag.append(nltk.pos_tag(review))         

#creating a tupple of word, postag and important_attribute
only_tag=[]
only_text=[]

for i in range(0,len(pos_tag)):
    sub_tag=[]
    sub_text=[]
    for item in pos_tag[i]:
        sub_tag.append(item[1])
        sub_text.append(item[0])
    only_tag.append(sub_tag)
    only_text.append(sub_text)    
        
main_lst1=[]

for i in range(len(only_tag)):
    main_lst2=[]
    for j in range(len(only_tag[i])):
        sub_tup=()
        sub_tup=sub_tup+(only_text[i][j],only_tag[i][j],important_tag[i][j],sql_tag[i][j])
        main_lst2.append(sub_tup)
    main_lst1.append(main_lst2)         
     
#extracting the tag for the sql clause label

val=[]
for item in sql_tag:
    for i in range(len(item)):
        if item[i] not in val:
            val.append(item[i])



#%% Creating a feautures for the model of CRF

def word2features(sent, i):
    word = sent[i][0]
    postag=sent[i][1]
    imp_atr=sent[i][2]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'imp_atr':imp_atr,
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [sent[i][3] for i in range(len(sent))]

def sent2tokens(sent):
    return [token for token in sent]
            
#%%
X = [sent2features(s) for s in main_lst1]

Y=sql_tag
#Y=[sent2labels(s) for s in sql_tag]


Xtrain=X[0:2000]
Xtest=X[2000:]
Ytrain=Y[0:2000]
Ytest=Y[2000:]


#%%
from sklearn_crfsuite import CRF

crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

pred = cross_val_predict(estimator=crf, X=Xtrain, y=Ytrain, cv=5)

report = flat_classification_report(y_pred=pred, y_true=Ytrain)
#%%
crf.fit(Xtrain, Ytrain)

y_pred = crf.predict(Xtest)
metrics.flat_f1_score(Ytest, y_pred,average='weighted')

print(metrics.flat_classification_report(
    Ytest, y_pred, digits=3
))
#%%
print(report)

print(y_pred[0])
print(output_sql[2000])
print(main_lst1[2000])
print()

#%%



X=CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(lem_data)
#print(ngram_vectorizer.fit(data))
#X = ngram_vectorizer.transform(data)

X_train=X[0:2000,]
X_test=X[2000:,]
y_train=label_template[0:2000]
y_test=label_template[2000:]
#%%

#X_train, X_test, y_train, y_test = train_test_split(X, label_template, train_size = 0.75)
#logisticregression    
lr = LogisticRegression(C=1)
lr.fit(X_train, y_train)
print('Accuracy using Logistic Regression')
print(accuracy_score(y_test, lr.predict(X_test)))

#svm
from sklearn.svm import SVC  
svr_lin = SVC(kernel='linear', C=1)
svr_lin.fit(X_train, y_train)

#print('Accuracy using SVM')
#print(accuracy_score(y_test, svr_lin.predict(X_test)))
##%%
#print(confusion_matrix(y_test, svr_lin.predict(X_test))) 

#%%
#Extracting the sql query
value_temp=svr_lin.predict(X_test)[0]
templ_sel=template_dict[value_temp]
text_query=only_text[2000]
attribute_class=y_pred[0]
for i in range(0,len(text_query)):
    for j in range(0,len(attribute_class)):
        if i==j:
            
            if attribute_class[j]=='where_at':
                index_where=int(templ_sel.index('WHERE'))
                index_from=int(templ_sel.index('FROM'))
                
                if text_query[i]=='airline':
                    text='airline_code ='
                    templ_sel.insert(index_where+1,text )
                    templ_sel.insert(index_from-1,"airline_code" )
                    index_from=int(templ_sel.index('FROM'))
                    templ_sel.insert(index_from+1,text_query[i] )
                    
            if attribute_class[j]=='where_val':
                templ_sel.insert(-1,text_query[i],)
                
                
                 


#%%
                
               
    