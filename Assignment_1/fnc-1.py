import pandas as pd
import numpy as np
from raw_features import *
from models import *
from feature_selection import *
from dimension_reduction import *
import time



print("\n")

# read training and testing dataset
print("reading dataset...")
train_stance = create_dataframe('FNC1/train_stances.csv')
train_body = create_dataframe('FNC1/train_bodies.csv')
test_stance = create_dataframe('FNC1/competition_test_stances.csv')
test_body = create_dataframe('FNC1/competition_test_bodies.csv')


# merge test and train dataset on body id
train = pd.merge(train_stance,train_body[['Body_ID', 'articleBody']],on='Body_ID')
test = pd.merge(test_stance,test_body[['Body_ID', 'articleBody']],on='Body_ID')
test.sort_values(by=['Body_ID'])
train.sort_values(by=['Body_ID'])

train['Headline+Body'] = train['Headline']+" "+train['articleBody']
test['Headline+Body'] = test['Headline']+" "+test['articleBody']


X_train = train.drop(columns=['Headline','Body_ID','articleBody','Stance'])
X_test = test.drop(columns=['Headline','Body_ID','articleBody','Stance'])

target = ['agree', 'disagree', 'discuss', 'unrelated']
target_dict = dict(zip(target, range(len(target))))

f = lambda x: target_dict[x]
train['label'] = [f(train['Stance'][i]) for i in range(len(train['Stance']))]
test['label'] = [f(test['Stance'][i]) for i in range(len(test['Stance']))]


y_train = train['label']
y_test = test['label']




# getting tfidf
print("generating tf-idf...")
X_train_tfidf, X_test_tfidf = gettfidf(X_train,X_test)


print("\n")
predict(X_train_tfidf,y_train,X_test_tfidf,y_test)
print("\n")


print("selecting feature using mutual information...")
X_train_mi,X_test_mi = mutual_info(X_train_tfidf,y_train,X_test_tfidf)
print("\n")
predict(X_train_mi,y_train,X_test_mi,y_test)    
print("\n")


print("selecting feature using  chi2...")
X_train_chi,X_test_chi = chi2_features(X_train_tfidf,y_train,X_test_tfidf)
print("\n")
predict(X_train_chi,y_train,X_test_chi,y_test)    
print("\n")


print("reducing dimension using pca on features selected from MI...")
X_train_pca_mi,X_test_pca_mi = pca_reduction(X_train_mi,y_train,X_test_mi)
print("\n")
predict(X_train_pca_mi,y_train,X_test_pca_mi,y_test)    
print("\n")


print("reducing dimension using pca on features selected from chi2...")
X_train_pca,X_test_pca = pca_reduction(X_train_chi,y_train,X_test_chi)
print("\n")
predict(X_train_pca,y_train,X_test_pca,y_test)    
print("\n")

print("reducing dimension using lsi on features selected from mi...")
X_train_lsi,X_test_lsi = lsi_reduction(X_train_mi,y_train,X_test_mi)
print("\n")
predict(X_train_lsi,y_train,X_test_lsi,y_test)    
print("\n")

print("reducing dimension using lsi on features selected from chi2...")
X_train_lsi,X_test_lsi = lsi_reduction(X_train_chi,y_train,X_test_chi)
print("\n")
predict(X_train_lsi,y_train,X_test_lsi,y_test)    
print("\n")