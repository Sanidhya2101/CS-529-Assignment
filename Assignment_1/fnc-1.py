import pandas as pd
from raw_features import *
from models import *

model = input("Enter the Classifier: ")

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

y_train = train['Stance']
y_test = test['Stance']


# getting tfidf
print("generating tf-idf...")
X_train_tfidf, X_test_tfidf = gettfidf(train,test)


print("\n")
predict(model,X_train_tfidf,y_train,X_test_tfidf,y_test)
print("\n")