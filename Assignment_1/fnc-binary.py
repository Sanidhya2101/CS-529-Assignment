import pandas as pd
from raw_features import *
from models import *
import time

model = input("Enter the Classifier: ")

print("\n")

# read training and testing dataset
print("reading dataset...")
train= create_dataframe('FNC_Binary/FNC_Bin_Train.csv')
test = create_dataframe('FNC_Binary/FNC_Bin_Test.csv')


# merge test and train dataset on body id
test.sort_values(by=['Body_ID'])
train.sort_values(by=['Body_ID'])

y_train = train['Stance']
y_test = test['Stance']


# getting tfidf
print("generating tf-idf...")
X_train_tfidf, X_test_tfidf = gettfidf(train,test,'fnc-bin')


print("\n")
start = time.time()
predict(model,X_train_tfidf,y_train,X_test_tfidf,y_test)
end = time.time()
print("Total Execution time = " + str(end-start) + " sec")
print("\n")