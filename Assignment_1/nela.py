import pandas as pd
from raw_features import *
from models import *
import time

model = input("Enter the Classifier: ")

print("\n")

# read training and testing dataset
print("reading dataset...")
train= create_dataframe('NELA/NELA_Train.csv')
test = create_dataframe('NELA/NELA_Test.csv')

train.rename(columns = {'Body': 'articleBody'}, inplace = True)
test.rename(columns = {'Body': 'articleBody'}, inplace = True)

y_train = train['Label']
y_test = test['Label']


# getting tfidf
print("generating tf-idf...")
X_train_tfidf, X_test_tfidf = gettfidf(train,test,'nela')


print("\n")
start = time.time()
predict(model,X_train_tfidf,y_train,X_test_tfidf,y_test)
end = time.time()
print("Total Execution time = " + str(end-start) + " sec")
print("\n")