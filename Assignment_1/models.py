from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,accuracy_score
import time



def predict(model,X_train,y_train,X_test,y_test):

    if model == 'SVM':
        print("running SVM classifier...")
        
        start = time.time()
        svm_classifier = LinearSVC()
        svm_classifier.fit(X_train,y_train)
        end = time.time()
        print("Total Execution time = " + str(end-start) + " sec")
        
        print("predicting test dataset...")
        svm_prediction = svm_classifier.predict(X_test)

        print("micro F1-score: " + str(f1_score(y_test, svm_prediction, average='micro')))
        print("macro F1-score: " + str(f1_score(y_test, svm_prediction, average='macro')))
        print(accuracy_score(y_test,svm_prediction))
        

    elif model == 'Naive Bayes':
        
        start = time.time()
        print("running Naive Bayes classifier...")
        bayes_classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        bayes_classifier.fit(X_train,y_train)
        end = time.time()
        print("Total Execution time = " + str(end-start) + " sec")
        

        print("predicting test dataset...")
        bayes_prediction = bayes_classifier.predict(X_test)

        print("micro F1-score: " + str(f1_score(y_test, bayes_prediction, average='micro')))
        print("macro F1-score: " + str(f1_score(y_test, bayes_prediction, average='macro')))
        print(accuracy_score(y_test,bayes_prediction))
        

    elif model == "Decision Tree":
        
        start = time.time()
        print("running Decision Tree Classifier...")
        decision_tree_classifier = DecisionTreeClassifier(random_state=0)
        decision_tree_classifier.fit(X_train,y_train)
        end = time.time()
        print("Total Execution time = " + str(end-start) + " sec")
        

        print("predicting test dataset...")
        decision_tree_prediction = decision_tree_classifier.predict(X_test)

        print("micro F1-score: " + str(f1_score(y_test, decision_tree_prediction, average='micro')))
        print("macro F1-score: " + str(f1_score(y_test, decision_tree_prediction, average='macro')))
        print(accuracy_score(y_test,decision_tree_prediction))
        
        

    elif model == "Random Forest":

        start = time.time()
        print("running Random Forest Classifier...")
        rf_classifier = RandomForestClassifier(n_estimators=10)
        rf_classifier.fit(X_train,y_train)
        end = time.time()
        print("Total Execution time = " + str(end-start) + " sec")
        

        print("predicting test dataset...")
        rf_prediction = rf_classifier.predict(X_test)

        print("micro F1-score: " + str(f1_score(y_test, rf_prediction, average='micro')))
        print("macro F1-score: " + str(f1_score(y_test, rf_prediction, average='macro')))
        print(accuracy_score(y_test,rf_prediction))
        
        

