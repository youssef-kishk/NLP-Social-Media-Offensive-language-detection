##############################
#   Created on: January 2019 #
#                            #
#                            #
#     Author: Youssef Kishk  #
#             Rimon Adel     #
#             Adel Atef      #
#             Sandra sherif  #
#                            #
#                            #
#                            #
#                            #
##############################
import re
import pandas as pd
import pickle
from pprint import pprint
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
dataset = pd.read_csv('offenseval-training-v1.tsv', delimiter='\t')
mustBeRemovedList = ["@USER", "url"]


# #################################################################################################################
def remove_userTag():
    datasetwithoutUserTag = []
    for line in dataset['tweet']:
        finalListOfWords = []
        tweets = []
        words = line.split()
        for word in words:
            if word not in mustBeRemovedList:
                finalListOfWords.append(word)
        tweets = " ".join(finalListOfWords)
        datasetwithoutUserTag.append(tweets)
    return datasetwithoutUserTag
# #################################################################################################################

# #################################################################################################################
noise_list = set(stopwords.words("english"))
# noise detection
def remove_noise(input_text):
    words = word_tokenize(input_text)
    noise_free_words = list()
    i = 0;
    for word in words:
        if word.lower() not in noise_list:
            noise_free_words.append(word)
        i += 1
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text
# #################################################################################################################

# #################################################################################################################
def lemetize_words(input_text):
    words = word_tokenize(input_text)
    new_words = []
    lem = WordNetLemmatizer()
    for word in words:
        word = lem.lemmatize(word, "v")
        new_words.append(word)
    new_text = " ".join(new_words)
    return new_text
# #################################################################################################################

# #################################################################################################################
def cleaning():
    corpus = []
    datasetwithoutUserTag = remove_userTag()
    for line in datasetwithoutUserTag:
        review = re.sub('[^a-zA-Z]', ' ', line)
        review = review.lower()
        # remove non segnificant words
        review = remove_noise(review)
        review = lemetize_words(review)
        corpus.append(review)
    return corpus
# #################################################################################################################

# #################################################################################################################
def bagOfWordsCreation(corpus):
    cv = CountVectorizer(max_features=12000)
    bagOfWords = cv.fit_transform(corpus).toarray()
    rowsValues = []
    for line in dataset['subtask_a']:
        if line == "OFF":
            rowsValues.append(1)
        else:
            rowsValues.append(0)
    return (bagOfWords, rowsValues)
# #################################################################################################################

# #################################################################################################################  
def classifiers(classifier):
    # fitting classifer to the training set
    classifier_to_save = classifier.fit(bagOfWords_train, rowsValues_train)

    # predict the test set resulty
    rowsValues_pred = classifier.predict(bagOfWords_train)
    # confusion matrix
    cm = confusion_matrix(rowsValues_train, rowsValues_pred)
    print('confusuion matrix train before tunning\n', cm)
    accuracyTrain = (cm[0][0] + cm[1][1]) / len(rowsValues_train)

    rowsValues_pred = classifier.predict(bagOfWords_test)
    cm = confusion_matrix(rowsValues_test, rowsValues_pred)
    print('confusuion matrix test before tunning\n', cm)
    accuracyTest = (cm[0][0] + cm[1][1]) / len(rowsValues_test)

    return accuracyTrain, accuracyTest, classifier_to_save
# #################################################################################################################  

# #################################################################################################################  
def save_classifier(classifier_name, classifier_s):
    save_classifier = open(classifier_name + ".pickle", "wb")
    pickle.dump(classifier_s, save_classifier)
    save_classifier.close()
    return
# #################################################################################################################  

# #################################################################################################################  
def use_saved_classifierBeforeTunning(classifier_name):
    classifier_f = open(classifier_name + ".pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()

    # predict the test set resulty
    rowsValues_pred = classifier.predict(bagOfWords_train)
    # confusion matrix
    cm = confusion_matrix(rowsValues_train, rowsValues_pred)
    print('confusuion matrix train before tunning\n', cm)
    accuracyTrain = (cm[0][0] + cm[1][1]) / len(rowsValues_train)

    rowsValues_pred = classifier.predict(bagOfWords_test)
    cm = confusion_matrix(rowsValues_test, rowsValues_pred)
    print('confusuion matrix test before tunning\n', cm)
    accuracyTest = (cm[0][0] + cm[1][1]) / len(rowsValues_test)

    return accuracyTrain, accuracyTest
# #################################################################################################################  

# #################################################################################################################  
def use_saved_classifierAfterTunning(classifier_name):
    classifier_f = open(classifier_name + ".pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()

    # predict the test set resulty
    rowsValues_pred = classifier.predict(bagOfWords_train)
    # confusion matrix
    cm = confusion_matrix(rowsValues_train, rowsValues_pred)
    print('confusuion matrix train after tunning\n', cm)
    accuracyTrain = (cm[0][0] + cm[1][1]) / len(rowsValues_train)

    rowsValues_pred = classifier.predict(bagOfWords_test)
    cm = confusion_matrix(rowsValues_test, rowsValues_pred)
    print('confusuion matrix test after tunning\n', cm)
    accuracyTest = (cm[0][0] + cm[1][1]) / len(rowsValues_test)

    return accuracyTrain, accuracyTest
# ################################################################################################################# 
    
# #################################################################################################################  
def create_parameter_grid_randomForest():
    # Number of trees in random forest
    n_estimators = [int(x) for x in pd.np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in pd.np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   }
    print('random_grid')
    pprint(random_grid)
    return random_grid
# #################################################################################################################  
    
# #################################################################################################################  
def create_parameter_grid_KNNClassifier():
    # Number of neighbors to use
    n_neighbors = [int(x) for x in pd.np.linspace(start=3, stop=300, num=10)]

    # Create the random grid
    random_grid = {'n_neighbors':  n_neighbors,
                   }
    print('random_grid')
    pprint(random_grid)
    return random_grid
# #################################################################################################################  
    
# #################################################################################################################  
def create_parameter_grid_DecisionTreeClassifier():
    # The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity
    # and “entropy” for the information gain.
    criterion = ['gini', 'entropy']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in pd.np.linspace(10, 150, num=11)]
    max_depth.append(None)

    # Create the random grid
    random_grid = {'criterion': criterion,
                   'max_depth': max_depth,
                   }
    print('random_grid')
    pprint(random_grid)
    return random_grid
# #################################################################################################################  
    
# #################################################################################################################  
def create_parameter_grid_LogisticRegression():
    # Inverse of regularization strength; must be a positive float.
    # Like in support vector machines, smaller values specify stronger regularization
    C = [float(x) for x in pd.np.linspace(start=0.1, stop=5.0)]

    # Create the random grid
    random_grid = {'C':  C,
                   }
    print('random_grid')
    pprint(random_grid)
    return random_grid
# #################################################################################################################  

# #################################################################################################################  
def random_search_training(randomGrid,classifier):
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = classifier
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=randomGrid, n_iter=30, cv=3, verbose=0,
                                   random_state=42, n_jobs=-1, refit=True)
    # Fit the random search model
    rf_random.fit(bagOfWords_train, rowsValues_train)
    print('rf_random.best_params')
    var = rf_random.best_params_
    print(var)
    best_random = rf_random.best_estimator_
    return best_random
# #################################################################################################################  
    
# #################################################################################################################  
def evaluate(classifier):
    rowsValues_pred = classifier.predict(bagOfWords_test)
    cm = confusion_matrix(rowsValues_test, rowsValues_pred)
    print('confusuion matrix test\n', cm)
    accuracyTest = (cm[0][0] + cm[1][1]) / len(rowsValues_test)
    return accuracyTest
# #################################################################################################################  
def saveDecicsionTreeClassiferBeforeAndAfterTunning():
    #Find accuracy of training and validation data before tunning
    accuracyTrain, accuracyTest, classifier_s = classifiers(DecisionTreeClassifier(max_depth=30))
    save_classifier("decisionTreeClassifier", classifier_s)
    #tunning process
    random_Grid_DecisionTree = create_parameter_grid_DecisionTreeClassifier()
    DecisionTree_classifier_s_t = random_search_training(random_Grid_DecisionTree,DecisionTreeClassifier())
    save_classifier("decisionTreeClassifierTuned", DecisionTree_classifier_s_t)
    return
# ################################################################################################################# 
def saveLogisticRegClassiferBeforeAndAfterTunning():
    #Find accuracy of training and validation data before tunning
    accuracyTrain, accuracyTest, classifier_s = classifiers(LogisticRegression())
    save_classifier("logisticRegression", classifier_s)
    #tunning process
    random_Grid_logistic = create_parameter_grid_LogisticRegression()
    logistic_classifier_s_t = random_search_training(random_Grid_logistic,LogisticRegression())
    save_classifier("LogisticRegressionTuned", logistic_classifier_s_t)
    return
# ################################################################################################################# 
def saveKNNClassiferBeforeAndAfterTunning():
    #Find accuracy of training and validation data before tunning
    knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                         metric='cosine', metric_params=None, n_jobs=1)
    accuracyTrain, accuracyTest, classifier_s = classifiers(knn)
    save_classifier("KNN", classifier_s)
    #tunning process
    random_Grid_knn = create_parameter_grid_KNNClassifier()
    knn_classifier_s_t = random_search_training(random_Grid_knn,neighbors.KNeighborsClassifier())
    save_classifier("KNNTuned", knn_classifier_s_t)
    return
# ################################################################################################################# 
def saveRandomForestClassiferBeforeAndAfterTunning():
    #build classifier
    accuracyTrain, accuracyTest, classifier_s = classifiers(RandomForestClassifier())
    save_classifier("randomForest", classifier_s)
    
   # classifier_tuned = RandomForestClassifier(n_estimators = 1200, min_samples_split= 2, min_samples_leaf= 4, max_features= 'sqrt' , max_depth =  None)
   # accuracyTrainTuned1, accuracyTestTuned1, classifier_s_t = classifiers(classifier_tuned)
   
    random_Grid = create_parameter_grid_randomForest()
    var= random_search_training(random_Grid,RandomForestClassifier())
    save_classifier("randomForestTuned", var)
 
    return
# #################################################################################################################  
#End of functions 
# #################################################################################################################  
#Code Starts Running From here by calling set of the above implemented functions    
corpus = cleaning()
bagOfWords, rowsValues = bagOfWordsCreation(corpus)
# splitting data into training and testing data
bagOfWords_train, bagOfWords_test, rowsValues_train, rowsValues_test = train_test_split(bagOfWords, rowsValues,
                                                                                        test_size=0.2, random_state=0)
# ################################################################################################################# 
#Random Forst Classifer
print('\nRandom Forest')
#saveRandomForestClassiferBeforeAndAfterTunning()

#use saved classifier to predicit training and test sets
accuracyTrain2, accuracyTest2= use_saved_classifierBeforeTunning("randomForest")


#use saved tuned classifier to predicit training and test sets
accuracyTrainTuned2, accuracyTestTuned2 = use_saved_classifierAfterTunning("randomForestTuned")


print('accuracy Train after saving base classifier = ', accuracyTrain2)
print('accuracy Test after saving base classifier = ', accuracyTest2)
print('accuracyTrainTuned after saving tuned classifier = ', accuracyTrainTuned2)
print('accuracyTestTuned after saving tuned classifier  = ', accuracyTestTuned2)
print('\n********************************************************\n')
# ################################################################################################################# 

# ################################################################################################################# 
#Naive Base Classifer
print('Naive Base')
#accuracyTrain, accuracyTest, classifier_s = classifiers(MultinomialNB())
#save_classifier("naiveBase", classifier_s)
accuracyTrain2, accuracyTest2 = use_saved_classifierBeforeTunning("naiveBase")

print('accuracy Train2 = ', accuracyTrain2)
print('accuracy Test2  = ', accuracyTest2)
print('\n********************************************************\n')
# ################################################################################################################# 


# ################################################################################################################# 
#Decision Tree Classifer
print('DecisionTreeClassifier')
#saveDecicsionTreeClassiferBeforeAndAfterTunning()

#use saved before tunning classifier to predicit training and test sets
accuracyTrain2, accuracyTest2 = use_saved_classifierBeforeTunning("decisionTreeClassifier")

#use saved tuned classifier to predicit training and test sets
accuracyTrainTuned2, accuracyTestTuned2 = use_saved_classifierAfterTunning("decisionTreeClassifierTuned")

print('accuracy Train after saving base classifier = ', accuracyTrain2)
print('accuracy Test after saving base classifier = ', accuracyTest2)
print('accuracyTrainTuned after saving tuned classifier = ', accuracyTrainTuned2)
print('accuracyTestTuned after saving tuned classifier  = ', accuracyTestTuned2)
print('\n********************************************************\n')
# ################################################################################################################# 


# ################################################################################################################# 
#Logistic Regrission Classifier
print('LogisticRegression')

#saveLogisticRegClassiferBeforeAndAfterTunning()

#use saved before tunning classifier to predicit training and test sets
accuracyTrain2, accuracyTest2 = use_saved_classifierBeforeTunning("logisticRegression")
#use saved tuned classifier to predicit training and test sets
accuracyTrainTuned2, accuracyTestTuned2 = use_saved_classifierAfterTunning("LogisticRegressionTuned")


print('accuracy Train after saving base classifier = ', accuracyTrain2)
print('accuracy Test after saving base classifier = ', accuracyTest2)
print('accuracyTrainTuned after saving tuned classifier = ', accuracyTrainTuned2)
print('accuracyTestTuned after saving tuned classifier  = ', accuracyTestTuned2)
print('\n********************************************************\n')
# ################################################################################################################# 


# ################################################################################################################# 
#KNN Classifer
print('KNN')
#saveKNNClassiferBeforeAndAfterTunning()

#use saved before tunning classifier to predicit training and test sets
accuracyTrain2, accuracyTest2 = use_saved_classifierBeforeTunning("KNN")
                                     
#use saved tuned classifier to predicit training and test sets
accuracyTrainTuned2, accuracyTestTuned2 = use_saved_classifierAfterTunning("KNNTuned")


print('accuracy Train after saving base classifier = ', accuracyTrain2)
print('accuracy Test after saving base classifier = ', accuracyTest2)
print('accuracyTrainTuned after saving tuned classifier = ', accuracyTrainTuned2)
print('accuracyTestTuned after saving tuned classifier  = ', accuracyTestTuned2)
print('\n********************************************************\n')
# ################################################################################################################# 