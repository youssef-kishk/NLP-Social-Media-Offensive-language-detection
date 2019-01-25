# NLP-Social-Media-Offensive-language-detection
Offensive language is pervasive in social media. Individuals frequently take advantage of the perceived anonymity of computer-mediated communication, using this to engage in behavior that many of them would not consider in real life. Online communities, social media platforms, and technology companies have been investing heavily in ways to cope with offensive language to prevent abusive behavior in social media.


This project in a Natural Language Processing(NLP) application on determining whether a social media tweet,post is offensive or not.


Attached the complete project report,Source code and the training data set used,but notice that the data set is unbalanced.


The project passses through several steps to reach the final goal of determing whether the social media post,tweet is offensive or not.
The steps are:
1-Reading the training Dataset
2-Cleaning of the dataset and removing the noise using stop words
3-Lexicon Normalization uding Lemmitization
4-Features extraction on text data using bag of words model
5-Several classifiers are used to notice the difference between them, the used classifiers in the project are:
Random Forest Classifier ,Naïve Bayes Classifier ,Decision Tree Classifier ,Logistic Regression Classifier ,K-Nearest Neighbor Classifier.
6-The final step is tunning of the results to improve it using K-Fold cross validation.



For Testing the project, comment all the classifers code except the calling of the saving of the classifer function which is already commented in the uploaded code, run the project to save the classifers and after it finishes return the project as its uploaded to review the results.


Note:Saving the classifers would take some period of time
