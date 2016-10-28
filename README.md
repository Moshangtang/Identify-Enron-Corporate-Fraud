# Identify-Enron-Corporate-Fraud
Investigated the Enron email corpus data with Gaussian naive Bayesian, Support Vector Machine and Decision Tree machine learning techniques.

## Enron Submission Free-Response Questions
1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were
there any outliers in the data when you got it, and how did you handle
those? [relevant rubric items: “data exploration”, “outlier investigation”]
Ans: The goal is to use features like total_payments, emails to determine
who is the person of interest. The supervised classification is useful since
all the features with labels (poi/non-poi) on it. We could train the classifier
and then use it for prediction.
After exploring the data, I found that the total number of data is 146 and the
number of poi is 18. The total number of feature is 21. There are a significant
amount of null-values in this data. This is understandable as the data is
mostly financial data of the employees and not every employee has all the
fields of financial values. But the person named ‘BELFER, ROBERT’ did not
contain most of the financial values. So I delete this item to avoid bias of the
data. The item with key of ‘TOTAL’ was included which summed all person’s
financial data. So I removed it. I also removed the item with key of ‘THE
TRAVEL AGENCY IN THE PARK’ since it is not considered as enron
employee.
2. What features did you end up using in your POI identifier, and what
selection process did you use to pick them? Did you have to do any scaling?
Why or why not? As part of the assignment, you should attempt to engineer
your own feature that does not come ready-made in the dataset -- explain
what feature you tried to make, and the rationale behind it. (You do not
necessarily have to use it in the final analysis, only engineer and test it.) In
your feature selection step, if you used an algorithm like a decision tree,
please also give the feature importances of the features that you use, and
if you used an automated feature selection function like SelectKBest, please
report the feature scores and reasons for your choice of parameter
values. [relevant rubric items: “create new features”, “properly scale
features”, “intelligently select feature”]
Ans: I created a feature of total_package as sum of total_payment and
total_stock_value since I thought this might be a good feature. However,
after looking up the importance value from decision tree classifier, I decided
not to include it. When I use SVM as a classifier, I use selectKbest to select
the best 4 features. When I use descision tree as a classifier, I use
importance value to select features.
The initial features I chose are: ['poi',’total_package’, 'total_payments',
'total_stock_value', 'salary','bonus', 'long_term_incentive', 'expenses',
'from_this_person_to_poi', 'from_poi_to_this_person']
The importances for them are: [0.06674035. 0. 0.15131931 0.
0.28242642 0. 0.56625427 0. 0. ]
According to the importances I got, I removed one or two of the features,
then run the program again to get the importances, then removed features
again. After several times, I remove all the features denoted as red. The
final features I chose are :[ 'poi', 'total_stock_value', 'bonus', 'expenses']
with the importances of [ 0.19218724 0.30202169 0.50579107].
I also scaled the features using max_min_scaler this time, however, it did
not give me a better result, so I decided not to use it.
3. What algorithm did you end up using? What other one(s) did you try? How
did model performance differ between algorithms? [relevant rubric item:
“pick an algorithm”]
Ans: I end up using decision tree as the classifier which give me almost 0.4
for both of the precision and recall value. I tried GaussianNB which
precision and recall value are not higher than 0.3. I also tried SVM which
even did not give me some true positive data.
4. What does it mean to tune the parameters of an algorithm, and what can
happen if you don’t do this well? How did you tune the parameters of your
particular algorithm? (Some algorithms do not have parameters that you
need to tune -- if this is the case for the one you picked, identify and briefly
explain how you would have done it for the model that was not your final
choice or a different model that does utilize parameter tuning, e.g. a
decision tree classifier). [relevant rubric item: “tune the algorithm”]
Ans: We have to tune the parameter of an algorithm in order to get the best
metric score like precison, recall or f1 score. If we did not do this, the
classifier may either be overfitted or underfitted.
I used GridSearchCV for parameter tune. The turned parameter are
{'criterion': ('gini','entropy'), 'splitter':('best','random'), 'min_samples_split':
[2,5,10,15,20], 'max_leaf_nodes: [5,10,30,50,100]}. Then I chose the best
parameter combination through f1 score.
5. What is validation, and what’s a classic mistake you can make if you do it
wrong? How did you validate your analysis? [relevant rubric item:
“validation strategy”]
Ans: validation is a method to make sure the results of an algorithm will
generalize to an independent data set by choosing and setting
training/testing dataset in properly way. If we use the same dataset for
training and testing, the metric score must be too high to believe. If the
training data set and testing dataset are shaped in different way. The metric
score must be too low to be true.
In this project, I used StratifiedShuffleSplit model to prepare the training and
testing dataset which is the same as in tester file.
6. Give at least 2 evaluation metrics and your average performance for each
of them. Explain an interpretation of your metrics that says something
human-understandable about your algorithm’s performance. [relevant
rubric item: “usage of evaluation metrics”]
Ans: I use precision, recall and f1 score to evaluate algorithm performance.
Precision is the probability of a given perdition to be accurate. If precision
score is high, the numbers that are predicted to be true but actually not will
be less. Recall is the percentage of a given class that is predicted
accurately. I want them both to be high. If recall is high, the number that are
predicted to be false but actually is true will be less. After carefully choosing
the feature, the precision of my classifier is about 0.40, the recall is around
0.40. The F1 score is a metric for combination of precision and recall, I use
it to choose the best parameters for classifier. The F1 score is about 0.40.
