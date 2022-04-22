import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, make_scorer, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from scipy import stats



## 1

X = np.arange(0.0, 1.0, 0.01)
Y = X[::-1]


def Gini_impurity(P1, P2):
    denom = P1 + P2
    Gini_index = 2 * (P1 / denom) * (P2 / denom)
    return Gini_index


Gini = Gini_impurity(X, Y)
plt.plot(X, Gini)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.title('Gini Impurity Graph')
plt.xlabel('P1=1')
plt.ylabel('Impurity Measure')
plt.ylim([0, 1.1])
plt.show()

## 2

# In the case of KNN in Regression, we will be using almost the same
# approach as Classification. There will be a one difference where
# we won't be taking the majority class label for the test sample.
# Instead, we will be taking the average of all the class labels and set the value as a class
# label for the test sample
#

path = "ML Test/data_cleaned.csv"
data = pd.read_csv(path)

data.head()

x = data.drop(['Survived'], axis=1)
y = data['Survived']

# scaling the data
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# new scaled data
x = pd.DataFrame(x_scaled, columns=x.columns)
x.head()

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, stratify=y)

# Model
classifier = knn(n_neighbors=12)
classifier.fit(x_train, y_train)

test_prediction = classifier.predict(x_test)
k = f1_score(test_prediction, y_test)
print('Test F1 score  ', k)


# def elbow curve

def elbow(K):
    test_err = []

    # train the model for every value of k
    for i in K:
        # init a Knn instance
        classify = knn(n_neighbors=i)
        classify.fit(x_train, y_train)

        # appending f1 scores to a list calculated using the predictions
        temp = classify.predict(x_test)
        temp = f1_score(temp, y_test)
        err = 1 - temp
        test_err.append(err)

    return test_err


# def a range of k
k_range = range(6, 20, 2)

# call the elbow func
test = elbow(k_range)

# plotting
plt.plot(k_range, test)
plt.xlabel('K neighbors')
plt.ylabel('Test error')
plt.title('Elbow Curve')


# seems that k=12 is the optimal value of k for the data(lowest error)
# before when we trained the model the f1 score was 0.6746 with k of 5, now with k of 12 the f1 score is: 0.7152


## 3

# Over fitting occurs when your model learns too much from training data
# and isn’t able to generalize the underlying information.
# When this happens, the model is able to describe training data very accurately but loses
# precision on every dataset it has not been trained on.
# This is not optimal because we want our model to be reasonably good on data that it has never seen before.
#
# some ways to avoid over fitting:
# 1. Train with more data. collect more data as a way of increasing the accuracy of the model (could be expensive)
# 2. Data Augmentation. Data augmentation makes a sample data look slightly different every time
# it is processed by the model
# 3. Data simplification. The data simplification method is used to reduce over fitting
# by decreasing the complexity of the model to make it simple enough that it does not over fit.
# Some of the actions that can be implemented include pruning a decision tree,
# reducing the number of parameters in a neural network, and using dropout on a neutral network.
# 4. Ensemble. Ensembling is a machine learning technique that works by combining predictions
# from two or more separate models. The most popular ensembling methods include boosting and bagging.
# Boosting works by using simple base models to increase their aggregate complexity.
# It trains a large number of weak learners arranged in a sequence,
# such that each learner in the sequence learns from the mistakes of the learner before it.
# Bagging works by training a large number of strong learners arranged in a parallel pattern
# and then combining them to optimize their predictions.


## 4

# Random forest consists of a large number of individual decision trees that operate as an ensemble. Each individual
# tree in the random forest output's a class prediction and the class with the most votes becomes our model’s
# prediction. A large number of relatively uncorrelated trees operating as a committee will outperform any of the
# individual constituent models.
#    Decision Tree                         vs.                   Random Forest
#  Possibility of overfitting                                 prevent overfitting
#  Less accurate results                                      More accurate results
#  Simple/Easy to interpret                                   Hard to interpret
#  Less computation                                           More computation time
#  Simple to visualize                                        Complex to visualize
#  Fast processing                                            Slow processing
#                                                             Builds a robust model
#                                                             Can use Classifications and Regression problems
# Provides a clear idea of what all features are important for classification.


## 5
# Answered in question 3



## 6






## 7
# Ridge and Lasso regularization
# Lasso/l1 L1 regularization term is the sum of absolute values of each element. For a length N vector, it would be |w[1]| + |w[2]| + ... + |w[N]|.
# (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
# Ridge/l2 L2 regularization term is the sum of squared values of each element. For a length N vector, it would be w[1]²  + w[2]²  + ... + w[N]²
# ||y - Xw||^2_2 + alpha * ||w||^2_2

class RidgeRegression():

    def __init__(self, learning_rate, iterations, l2_penality):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_penality = l2_penality

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape

        # weight initialization
        self.W = np.zeros(self.n)

        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning

        for i in range(self.iterations):
            self.update_weights()
        return self

    # Helper function to update weights in gradient descent

    def update_weights(self):
        Y_pred = self.predict(self.X)

        # calculate gradients
        dW = (- (2 * (self.X.T).dot(self.Y - Y_pred)) +
              (2 * self.l2_penality * self.W)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    # Hypothetical function  h( x )
    def predict(self, X):
        return X.dot(self.W) + self.b


class LassoRegression():

    def __init__(self, learning_rate, iterations, l1_penality):

        self.learning_rate = learning_rate

        self.iterations = iterations

        self.l1_penality = l1_penality

    # Function for model training

    def fit(self, X, Y):

        # no_of_training_examples, no_of_features

        self.m, self.n = X.shape

        # weight initialization

        self.W = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y

        # gradient descent learning

        for i in range(self.iterations):
            self.update_weights()

        return self

    # Helper function to update weights in gradient descent

    def update_weights(self):

        Y_pred = self.predict(self.X)

        # calculate gradients

        dW = np.zeros(self.n)

        for j in range(self.n):

            if self.W[j] > 0:

                dW[j] = (- (2 * (self.X[:, j]).dot(self.Y - Y_pred))

                         + self.l1_penality) / self.m

            else:

                dW[j] = (- (2 * (self.X[:, j]).dot(self.Y - Y_pred))

                         - self.l1_penality) / self.m

        db = - 2 * np.sum(self.Y - Y_pred) / self.m

        # update weights

        self.W = self.W - self.learning_rate * dW

        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function  h( x )

    def predict(self, X):

        return X.dot(self.W) + self.b

## 8

data = pd.read_csv('ML Test/BankChurners.csv')
data.head()

data = data.drop(['CLIENTNUM'],axis=1)
data = data.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1'], axis=1)
data = data.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)
data.head()

count=pd.value_counts(data['Attrition_Flag']).tolist()
plt.figure(figsize=(11,11))
plt.title("Percentage of Attrited Customer and Existing Customer")
plt.pie(x=count, labels=["Attrited Customer", "Existing Customers"],autopct='%.2f%%')


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,15))

attrited_gender = data.loc[data["Attrition_Flag"] == "Attrited Customer", ["Gender"]].value_counts().tolist()
ax1.pie(x=attrited_gender,labels=["Male","Female"],autopct='%.2f%%')
ax1.set_title('Gender vs Attrited Customer')

existing_gender=data.loc[data["Attrition_Flag"] == "Existing Customer", ["Gender"]].value_counts().tolist()
ax2.pie(x=existing_gender,labels=["Male","Female"],autopct='%.2f%%')
ax2.set_title('Gender vs Existing Customer')


plt.figure(figsize=(28,11))
plt.title("Distribution of Age with respect to Churned or not")
sns.countplot(data=data,x=data["Customer_Age"],hue="Attrition_Flag")

fig,ax=plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=True)

sns.boxplot(x=data['Total_Ct_Chng_Q4_Q1'])

columns = ["Customer_Age", 'Dependent_count', 'Months_on_book',
           'Total_Relationship_Count', 'Months_Inactive_12_mon',
           'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
           'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
           'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
print(data.shape)
for column in columns:
    z = np.abs(stats.zscore(data[column]))
    data = data[(z < 3)]
print(data.shape)

sns.boxplot(x=data['Total_Ct_Chng_Q4_Q1'])


X = data.drop("Attrition_Flag",axis=1)
Y = data["Attrition_Flag"]

categorical_col = ['Gender','Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
for cols in categorical_col:
    le = LabelEncoder()
    X[cols] = le.fit_transform(X[cols])
Y = le.fit_transform(Y)
data.info()


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

classifiers = [[RandomForestClassifier(), 'Random Forest'], [KNeighborsClassifier(), 'K-Nearest Neighbours']]
score_list = []
cross_val_list = []

for classifier in classifiers:
    model = classifier[0]
    model.fit(X_train, Y_train)
    model_name = classifier[1]
    prediction = model.predict(X_test)

    scores = model.score(X_test, Y_test)
    cross_val = cross_val_score(model, X_test, Y_test).mean()

    score_list.append(scores)
    cross_val_list.append(cross_val)

    print(model_name, "Score :" + str(round(scores * 100, 2)) + '%')
    print(model_name, "Cross Validation Score :" + str(round(cross_val * 100, 2)) + '%')





































