import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, \
    f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 999999)


def get_description(df):
    print(df.head())
    print(df.shape)
    df.info()
    print(df.describe())
    print(df.isnull().sum())
    print(df.size)
    df.drop_duplicates(inplace=True)
    print(df.shape)


def plot(ytest, prediction):
    plt.scatter(range(len(ytest)), ytest, marker='o')
    plt.scatter(range(len(prediction)), prediction, marker='x')
    plt.show()


df = pd.read_csv('zoo.csv')
df1 = pd.read_csv('class.csv')
df['class_type'] = LabelEncoder().fit_transform(df['class_type'])
get_description(df)
get_description(df1)
'This dataset displays the seven different classes of animals, along with defining traits of each'
'It can be used to train ML models and predict classes of animals.'
'It features two different datasets, one for predicting, which contains the actual important values and the other for information, as an extension of the first.'
sns.heatmap(data=df.select_dtypes(int).corr(), annot=True)
plt.show()
# sns.countplot(x='backbone',data=df,hue='legs')
# plt.show()
sns.countplot(data=df, x='backbone', hue='class_type')
plt.show()
# sns.countplot(data=df,x='backbone',hue='tail')
# plt.show()
sns.countplot(data=df, x='aquatic', hue='class_type')
plt.show()
sns.countplot(data=df, x='class_type', hue='hair')
plt.show()
x = df.drop(['animal_name', 'class_type'], axis=1)
y = df['class_type']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=32)
print('LogisticRegression')
params = {'C': [0.001, 0.01, 0.01, 0.1, 1, 10, 50, 80, 100, 200]}
g = GridSearchCV(LogisticRegression(max_iter=9999999999999999999), params, cv=5)
g.fit(xtrain, ytrain)
model = LogisticRegression(C=g.best_params_['C'], max_iter=9999999999999999999)
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
print(accuracy_score(ytest, pred), precision_score(ytest, pred, average='weighted', zero_division=0),
      f1_score(ytest, pred, average='weighted'))
c = ConfusionMatrixDisplay(confusion_matrix(ytest, pred))
c.plot()
plt.show()
print('DecisionTreeClassifier')
params = {'max_depth': [1, 3, 5, 10, 20, 50, 80, 100, 200], 'criterion': ['entropy', 'gini', 'log_loss']}
g = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
g.fit(xtrain, ytrain)
model = DecisionTreeClassifier(max_depth=g.best_params_['max_depth'], criterion=g.best_params_['criterion'])
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
print(accuracy_score(ytest, pred), precision_score(ytest, pred, average='weighted', zero_division=0),
      f1_score(ytest, pred, average='weighted'))
c = ConfusionMatrixDisplay(confusion_matrix(ytest, pred))
c.plot()
plt.show()
print('RandomForestClassifier')
g = GridSearchCV(RandomForestClassifier(), params, cv=5)
g.fit(xtrain, ytrain)
model = RandomForestClassifier(max_depth=g.best_params_['max_depth'], criterion=g.best_params_['criterion'])
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
print(accuracy_score(ytest, pred), precision_score(ytest, pred, average='weighted', zero_division=0),
      f1_score(ytest, pred, average='weighted'))
c = ConfusionMatrixDisplay(confusion_matrix(ytest, pred))
c.plot()
plt.show()
print('KNeighborsClassifier')
params = {'n_neighbors': [1, 2, 3, 4, 5], 'metric': ['euclidean', 'manhattan']}
g = GridSearchCV(KNeighborsClassifier(), params, cv=5)
g.fit(xtrain, ytrain)
model = KNeighborsClassifier(n_neighbors=g.best_params_['n_neighbors'], metric=g.best_params_['metric'])
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
print(accuracy_score(ytest, pred), precision_score(ytest, pred, average='weighted', zero_division=0),
      f1_score(ytest, pred, average='weighted'))
c = ConfusionMatrixDisplay(confusion_matrix(ytest, pred))
c.plot()
plt.show()
print('XGBoost Classifier')
params = {'max_depth': [1, 3, 5, 10, 20, 50, 80, 100, 200], 'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]}
g = GridSearchCV(XGBClassifier(), params, cv=5)
g.fit(xtrain, ytrain)
model = XGBClassifier(max_depth=g.best_params_['max_depth'], learning_rate=g.best_params_['learning_rate'])
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
print(accuracy_score(ytest, pred), precision_score(ytest, pred, average='weighted', zero_division=0),
      f1_score(ytest, pred, average='weighted'))
c = ConfusionMatrixDisplay(confusion_matrix(ytest, pred))
c.plot()
plt.show()
print('SVC')
params = {'C': [0.001, 0.01, 0.01, 0.1, 1, 10, 50, 80, 100, 200], 'kernel': ['poly', 'linear', 'rbf', 'sigmoid']}
g = GridSearchCV(SVC(), params, cv=5)
g.fit(xtrain, ytrain)
model = SVC(C=g.best_params_['C'], kernel=g.best_params_['kernel'])
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
print(accuracy_score(ytest, pred), precision_score(ytest, pred, average='weighted', zero_division=0),
      f1_score(ytest, pred, average='weighted'))
c = ConfusionMatrixDisplay(confusion_matrix(ytest, pred))
c.plot()
plt.show()
'''According to the given scores, SVC gave the optimal results, with an accuracy score of 95.23%.'''

