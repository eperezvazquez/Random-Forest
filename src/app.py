import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

train_data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv', sep=",")

#Search to duplicate data
train_duplicates = train_data['PassengerId'].duplicated().sum()
train_duplicates
#Eliminating irrelevant data, not use
drop_cols = ['PassengerId','Cabin', 'Ticket', 'Name']
train_data.drop(drop_cols, axis = 1, inplace = True)
matrix = train_data.corr()
print(matrix)
plt.figure(figsize=(12, 8))
sns.heatmap(train_data.corr(), annot=True,cmap='viridis', vmax=1, vmin=-1, center=0)
plt.show()
#Checking correlation between Pclass and Fare:
plt.figure(figsize = (8, 4))
sns.boxplot(y = train_data.Pclass, x = train_data.Fare, orient = 'h', showfliers = False, palette = 'gist_heat')
plt.ylabel('Passenger Class')
plt.yticks([0,1,2], ['First Class','Second Class', 'Third Class'])
plt.show()
#Checking correlation between Pclass and Age:
#The parameter showfliers = False is ignoring the outliers. But if we do not establish that parameter, we can use boxplots to view outliers.
plt.figure(figsize = (8, 4))
sns.boxplot(y = train_data.Pclass, x = train_data.Age, orient = 'h', showfliers = False, palette = 'gist_heat')
plt.ylabel('Passenger Class')
plt.yticks([0,1,2], ['First Class','Second Class', 'Third Class'])
plt.show()
#Checking correlation between Survived and Age:
plt.figure(figsize = (8, 4))
sns.boxplot(y = train_data.Survived, x = train_data.Age, orient = 'h', showfliers = False, palette = 'gist_heat')
plt.ylabel('Survived')
plt.yticks([0,1], ['No','Si'])
plt.show()
#Let's evaluate our 'Fare' variable.
plt.figure(figsize=(6,6))
sns.boxplot(data=train_data['Fare'])
plt.title('Looking for outliers in Fare feature')
plt.ylabel('Fare')
plt.show()
fare_stat = train_data['Fare'].describe()
fare_stat
IQR = fare_stat['75%']-fare_stat['25%']
upper = fare_stat['75%'] + 1.5*IQR
lower = fare_stat['25%'] - 1.5*IQR
print('The upper & lower bounds for suspected outliers are {} and {}.'.format(upper,lower))
#visualizing data with fare above 300
train_data[train_data['Fare'] > 300]
#Dropping data with fare above 300
train_data.drop(train_data[(train_data['Fare'] > 300)].index, inplace=True)
#drop 3 outliers
train_data.shape
#get the amount missing values per columns
train_data[train_data.columns].isnull().sum().sort_values(ascending=False)
#get the percentage of missing values in each column
train_data[train_data.columns].isnull().sum().sort_values(ascending=False)/len(train_data)
# Handling Missing Values in train_data

## Fill missing AGE with Median of the survided and not survided is the same
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

## Fill missing EMBARKED with Mode
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

train_data.describe()
# We will create a new column to show how many family members of each passenger were in the Titanic.
# We will calculate it based on the sum of SibSp (siblings and spouse) and Parch  (parents and children)

print(train_data)

train_data["fam_mbrs"] = train_data["SibSp"] + train_data["Parch"]

print(train_data)
#Train data

# Encoding the 'Sex' column
train_data['Sex'] = train_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# Encoding the 'Embarked' column
train_data['Embarked'] = train_data['Embarked'].map({'S' : 0, 'C': 1, 'Q': 2})
#Verifying all our features are now numbers

train_data.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaler = scaler.fit(train_data[['Age','Fare']])
train_data[['Age','Fare']] = train_scaler.transform(train_data[['Age','Fare']])
train_data.head()

X = train_data[list(train_data.columns[1:9])]
y = train_data[['Survived']]

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=34)

accuracies = list()
nro_feature = X_train.columns.size
depth_range = range(1, nro_feature+1)

# Testearemos la profundidad de 1 a cantidad de atributos +1
for depth in depth_range:
    tree_model = DecisionTreeClassifier(criterion='entropy',
                                             min_samples_split=20,
                                             min_samples_leaf=5,
                                             max_depth = depth,
                                             random_state=0)
    tree_model.fit(X_train, y_train)
    accuracies.append(tree_model.score(X_test, y_test))
    
# Mostramos los resultados obtenidos
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))
clf = DecisionTreeClassifier(criterion='entropy',
                             min_samples_split=20,
                             min_samples_leaf=5,
                             random_state=0, max_depth=5)

clf.fit(X_train, y_train)
print('Accuracy:',clf.score(X_test, y_test))
# tree.feature_importances_ es un vector con la importancia estimada de cada atributo
for name, importance in zip(train_data.columns[1:], clf.feature_importances_):
    print(name + ': ' + str(importance))

#show predicted dataset
clf_pred=clf.predict(X_test)
cm = confusion_matrix(y_test, clf_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()

plt.show()
cm = confusion_matrix(y_test, clf_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()

plt.figure(figsize=(20,10))
plot_tree(clf)
plt.show()

X2 = train_data[['Pclass','Sex','Age']]
y2 = train_data[['Survived']]

#Split the data
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, stratify=y2,random_state=34)

clf2 = DecisionTreeClassifier(criterion='entropy',
                             min_samples_split=20,
                             min_samples_leaf=20,
                             random_state=0, max_depth=4)

clf2.fit(X2_train, y2_train)
print('Accuracy:',clf2.score(X2_test, y2_test))
#show predicted dataset
clf2_pred=clf2.predict(X2_test)
cm2 = confusion_matrix(y2_test, clf2_pred, labels=clf2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=clf2.classes_)
disp.plot()

plt.show()
plt.figure(figsize=(20,10))
plot_tree(clf2)
plt.show()
max_features = range(1,X_train.columns.size+1)
criterion = ['gini', 'entropy']
max_depth = [2,3,4,5]
parameters = dict(max_features=max_features,
                      criterion=criterion,
                      max_depth=max_depth,
                      min_samples_split=[20,30],
                      min_samples_leaf=[5,10])
clf_GS = GridSearchCV(DecisionTreeClassifier(random_state=0), parameters)
clf_GS.fit(X, y)
print('Best Criterion:', clf_GS.best_estimator_.get_params()['criterion'])
print('Best max_depth:', clf_GS.best_estimator_.get_params()['max_depth'])
print('Best min_samples_split:', clf_GS.best_estimator_.get_params()['min_samples_split'])
print('Best min_samples_leaf:', clf_GS.best_estimator_.get_params()['min_samples_leaf'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['max_features'])
clf_GS.best_estimator_.get_params()
# tree.feature_importances_ es un vector con la importancia estimada de cada atributo
for name, importance in zip(X_train.columns[0:], tree_model.feature_importances_):
    print(name + ': ' + str(importance))

#save machine learning model pythonpython by Clumsy Caribou on Mar 06 2020 Comment

clf.fit(X_train, y_train)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
 
# load the model from disk
loaded_clf= pickle.load(open(filename, 'rb'))
result = loaded_clf.score(X_test, y_test)