import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#using Kfold and cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../data/raw/train.csv')
test = pd.read_csv('../data/raw/test.csv')

target = train['Survived']
train = train.drop('Survived', axis=1) #dropping survived column 
data = train.append(test) #concatenating both train and test data for pre processing
data.head()
target = pd.DataFrame(target)
target.head()
data.isnull().sum()
data.drop(['Age', 'Cabin'], axis = 1, inplace = True)
#filling missing fare value with mean value
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
#using forward-fill to fill the missing value in the Embarked column
data['Embarked'] = data['Embarked'].fillna(method = 'ffill')
data.isnull().sum() #all the missing values have been catered for through data munging
data.info() #all the missings value have been catered for
#preprocessing and dropping of some columns with low correlation
data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace = True) 
data['Embarked'].unique() 
data['Sex'].unique()
#transforming columns with one_hot_encoding
embarked_dummies = pd.get_dummies(data.Embarked)
data = pd.concat([data,embarked_dummies], axis=1)
data = data.drop("Embarked", axis=1)

sex_dummies= pd.get_dummies(data.Sex)
data = pd.concat([data,sex_dummies], axis=1)
data = data.drop("Sex", axis=1)
(data.shape) #train and train data shape
new_train_data = data.iloc[:891,]
new_train_data
new_test_data = data.iloc[891:,]
new_test_data

estimators = {
    'Logisitic_Regression': LogisticRegression(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC(gamma = 'auto'),
    'GaussianNB': GaussianNB(),
    'discriminant_analysis': LinearDiscriminantAnalysis()

for name, code in estimators.items():
    kfold = StratifiedKFold(n_splits=10, random_state=11 , shuffle=True)
    cv_result = cross_val_score(code, X = new_train_data, y = target,
                            cv = kfold, scoring= 'accuracy')
    print(f'{name:>20}: ' + 
          f'mean accuracy={cv_result.mean():.2%}; ' +
          f'standard deviation={cv_result.std():.2%}')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_train_data, target, random_state=11,test_size = 0.3)

display(X_train.shape)
display(X_test.shape)
display(y_train.shape)
display(y_test.shape)

def get_score(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=11)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    acc_score = accuracy_score(y_test, preds_val)
    return(acc_score)
my_score = get_score(X_train, X_test, y_train, y_test)
print(f'The accuracy score of the model is: {(my_score*100):.2f}%.')
# XGBoost for result improvement
from xgboost import XGBClassifier

my_model1 = XGBClassifier(random_state = 11)
my_model1.fit(X_train, y_train)

preds = my_model1.predict(X_test)
acc = (accuracy_score(y_test, preds))* 100
print(f'Accuracy Score: {acc:.2f}%')
cm = confusion_matrix(y_test, preds)
print(f'Confusion Matrix: \n{cm}')

sns.heatmap(cm, annot=True, cmap='nipy_spectral_r')
plt.show()
crpt = classification_report(y_test, preds)
print(f'Classification Report: \n\n{crpt}')

# save the model to disk
filename = 'finalized_model1.sav'
pickle.dump(clf, open(filename, 'wb'))
 
# load the model from disk
loaded_clf= pickle.load(open(filename, 'rb'))
result = loaded_clf.score(X_test, y_test)
