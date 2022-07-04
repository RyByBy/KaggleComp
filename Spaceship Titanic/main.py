import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import column_or_1d
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier


le = LabelEncoder()
pd.set_option('display.max_columns', 3000)


X = pd.read_csv('datasets/train.csv')
X_test = pd.read_csv('datasets/test.csv')
samples = pd.read_csv('datasets/sample_submission.csv')
# print(X.head())



'''Irrelevant for calculations'''
ids = X_test['PassengerId']
X = X.drop(['Name','PassengerId'],axis=1)
X_test = X_test.drop(['Name','PassengerId'],axis=1)

# print(X.isnull().sum())
# print(X_test.isnull().sum())

'''Setting Cabin number into 3 separate values'''
X[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = X['Cabin'].str.split('/', expand=True)
X_test[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = X_test['Cabin'].str.split('/', expand=True)

# print(X.head())

'''Data visualization'''
X.hist(bins = 25, figsize = (12, 12))
# plt.show()
plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
sns.countplot(x = X.HomePlanet, hue = X.Transported, palette="Set2")
plt.title('Survival figures from different Home Planets', fontsize=14)
plt.xlabel('Home planet', fontsize=15)
plt.ylabel('Number of passengers', fontsize=15)

plt.subplot(2,2,2)
sns.countplot(x = X.HomePlanet, hue = X.CryoSleep, palette="Set2")
plt.title('Survival figures from different Home Planets considering cryosleep', fontsize=14)
plt.xlabel('Home planet', fontsize=15)
plt.ylabel('Number of passengers', fontsize=15)

plt.subplot(2,2,3)
sns.boxplot(x = X.HomePlanet, y = X.Age, hue = X.Transported,  palette="Set2")
plt.title('Age range and Survival figures from different Home Planets', fontsize=14)
plt.xlabel('Home planet', fontsize=15)
plt.ylabel('Age distribution', fontsize=15)

plt.subplot(2,2,4)
sns.violinplot(x = X.HomePlanet, y = X.RoomService, hue= X.Transported, palette='Set2')
plt.title('Room survice expenditure and Survival figures from different Home Planets', fontsize=14)
plt.xlabel('Home planet', fontsize=15)
plt.ylabel('Room service expenditure', fontsize=15)

#2nd set of datas

plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
sns.violinplot(x = X.HomePlanet, y = X.ShoppingMall, hue= X.Transported, palette='Set2')
plt.title('Shopping Mall expenditure and Survival figures from different Home Planets', fontsize=14)
plt.xlabel('Home planet', fontsize=15)
plt.ylabel('Number of passengers', fontsize=15)

plt.subplot(2,2,2)
sns.violinplot(x = X.HomePlanet, y = X.FoodCourt, hue= X.Transported, palette='Set2')
plt.title('Food Court expenditure and Survival figures from different Home Planets', fontsize=14)
plt.xlabel('Home planet', fontsize=15)
plt.ylabel('Number of passengers', fontsize=15)

plt.subplot(2,2,3)
sns.violinplot(x = X.HomePlanet, y = X.Spa, hue= X.Transported, palette='Set2')
plt.title('Spa expenditure and Survival figures from different Home Planets', fontsize=14)
plt.xlabel('Home planet', fontsize=15)
plt.ylabel('Number of passengers', fontsize=15)

plt.subplot(2,2,4)
sns.violinplot(x = X.HomePlanet, y = X.VRDeck, hue= X.Transported, palette='Set2')
plt.title('VRDeck and Survival figures from different Home Planets', fontsize=14)
plt.xlabel('Home planet', fontsize=15)
plt.ylabel('Number of passengers', fontsize=15)

plt.show()

'''Filling missing values'''
X['RoomService'].fillna(X['RoomService'].median(),inplace=True)
X_test['RoomService'].fillna(X_test['RoomService'].median(),inplace=True)
X['FoodCourt'].fillna(X['FoodCourt'].median(),inplace=True)
X_test['FoodCourt'].fillna(X_test['FoodCourt'].median(),inplace=True)
X['ShoppingMall'].fillna(X['ShoppingMall'].median(),inplace=True)
X_test['ShoppingMall'].fillna(X_test['ShoppingMall'].median(),inplace=True)
X['Spa'].fillna(X['Spa'].median(),inplace=True)
X_test['Spa'].fillna(X_test['Spa'].median(),inplace=True)
X['VRDeck'].fillna(X['VRDeck'].mean(),inplace=True)
X_test['VRDeck'].fillna(X_test['VRDeck'].mean(),inplace=True)
X['CryoSleep'].fillna(False,inplace=True)
X_test['CryoSleep'].fillna(False,inplace=True)

'''Filling missing Age values with mean'''
X['Age'].fillna(X['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)

'''Filling missing VIP values with false'''
X['VIP'].fillna(False,inplace=True)
X_test['VIP'].fillna(False,inplace=True)

print(X['Cabin_deck'].value_counts())
print(X['Cabin_side'].value_counts())


'''Filling Cabins'''
for length_cnt, length in enumerate(X['Cabin_deck']):
    X['Cabin_deck'].fillna('F',inplace=True)
for length_cnt, length in enumerate(X['Cabin_side']):
    X['Cabin_side'].fillna('S',inplace=True)
for length_cnt, length in enumerate(X['Cabin_num']):
    X['Cabin_num'].fillna(length_cnt)
#
for length_cnt, length in enumerate(X_test['Cabin_deck']):
    X_test['Cabin_deck'].fillna('F',inplace=True)
for length_cnt, length in enumerate(X_test['Cabin_side']):
    X_test['Cabin_side'].fillna('S',inplace=True)
for length_cnt, length in enumerate(X_test['Cabin_num']):
    X_test['Cabin_num'].fillna(length_cnt)

'''Transofrming all the word values into numbers'''
X['HomePlanet'] = le.fit_transform(X['HomePlanet'])
X_test['HomePlanet'] = le.fit_transform(X_test['HomePlanet'])
X['CryoSleep'] = le.fit_transform(X['CryoSleep'])
X_test['CryoSleep'] = le.fit_transform(X_test['CryoSleep'])
X['Cabin_deck'] = le.fit_transform(X['Cabin_deck'])
X_test['Cabin_deck'] = le.fit_transform(X_test['Cabin_deck'])
X['Cabin_side'] = le.fit_transform(X['Cabin_side'])
X_test['Cabin_side'] = le.fit_transform(X_test['Cabin_side'])
X['Cabin_num'] = le.fit_transform(X['Cabin_num'])
X_test['Cabin_num'] = le.fit_transform(X_test['Cabin_num'])
X['Destination'] = le.fit_transform(X['Destination'])
X_test['Destination'] = le.fit_transform(X_test['Destination'])
X['VIP'] = le.fit_transform(X['VIP'])
X_test['VIP'] = le.fit_transform(X_test['VIP'])
X['Transported'] = le.fit_transform(X['Transported'])


'''Checking for missing values'''
print(X.isnull().sum())
print(X_test.isnull().sum())
print(X.head())

'''Creating y '''
y = column_or_1d(X[['Transported']], warn=False)

'''Dropping more useless values'''
train_data = X.drop(['Cabin','Transported'],axis=1)
test_data = X_test.drop(['Cabin'],axis=1)

'''Testing'''
X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.2,random_state=42, stratify=y)

'''Fitting and predicting'''
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_val)

preds_test = clf.predict(test_data)
samples['Transported'] = preds_test
samples['Transported'] = samples["Transported"].astype(bool)
samples.to_csv('submission.csv',index=False)

scores = accuracy_score(y_val, pred)
print("Accuracy: ",scores)