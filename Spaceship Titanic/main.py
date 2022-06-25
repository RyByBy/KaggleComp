import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import column_or_1d

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






'''Filling missing values with median'''
X['RoomService'].fillna(X['RoomService'].median(),inplace=True)
X_test['RoomService'].fillna(X_test['RoomService'].median(),inplace=True)
X['FoodCourt'].fillna(X['FoodCourt'].median(),inplace=True)
X_test['FoodCourt'].fillna(X_test['FoodCourt'].median(),inplace=True)
X['ShoppingMall'].fillna(X['ShoppingMall'].median(),inplace=True)
X_test['ShoppingMall'].fillna(X_test['ShoppingMall'].median(),inplace=True)
X['Spa'].fillna(X['Spa'].median(),inplace=True)
X_test['Spa'].fillna(X_test['Spa'].median(),inplace=True)
X['VRDeck'].fillna(X['VRDeck'].median(),inplace=True)
X_test['VRDeck'].fillna(X_test['VRDeck'].median(),inplace=True)
X['CryoSleep'].fillna(X['CryoSleep'].median(),inplace=True)
X_test['CryoSleep'].fillna(X_test['CryoSleep'].median(),inplace=True)

'''Filling missing Age values with mean'''
X['Age'].fillna(X['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)

'''Filling missing VIP values with false'''
X['VIP'].fillna(False,inplace=True)
X_test['VIP'].fillna(False,inplace=True)

'''Transofrming all the word values into numbers + filling rest of values'''
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
# print(X.isnull().sum())
# print(X_test.isnull().sum())
# print(X.head())

'''Creating y '''
y_train = column_or_1d(X[['Transported']], warn=False)
'''Dropping more useless values'''
X_train = X.drop(['Cabin','Transported'],axis=1)
X_test = X_test.drop(['Cabin'],axis=1)

'''Fitting and predicting into sklearn Bagging Classifier with GNB'''
clf = BaggingClassifier(base_estimator=GaussianNB(),n_estimators=10, random_state=0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
samples['Transported'] = pred
samples['Transported'] = samples["Transported"].astype(bool)
samples.to_csv('submission_bagging.csv',index=False)

'''Calculated prediction'''
print(pred.mean())

print(samples)