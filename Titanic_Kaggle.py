#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:02:23 2018

@author: vishal.bule
"""

import pandas as pa
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pylab as plot
 
path = '/Users/vishal.bule/Desktop/ML/MY_ML/Data/Titanic_train.csv'
train = pa.read_csv(path)

#I - Exploratory data analysis
#************************************

#Pandas allows you to have a sneak peak at your data.
train.head
train.shape
train.describe 

'''
PassengerId: and id given to each traveler on the boat
Pclass: the passenger class. It has three possible values: 1,2,3 (first, second and third class)
The Name of the passeger
The Sex
The Age
SibSp: number of siblings and spouses traveling with the passenger
Parch: number of parents and children traveling with the passenger
The ticket number
The ticket Fare
The cabin number
The embarkation. This describe three possible areas of the Titanic from which the people embark. Three possible values S,C,Q
'''

#fill in the null values with the median age
train['Age'] = train['Age'].fillna(train['Age'].median())


# How much percentage man and whomen in entire data set  => Man=66%  &  Woman=34% 
train['Sex'].value_counts().sort_index().plot.bar()
        #or
sns.countplot(train['Sex'])
        #or  Percentange
(train['Sex'].value_counts()/len(train)).plot.bar()

#Let's visualize survival using panda api with stacked=True.
train['Died']= 1 - train['Survived']

train.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar',colors=['g','r'],figsize=(10, 5),stacked=True)
train.groupby('Sex').agg('mean')[['Survived','Died']].plot(kind='bar',colors=['g','r'],figsize=(10, 5),stacked=True)


##Let's visualize survival using panda api without stacked '''
train.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar',colors=['g','r'],figsize=(10, 5))
train.groupby('Sex').agg('mean')[['Survived','Died']].plot(kind='bar',colors=['g','r'],figsize=(10, 5))

#As a matter of fact, the ticket fare correlates with the class as we see it in the chart below
train.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(10, 5))
#or 
train.groupby('Pclass').agg('mean')['Fare'].plot(kind='bar', figsize=(10, 5))
#or 
train.groupby('Pclass').agg('sum')['Fare'].plot(kind='bar', figsize=(10, 5))


#Let's visualize survival using seaborn api.
#Let's now correlate the survival with the age variable. '''
fig = plt.figure(figsize=(10, 5))
sns.violinplot(x='Sex',
               y='Age',
               hue='Survived',
               data=train,
               figsize=(10, 5),
               split=True,
               palette={0:'r',1:'g' }
               )

#Let's visualize survival using matplot/pyplot api.
#Let's now focus on the Fare ticket of each passenger and see how it could impact the survival.'''
figure = plt.figure(figsize=(10,5))                                                       
plt.hist([train[train['Survived'] == 1]['Fare'], 
          train[train['Survived'] == 0]['Fare']
         ], 
         stacked=True,
         color = ['g','r'],
         bins = 50, 
         label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passenger')
plt.legend()

#Let's visualize survival using matplot/pyplot api.
#Let's now combine the age, the fare and the survival on a single chart.'''
figure = plt.figure(figsize=(12,7))
ax = plt.subplot()
ax.scatter(train[train['Survived'] == 1]['Age'],
		  train[train['Survived'] == 1]['Fare'], 
           c='green', s=train[train['Survived'] == 1]['Fare']
           )
ax.scatter(train[train['Survived'] == 0]['Age'],
		  train[train['Survived'] == 0]['Fare'], 
           c='red', s=train[train['Survived'] == 0]['Fare']
           );
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title("The size of the circles is proportional to the ticket fare.")

#Let's visualize survival using seaborn api.
#Let's now see how the embarkation site affects the survival. ''' 
figure = plt.figure(figsize=(12,7))
sns.violinplot(x='Embarked',
               y='Fare',
               hue='Survived',
               data=train,
               figsize=(10, 5),
               split=True,
               palette={0:'r',1:'g' }
               )

#II - Feature engineering
#************************************
def status(feature):
    print 'Processing', feature, ': ok'

def get_combined_data():
    path = '/Users/vishal.bule/Desktop/ML/MY_ML/Data/Titanic_train.csv'
    train_data = pa.read_csv(path)
    
    path1 = '/Users/vishal.bule/Desktop/ML/MY_ML/Data/Titanic_test.csv'
    test_data = pa.read_csv(path1)
    
    # extracting and then removing the targets from the training data
    target=train_data['Survived']
    train_data.drop(['Survived'], 1, inplace=True)
    
    combined =train_data.append(test_data)
    combined.reset_index(inplace=True)
    # we'll also remove the PassengerID since this is not an informative feature
    combined.drop(['index','PassengerId'], inplace=True, axis=1)
    
    return combined

    
    
combined=get_combined_data()

str="Braund, Mr. Owen Harris" 
ans=str.split(',')
    
titles = set()
for name in train['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

def get_titile():
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    combined['Title'] = combined['Title'].map(Title_Dictionary)
    status('Title')
    return combined

combined=get_titile()

combined[combined['Title'].isnull()]
combined[combined['Fare'].isnull()].sum()
combined[combined['Embarked'].isnull()].sum()
combined[combined['Cabin'].isnull()].sum()

print combined.Embarked.isnull().sum()
print combined.Embarked.isnull().sum()

print combined.iloc[:891].Embarked.isnull().sum()


#Processing the ages
print combined.iloc[:891].Age.isnull().sum()
print combined.iloc[891:].Age.isnull().sum()

group_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])

group_train_median = group_train.median()
group_train_median = group_train_median.reset_index()[['Sex','Pclass','Title','Age']]


def fill_age(row):
    condition = (
                    (group_train_median['Sex'] == row['Sex']) &
                    (group_train_median['Title'] == row['Title']) & 
                    (group_train_median['Pclass'] == row['Pclass'])
                )
    return group_train_median[condition]['Age'].values[0]
 
def process_age():
    global combined
    # function that fills the missing values of the age vairable 
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'],axis=1 ) 
    status('Age')
    return combined

combined = process_age()

#process the names.
def process_name():
    global combined
    # we'll also remove the Name since this is not an informative feature
    combined.drop(['Name'], inplace=True, axis=1)
    
    # encoding in dummy variable
    title_dummies = pa.get_dummies(combined['Title'],prefix='Title')
    combined = pa.concat([combined,title_dummies],axis=1)
    
    #removing the title variable
    combined.drop(['Title'], inplace=True, axis=1)
    status('Name')
    return combined

combined = process_name()


# process the fare. imputed the missing fare value by the average fare computed on the train set
def process_fares():
     global combined
     # there's one missing fare value - replacing it with the mean.
     combined.Fare.fillna(combined.iloc[:891].Fare.mean(),inplace=True)
     status('Fare')
     return combined

combined = process_fares()

# process the Embarked
def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    combined.Embarked.fillna('S', inplace=True)
    
    # encoding in dummy variable
    embarked_dummies = pa.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pa.concat([combined,embarked_dummies],axis=1)
    #removing the Embarked variable
    combined.drop(['Embarked'], inplace=True, axis=1)
    status('Embarked')
    return combined

combined = process_embarked()
    
# Processing Cabin
train_cabin,test_cabin = set(),set()

for c in combined.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('U')
        
for c in combined.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('U')
        
print train_cabin
print test_cabin

#process cabin
def process_cabin():
    global combined
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    
    # dummy encoding ..
    cabin_dummies = pa.get_dummies(combined['Cabin'], prefix = 'Cabin')
    combined = pa.concat([combined,cabin_dummies], axis=1)
    
    #removing the Cabin variable
    combined.drop(['Cabin'], inplace=True, axis=1)
    
    status('Cabin')
    return combined

combined = process_cabin()

#process sex
def process_sex():
    global combined
    # mapping string values to numerical one
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    status('Sex')
    
    return combined

combined = process_sex()

#Processing Pclass
def process_pclass():
    global combined
    
    # dummy encoding for pclass
    pclass_dummies = pa.get_dummies(combined['Pclass'],prefix = 'Pclass')
    combined=pa.concat([combined,pclass_dummies], axis=1)
    
    #removing the Cabin variable
    combined.drop(['Pclass'], inplace=True, axis=1)
    
    status('Pclass')
    return combined

combined = process_pclass()

#Processing Ticket

def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = map(lambda t: t.strip(),ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket ))
    if len(ticket) > 0 :
        return ticket[0]
    else:
        return 'XXX'

tickets = set()
for t in combined['Ticket']:
    tickets.add(cleanTicket(t))


print len(tickets)

def process_ticket():
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if  ticket is a digit
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(),ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket ))
        if len(ticket) > 0 :
            return ticket[0]
        else:
            return 'XXX'
    
    # Extracting dummy variables from tickets:
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    ticket_dummies = pa.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pa.concat([combined,ticket_dummies], axis=1)
    
    #removing the Cabin variable
    combined.drop(['Ticket'], inplace=True, axis=1)
    
    status('Ticket')
    return combined

combined = process_ticket()
    

    
       
def process_family():
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    
    status('family')
    return combined

combined = process_family()

    
    
#III - Modeling    
#************************************

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV



    
def compute_score(clf, X, y, scoring='accuracy' ):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)

def recover_train_test_target():
    global combined
    
    path = '/Users/vishal.bule/Desktop/ML/MY_ML/Data/Titanic_train.csv'
    #target = pa.read_csv(path, usecols = ['Survived'])['Survived'].values
    targets = pa.read_csv(path)['Survived'].values
    
    train = combined.iloc[:891]
    test = combined.iloc[891:] 
    return train, test, targets
 
train, test, targets = recover_train_test_target()
    
#Feature selection

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train,targets)

#look at the importance of each feature.   
features =pa.DataFrame()
features['feature']=train.columns
features['importance']=clf.feature_importances_
features.sort_values(by=['importance'],ascending=True, inplace=True)
features.set_index('feature',inplace=True)
features.plot(kind='barh',figsize=(25, 25))


# use model to select best fit features for train data set 

model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
print train_reduced.shape

# use model to select best fit features for test data set
test_reduced = model.transform(test)
print test_reduced.shape
 
#Let's try different base models

logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()
models = [logreg, logreg_cv, rf, gboost]

for model in models:
    print 'Cross-validation of : {0}'.format(model.__class__)
    score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')
    print 'CV score = {0}'.format(score)
    print '****'
