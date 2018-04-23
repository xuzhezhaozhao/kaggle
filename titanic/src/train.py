# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

# preview the data
print("\ntrain_df head")
train_df.head()
print("\ntrain_df tail")
train_df.tail()

print("\ntrain_df info")
train_df.info()
print('_'*40)
print("\ntest_df info")
test_df.info()

print("\ntrain_df describe")
print(train_df.describe())
print("\ntrain_df describe(include=['O'])")
print(train_df.describe(include=['O']))
print("\ntrain_df describe(percentiles=[.61, .62])")
print(train_df.describe(percentiles=[.61, .62]))

print("\npivoting features analysis...")
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
