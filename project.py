# Gareth Duffy, G00364693, 23-3-2018
# Python complete project script

# Index[1]
# Various methods for importing iris dataset:

#1-Direct method open Iris CSV:
# Iris dataset realligned with justification of spaces and columns
with open("data/iris.csv") as f:
  for line in f:
    table = line.split(',')  # Splits whitespace
    print('{0[0]:12} {0[1]:12} {0[2]:12} {0[3]:12} {0[4]:12}'.format(table))

#2-Import via Pandas method to use DataFrame on CSV (with row numbers):
import pandas  # load library
iris = pandas.read_csv("data/iris.csv") # the iris dataset is now a Pandas DataFrame
print(iris) # prints all 150 rows and 5 columns of iris dataset

#3-Import URL method for Iris dataset
import pandas # load library
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pandas.read_csv(url, names=names)
print(dataset)

#4-Import via SkyLearn method for Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
print(iris)

#5-Import via Seaborn method for Iris dataset
import seaborn as sns
iris = sns.load_dataset("iris")

# Index[2]
# Check the versions of Python libraries & import those libraries: 
# Check the versions code from: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
 
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
import seaborn 
# seaborn
print('seaborn: {}'.format(seaborn.__version__))

# Index[3]
# We can get a glimpse of how many examples (rows) and how many attributes (columns) the Iris dataset contains with the shape property:
# shape
print(dataset.shape)

# It is also a good idea to actually eyeball your data # using the head function we can see the first 30 rows of the Iris data:
# head
print(dataset.head(30))

# Index[4]
# the number of instances (rows) that belong to each species. We can view this as an absolute count.
# class distribution
print(dataset.groupby('species').size())

# Index [5]
# Statistical Summary
# We can take a look at a summary of each Iris flower attribute.
# This includes the count, mean, min and max values as well as some percentiles.
# descriptions
print(dataset.describe())
#Count
#Mean
#Standard Devaition
#Minimum Value
#25th Percentile
#50th Percentile (Median)
#75th Percentile
#Maximum Value

# Index [5b]
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris = sns.load_dataset("iris")

# Barplot of the anatomical features of the Iris species:
# This plot shows how the three species of iris differ on basis of the four features. 

y = iris.species
X = iris.drop('species',axis=1)
dataset = pd.melt(iris, "species", var_name="measurement") 
sns.factorplot(x="measurement", y="value", hue="species", data=dataset, size=7, kind="bar",palette="bright") 
plt.show() 

# Index[6]
# Skewness and Kurtosis measurements of Iris data

skew = dataset.skew() # Skew function imported from scipy library.
print("Skewness of Iris data") # Explanatory output string label
print(skew)
kurtosis = dataset.kurtosis() # Kurtosis function imported from scipy library.
print("Kurtosis of Iris data") # Explanatory output string label
print(kurtosis)

# Index [7]
# Histograms of Iris variables with distribution curves  
# https://python-graph-gallery.com/25-histogram-with-several-variables-seaborn/

# library and data
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('iris')
 
# plot
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.distplot( df["sepal_length"] , color="skyblue", ax=axes[0, 0]) # assigning variable, colour theme and axes to graph
sns.distplot( df["sepal_width"] , color="olive", ax=axes[0, 1])
sns.distplot( df["petal_length"] , color="gold", ax=axes[1, 0])
sns.distplot( df["petal_width"] , color="teal", ax=axes[1, 1])
plt.show()

# Index [8] & [9]
# Scatterplots of Iris variables
# https://www.kaggle.com/farheen28/iris-dataset-analysis-using-knn

import seaborn as sns
import matplotlib.pyplot as plt
sns.FacetGrid(dataset, hue="species", size=5).map(plt.scatter, "sepal_length", "sepal_width").add_legend()
plt.show()
sns.FacetGrid(dataset, hue="species", size=5).map(plt.scatter, "petal_length", "petal_width").add_legend()
plt.show()

# Index [10]
# Heatmap of Iris data 
# https://github.com/rabiulcste/Kaggle-Kernels-ML/blob/master/

plt.figure(figsize=(7,4)) 
#draws  heatmap with input as the correlation matrix calculted by(df.corr())
sns.heatmap(dataset.corr(),cbar = True, square = True, annot=True, fmt='.2f',annot_kws={'size': 15},cmap='Dark2') 
plt.show()

# Index [11]
# Adrews Curves Plot for multivariate data
# http://pandas.pydata.org/pandas-docs/version/0.15/visualization.html

from pandas import read_csv
from pandas.plotting import andrews_curves

dataset = pandas.read_csv(url, names=names)
plt.figure()
andrews_curves(dataset, 'species')
plt.show()

# Index [12]
# Evaluating algorithms on Iris data and predictions

# Split-out validation dataset

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7 # find out what this means
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric

seed = 7
scoring = 'accuracy'

# Spot checking algorithms

models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))

# Evaluating each model in turn

results = []
names = []
for name, model in models: # for loop to conduct an evaluation of each model
	kfold = model_selection.KFold(n_splits=10, random_state=seed) # assigning folds on dataset
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# Index [13]  
# Comparing algorithms

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Index [14]
# Making predictions on the validation dataset

lda = LinearDiscriminantAnalysis() 
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Index [15]
# Basic supervised Machine Learning
# https://github.com/will-gebbie
# import dependencies
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_score


# Call Iris dataset 
iris = datasets.load_iris()

data, target = iris.data, iris.target

# Choose the classification model
clf = tree.DecisionTreeClassifier() # selection of classifier type

# Train the model with the existing Iris data
clf = clf.fit(data, target) # clf.fit method uses existing Iris data to predict "target" species

# Test the classifier model

seplen = input("Please enter sepal length: ")
sepwid = input("Please enter sepal width: ")
petlen = input("Please enter petal length: ")
petwid = input("Please enter petal width: ")

attrArr = [float(seplen), float(sepwid), float(petlen), float(petwid)]
flower = [attrArr]

flowerType = " " 

if clf.predict(flower) == [0]: # if & elif statements to identify flower species
	flowerType = "Setosa"
elif clf.predict(flower) == [1]:
	flowerType = "Versicolor"
elif clf.predict(flower) == [2]:
	flowerType = "Virginica"

scores = cross_val_score(clf, data, target, cv=10)

result = "The flower is most likely an Iris " + flowerType
acc = "Accuracy: %.2f (+/- %.2f)" % (scores.mean()*100, scores.std()*200) # gives us an accuracy score with confidence intervals

print(result, end= ', ') # Prints predicted Iris species
print(acc) # Prints out accuracy estimate 

#--------------------------------------------------------------------------------------------------------------------------------------------

# Index[A] (Addional Python code for Iris dataset)

# Converting the categorical Iris data (species) into to numerical data

# Importing the libraries
import pandas as pd

# Importing the Iris dataset
iris = pd.read_csv('data/iris.csv')

print(iris.head(5)) # print out of Iris dataset head BEFORE for loop conversion begins

for i in range(0,len(iris)): # for loop conversion to convert strings to integers

    if iris['species'][i]=='setosa':
        iris['species'][i]=1
    elif iris['species'][i]=='versicolor':
        iris['species'][i]=2
    else:
        iris['species'][i]=3

print (iris.head(52)) # prints out inputted number of converted rows of Iris dataset

