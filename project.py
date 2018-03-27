# Gareth Duffy 23-3-2018
# Python complete project script

# Index[1]

# Various methods for importing iris dataset:

#1-Direct method open iris CSV:
# Iris dataset realligned with justification of spaces and columns
with open("data/iris.csv") as f:
  for line in f:
    table = line.split(',')  # Splits whitespace
    print('{0[0]:12} {0[1]:12} {0[2]:12} {0[3]:12} {0[4]:12}'.format(table))

#2-Import Pandas method to use DataFrame on CSV (with row numbers):
import pandas  # load library
iris = pandas.read_csv("data/iris.csv") # the iris dataset is now a Pandas DataFrame
print(iris) # prints all 150 rows and 5 columns of iris dataset

#3-Import URL method for iris dataset
import pandas # load library
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pandas.read_csv(url, names=names)
print(dataset)

#4-Import SkyLearn method for iris dataset
from sklearn import datasets
iris = datasets.load_iris()
print(iris)

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
