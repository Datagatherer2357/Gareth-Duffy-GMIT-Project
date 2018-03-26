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
