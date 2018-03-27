# Gareth-Duffy-GMIT
# Project 2018-Programming & Scripting
# Start date: 22-3-2018 End date 29-4-2018

**PROJECT OUTLINE & OBJECTIVES:**

This project concerns the timeless and still very relevant Fisher Iris dataset. The project entails researching the dataset, and writing documentation and Python code based on that research. 

The project attempts to break down the outlined requirements below into smaller tasks which are easier to solve and plug them together following to their indiviudal completion:

1. Research background information about the data set and write a summary about it.

2. Keep a list of references you used in completing the project.

3. Download the data set and write some Python code to investigate it.

4. Summarise the data set by, for example, calculating the maximum, minimum and
   mean of each column of the data set. A Python script will quickly do this for you.

5. Write a summary of your investigations.

6. Include supporting tables and graphics as you deem necessary.


**INTRODUCTION:**

The Iris dataset is a multivariate dataset introduced by the British statistician and biologist Ronald Fisher in his classic 1936 paper, “The Use of Multiple Measurements in Taxonomic Problems” as an example of linear discriminant analysis and can be found on the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/iris), (https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset/data). 
The data were originally collected and published by the statistically-minded botanist Edgar S. Anderson (https://stats.stackexchange.com/questions/74776/what-aspects-of-the-iris-data-set-make-it-so-successful-as-an-example-teaching). 

Multivariate (Data analysis) refers to any statistical technique used to analyze data which arises from more than one variable (http://www.camo.com/multivariate_analysis.html).

Linear discriminant analysis (LDA) is a classification method originally developed by Fisher. It is simple, mathematically robust and typically produces models whose accuracy is as good as more complex methods. 
LDA is based upon the concept of searching for a linear combination of variables (predictors) that best separates two classes (http://chem-eng.utoronto.ca/~datamining/dmc/lda.htm). 
If you have more than two classes then LDA is the preferred linear classification technique. Conversely for example, logistic regression is a classification algorithm traditionally limited to only two-class classification problems (https://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/).

Based on Fisher's linear discriminant model, this data set became a typical test case for many statistical classification techniques in machine learning such as testing out machine learning algorithms and visualisations (e.g. scatterplots), as well as techniques such as  “support vector machines”, i.e. supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 

Machine learning (ML) is often, incorrectly, interchanged with artificial intelligence (AI), but ML is actually a sub field/type of AI. ML is also often referred to as predictive analytics, or predictive modelling. Coined by American computer scientist Arthur Samuel in 1959, the term ‘machine learning’ is defined as a “computer’s ability to learn without being explicitly programmed”. 
At its most basic, ML uses programmed algorithms that receive and analyse input data to predict output values within an acceptable range. As new data is fed to these algorithms, they learn and optimise their operations to improve performance, developing ‘intelligence’ over time. (https://www.sas.com/en_ie/insights/articles/analytics/machine-learning-algorithms.html).

The majority of ML uses supervised learning. Supervised learning requires that the algorithm’s possible outputs are already known and that the data used to train the algorithm is already labeled with correct answers. However, when data are not labelled, supervised learning is not possible, and an unsupervised learning approach is needed i.e. a type of ML algorithm used to draw inferences from datasets consisting of input data without labeled responses (https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/).

The most common unsupervised learning method is cluster analysis, which endeavours to find natural clustering of the data to groups, and then map new data to these formed groups (https://gist.github.com/curran/a08a1080b88344b0c8a7). With the Iris dataset however, the use of cluster analysis is not typical, since the dataset only contains two clusters with obvious separation. One of the clusters contains Iris setosa which is linearly separable from the other two. The other cluster contains both Iris virginica and Iris versicolor and is not separable without the species information Fisher used (https://en.wikipedia.org/wiki/Iris_flower_data_set).

The dataset is an excellent example of a traditional resource that has become a staple of the computing world, especially for testing purposes. New types of sorting models and taxonomy algorithms often use the Iris flower dataset as an input, to examine how various technologies sort and handle data sets. For example, programmers might download the Iris flower dataset for the purposes of testing a decision tree, or a piece of ML software. For this reason, the Iris dataset is built into some coding libraries, in order to make this process easier (e.g. Python’s "ScikitLearn" module comes preloaded with it) (https://www.techopedia.com/definition/32880/iris-flower-data-set).

The Iris dataset is probably the best known dataset to be found in the pattern recognition literature. The dataset is small but not trivial, simple but challenging, and the examples (cases) are real data, useful and of good analytical quality. 


**METHOD:**

**Software & Dependencies:**

Software used for this project: Python 3.6, Anaconda Navigator, PyPI, Microsoft Visual Studio Code, Windows Edge, Microsoft Office.

This project requires Python 3.6 and the following Python libraries installed: Pandas, SciPy, ScikitLearn ,Numpy, Matplotlib, and Seaborn


**Qualities and attributes of the Iris dataset:**

The Iris dataset contains 150 examples (rows), and 5 variables (columns) named; sepal length, sepal width, petal length, petal width, and species. 
There are 3 species of iris flower (Setosa, Virginica, Versicolour), with 50 examples of each type. The number of data points for each class is equal, thus it is a balanced dataset. 

Each row of the table represents one Iris flower, including its species and dimensions of its botanical anatomy (sepal length, sepal width, petal length, petal width). 

Each flower measurement is measured in centimetres and is of float data type. The species variables which are of string type. 

One flower species, the Iris Setosa, is “linearly separable” from the other two, but the other two are not linearly separable from each other. This refers to the fact that classes of patterns can be separated with a single decision surface, which means we can draw a line on the graph plane between Iris Setosa samples and samples corresponding to the other two species. We will see this in the figures to follow (https://www.kaggle.com/uciml/iris/discussion/18365).


**PROCEDURE:**

* #1 Python version 3.6 was downloaded via Anaconda Navigator 3 to Windows 10 OS (https://www.anaconda.com/).

#2 Microsoft Visual Studio Code was dowloaded (https://code.visualstudio.com/).

#3 Microsoft Visual Studio Code was configurated with GitHub (https://github.com/).

#4 The Iris dataset was imported to Python as a CSV file (see Index[1] of project.py script).

   (NOTE: All Indices e.g. "Index[1]" are reference points in the Python file "project.py" which is stored in this repository)

#5 In Python, the versions of the necessary Python libraries were checked and imported (see Index[2] of project.py script).
   
   The output (Index[2]) can be seen in this URL: https://image.ibb.co/hgxpqS/Index2.png  

#6 Next I used the "shape" method to reveal how many examples (rows) and how many attributes (columns) the Iris dataset contains.
   
   I also felt it was a good idea to eyeball the dataset using the head function to see the first 30 rows of the dataset.
   
   The output (Index[3]) can be seen in this URL: https://image.ibb.co/f1vW4n/Index3.png
   
#7 Next I used the "groupby" function in Python to print out the class distribution i.e. the number of instances (rows) that belong to      each species. The output (Index[4]) can be seen in this URL: https://image.ibb.co/kX1bN7/Index4.png
 
#8 Next I went on to evaluate my descriptive summary statistics, and then conduct inferential analyses of the dataset (See results in      the next section).
   
   
**RESULTS:**

**Descriptive summary statistics:**

Descriptive statistical analysis was used to establish means, standard deviations, ranges,  skewness/kurtosis and other important measurements pertaining to iris flower anatomy. Tables (Pandas DataFrames) and figures including histograms, scatterplots and more advanced plots e.g. an Andrew’s Curve were used for graphical representation of descriptive features of the dataset attributes.

Firstly, I established a summary for each Iris flower attribute. This included the count, mean, min and max values as well as some upper and lower percentiles. The output (Index[5]) can be seen in this URL: https://image.ibb.co/hioLFS/Index5.png

*(Discuss/interpret these summary stats)

Following the descriptive summary statistics, I went a little further to analze the shape of the spread of the Iris data. I coded Python to establish the skewness and kurtosis of each variable in the dataset (please see Index[6] URL image of output below)

The "Skew" of data refers to a distribution that is assumed Gaussian (normal or bell curve) that is shifted or pulled in one direction or another, typically to the left or right end of the spread of data quantities. Many machine learning algorithms assume a Gaussian distribution. Knowing that an attribute has a skew may allow you to perform data preparation to correct the skew (e.g. omit high or low scoring outliers) and later improve the accuracy of your models (https://machinelearningmastery.com/understand-machine-learning-data-descriptive-statistics-python/).

The skewness result shows either a positive (right) or negative (left) skew. Values closer to zero show less skew.

Kurtosis on the other hand is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. Contrary to a skewed distribution, kurtosis is evidenced by seeing a very tall or very short distribution line on the spread of data.

The output (Index[6]) can be seen in this URL: https://image.ibb.co/jqM4QS/Index6.png

Next I endeavoured to demonstrate the distribution curves pertaining to the data of the Iris variables in order to further evaluate and highlight the spread and shape of the data measurments. I achieved this by plotting a histogram for each of the 4 float variables in the dataset. The histograms also contain distribution curves to emphasize the spread of Iris flower data. 
The Iris dataset has 4 numeric variables and I wanted to visualize their distributions *together* so I split the output windows into several parts. 

The output (Index[7]) can be seen in this URL: https://image.ibb.co/jpvqh7/Index7.png

Now we have esatablished a good descriptive picture of the distribution patterns of the Iris data measurements by combining both our printed skewness and kurtosis values with the histograms. 

*(Discuss/interpret the skewness and kurtosis of the printout data and graphs)



**DISCUSSION:**

**REFERENCES:** 

**APPENDICES(Tables & Figures):**
