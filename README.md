## **Gareth Duffy (G00364693) GMIT**

* Project 2018-Programming & Scripting
* Start date: 22-3-2018 End date 29-4-2018

<img src="https://image.ibb.co/gw4Gen/Index_GMIT.png" alt="Index GMIT" border="0" />

**PROJECT OUTLINE & OBJECTIVES**

This project concerns the timeless and ever relevant Fisher Iris dataset. 
The project requirements entail researching the dataset and writing documentation and Python code based on that research. 

The project attempts to break down the outlined requirements below into smaller tasks which are easier to solve and plug them together following to their indiviudal completion:

1. Research background information about the data set and write a summary about it.

2. Keep a list of references you used in completing the project.

3. Download the data set and write some Python code to investigate it.

4. Summarise the data set by, for example, calculating the maximum, minimum and
   mean of each column of the data set. A Python script will quickly do this for you.

5. Write a summary of your investigations.

6. Include supporting tables and graphics as you deem necessary.

<img src="https://image.ibb.co/bUBF5S/Index.png" alt="Index" border="0" />

**INTRODUCTION**


The Iris dataset is a multivariate dataset introduced by the British statistician and biologist Ronald Fisher in his classic 1936 paper, “The Use of Multiple Measurements in Taxonomic Problems” as an example of linear discriminant analysis and can be found on the UCI Machine Learning Repository [1], [2]. 
The data were originally collected and published by the statistically-minded botanist Edgar S. Anderson [3]. 

Multivariate (Data analysis) refers to any statistical technique used to analyze data which arises from more than one variable [4].

Linear discriminant analysis (LDA) is a classification method originally developed by Fisher. It is simple, mathematically robust and typically produces models whose accuracy is as good as more complex methods. 
LDA is based upon the concept of searching for a linear combination of variables (predictors) that best separates two classes [5]. 
If you have more than two classes then LDA is the preferred linear classification technique. 
Conversely for example, logistic regression is a classification algorithm traditionally limited to only two-class classification problems [6].

<img src="https://image.ibb.co/bFXDQS/Index0.png" alt="Index0" border="0" />

Based on Fisher's linear discriminant model, this data set became a typical test case for many statistical classification techniques in machine learning such as testing out machine learning algorithms and visualisations (e.g. scatterplots), as well as techniques such as  “support vector machines”, i.e. supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 

Machine learning (ML) is often, incorrectly, interchanged with artificial intelligence (AI), but ML is actually a sub field/type of AI. ML is also often referred to as predictive analytics, or predictive modelling. Coined by American computer scientist Arthur Samuel in 1959, the term ‘machine learning’ is defined as a “computer’s ability to learn without being explicitly programmed”. 
At its most basic, ML uses programmed algorithms that receive and analyse input data to predict output values within an acceptable range. As new data is fed to these algorithms, they learn and optimise their operations to improve performance, developing ‘intelligence’ over time [7].

The majority of ML uses supervised learning. Supervised learning requires that the algorithm’s possible outputs are already known and that the data used to train the algorithm is already labeled with correct answers (You will see an example of ML on the Iris data in section below). However, when data are not labelled, supervised learning is not possible, and an unsupervised learning approach is needed i.e. a type of ML algorithm used to draw inferences from datasets consisting of input data without labeled responses [6].

The most common unsupervised learning method is cluster analysis, which endeavours to find natural clustering of the data to groups, and then map new data to these formed groups [8]. With the Iris dataset however, the use of cluster analysis is not typical, since the dataset only contains two clusters with obvious separation. One of the clusters contains Iris setosa which is linearly separable from the other two. The other cluster contains both Iris virginica and Iris versicolor and is not separable without the species information Fisher used [9].

The dataset is an excellent example of a traditional resource that has become a staple of the computing world, especially for testing purposes. New types of sorting models and taxonomy algorithms often use the Iris flower dataset as an input, to examine how various technologies sort and handle data sets. For example, programmers might download the Iris flower dataset for the purposes of testing a decision tree, or a piece of ML software. For this reason, the Iris dataset is built into some coding libraries, in order to make this process easier (e.g. Python’s "ScikitLearn" module comes preloaded with it) [10].

The Iris dataset is probably the best known dataset to be found in the pattern recognition literature. The dataset is small but relevent, simple but challenging, and the examples (cases) are of real data, useful and of sound analytical quality. 


**METHOD**

***Software & Dependencies***

Software used for this project: Python 3.6, Anaconda Navigator, PyPI, Microsoft Visual Studio Code, Windows Edge, Microsoft Office.

This project requires Python 3.6 and the following Python libraries installed: Pandas, SciPy, ScikitLearn ,Numpy, Matplotlib, and Seaborn


***Qualities and attributes of the Iris dataset***

The Iris dataset contains 150 examples (rows), and 5 variables (columns) named; sepal length, sepal width, petal length, petal width, and species. 

<img src="https://image.ibb.co/kecPX7/IndexA.png" alt="IndexA" border="0" />

There are 3 species of iris flower (Setosa, Virginica, Versicolour), with 50 examples of each type. The number of data points for each class is equal, thus it is a balanced dataset. 

<img src="https://image.ibb.co/jdf05S/IndexB.png" alt="IndexB" border="0" />

Each row of the table represents one Iris flower, including its species and dimensions of its botanical anatomy (sepal length, sepal width, petal length, petal width). 

Each flower measurement is measured in centimetres and is of float data type. The species variables which are of string type. 

One flower species, the Iris Setosa, is “linearly separable” from the other two, but the other two are not linearly separable from each other. This refers to the fact that classes of patterns can be separated with a single decision surface, which means we can draw a line on the graph plane between Iris Setosa samples and samples corresponding to the other two species. You will see evidence of this in the figures to follow [11].


**PROCEDURE**


* #1 Python version 3.6 was downloaded via Anaconda Navigator 3 to Windows 10 OS (https://www.anaconda.com/).

* #2 Microsoft Visual Studio Code was dowloaded (https://code.visualstudio.com/).

* #3 Microsoft Visual Studio Code was configurated with GitHub (https://github.com/).

* #4 The Iris dataset was imported to Python as a CSV file (see Index[1] of project.py script).

  (NOTE: All Indices e.g. "Index[1]" are reference points in the Python file "project.py" which is stored in this repository)

* #5 In Python, the versions of the necessary Python libraries were checked and imported (see Index[2] of project.py script). 
     
     <img src="https://image.ibb.co/hgxpqS/Index2.png" alt="Index2" border="0" />
     
     (Index[2])

* #6 Next I used the "shape" method to reveal how many examples (rows) and how many attributes (columns) the Iris dataset contains.
   
     I also felt it was a good idea to eyeball the dataset using the "head" method to see the first 30 rows of the dataset.
     
     <img src="https://image.ibb.co/f1vW4n/Index3.png" alt="Index3" border="0" />
     
     (Index[3])
   
* #7 Next I used the "groupby" method in Python to print out the class distribution i.e. the number of instances (rows) that belong to      each species. 

   <img src="https://image.ibb.co/kX1bN7/Index4.png" alt="Index4" border="0" />
   
   (Index[4])
 
* #8 Next I went on to evaluate my descriptive summary statistics, and then conduct inferential analyses of the dataset (See results in      the next section).
   
   
**RESULTS**

***Descriptive summary statistics***

Descriptive statistical analysis was used to establish means, standard deviations, ranges, skewness/kurtosis and some other important measurements pertaining to iris flower anatomy. Tables (Pandas DataFrames) and figures i.e. A barplot and histograms were used for graphical representation of descriptive features of the dataset attributes.

Firstly, I established a summary for each Iris flower attribute by using the "describe" method from the Pandas library. This function returns a nice statistical summary including the count, mean, min and max values as well as some upper and lower percentiles. 

<img src="https://image.ibb.co/hioLFS/Index5.png" alt="Index5" border="0" />

(Index[5])

Next I programmed Python to generate a barplot of the anatomical features of the Iris species. This plot nicely shows how the three species of iris differ on basis of the four features. 

<img src="https://image.ibb.co/jK8YqS/Index5b.png" alt="Index5b" border="0" />

(Index[5b])

*(Discuss/interpret these summary stats)

Following the descriptive summary statistics, I went a little further to analze the shape of the spread of the Iris data. I coded Python to establish the skewness and kurtosis of each variable in the dataset (please see Index[6] URL image of output below)

The "Skew" of data refers to a distribution that is assumed Gaussian (normal or bell curve) that is shifted or pulled in one direction or another, typically to the left or right end of the spread of data quantities. Many machine learning algorithms assume a Gaussian distribution. Knowing that an attribute has a skew may allow you to perform data preparation to correct the skew (e.g. omit high or low scoring outliers) and later improve the accuracy of your models [6].

The skewness result shows either a positive (right) or negative (left) skew. Values closer to zero show less skew.

Kurtosis on the other hand, is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. Contrary to a skewed distribution, kurtosis is evidenced by seeing a very tall or very short distribution line on the spread of data.

<img src="https://image.ibb.co/jqM4QS/Index6.png" alt="Index6" border="0" />

(Index[6])

Next I endeavoured to demonstrate the distribution curves pertaining to the data of the Iris variables in order to further evaluate and highlight the spread and shape of the data measurments. I achieved this by plotting a histogram for each of the 4 float variables in the dataset. The histograms also contain distribution curves to emphasize the spread of Iris flower data. 
The Iris dataset has 4 numeric variables and I wanted to visualize their distributions *together* so I split the output windows into several parts. 

<img src="https://image.ibb.co/jpvqh7/Index7.png" alt="Index7" border="0" />

(Index[7])

These steps allowed me to esatablish a good descriptive picture of the distribution patterns of the Iris data measurements by combining both our skewness and kurtosis values with the histograms. 

*(Discuss/interpret the skewness and kurtosis of the printout data and graphs)


***Inferential statistics and figures***

To begin some inferential work on the Iris dataset I coded Python to run two scatterplots of the Iris data with the Seaborn library. 
One plot was tailored to illustrate any correlations between sepal length and sepal width (See Index[8] output below) and the second for petal length and petal width [12] (See Index[8] output below).

<img src="https://image.ibb.co/h3Buc7/Index8.png" alt="Index8" border="0" />

(Index[8])

From the regressional sepal scatterplot above we can easily distinguish the Iris setosa data points, but Iris versicolor and Iris verginica cannot be easily so distinguished based on their sepal width and sepal length. The sepal length and sepal width are somewhat correlated but still, not greatly so. We can aslo see that the setosa, is completely separated since they have small sepal length and small sepal width compared to the other species.

The real issue is that the virgincia and versicolor species are mixed apart. Therefore we will flip to the other side of the Iris scatterplot coin which demonstrates a very positive linear correlation between petal length and petal width across all 3 species. 
We can also see that the iris data is neatly partitioned among the 3 species and forms a very nice strong correlation line (See Index[9] output below).

<img src="https://image.ibb.co/mWmzAS/Index9.png" alt="Index9" border="0" />

(Index[9])

I went a little further by building on the correlational features of the Iris variables by illustrating their anatomical relationships. Here I used the Seaborn library to program Python to generate a "heatmap". 
This heatmap would output a matrix of all correlations between the 4 botanical parts of the Iris flowers (See Index[10] output below). 

<img src="https://image.ibb.co/krMGqS/Index10.png" alt="Index10" border="0" />

(Index[10])

From the matrix figure above, we can see it's clear that sepal length and sepal width show weak correlations, while petal width and petal length show very strong correlations. Indeed, this analysis bolsters our observations from the previous scatterplots and suggests that species of iris flower can be more easily identified using petals compared to sepals [13]. 

Because the Iris dataset is of the multivariate type, I felt it appropriate to paint a more elaborate graphical picture which would show multiple clusters of Iris data measurements.
For this task, "Andrews curves" allowed me to plot multivariate data as a large number of curves that are created using the attributes of samples as coefficients.
Essentially, Andrews curves work by mapping each observation, i.e. each Iris example onto a function. It has been shown the Andrews curves are able to preserve means, distance and variances, which means that Andrews curves that are represented by functions close together suggest that the corresponding data points will also be close together.

By coloring these curves differently for each class (or in this case species) it is possible to visualize data clustering in the Iris dataset. Curves belonging to samples of the same class will usually be closer together and form larger structures (See Index[11] output below) [14].

<img src="https://image.ibb.co/ceiCjn/Index11.png" alt="Index11" border="0" />

(Index[11])

*(Discuss/interpret the curves esp the setosa)

***Creating and evaluating algorithms for Iris data***

Next I chose to create a couple of models of the iris data and estimate their accuracy on unseen data.
This process would involve, separating out a validation dataset, setting up the test harness cross validation, building 2 contrasting models to predict Iris species from flower anatomical measurements, and finally selecting the best model.

For a solid estimate of the accuracy of the best model, I held back some data that the algorithms would not get to see and thus use this data to get an independent idea of how accurate the best model would actually be. 
To do this, I partitioned the Iris dataset into two, 80% of which would be used to train both models and 20% that would be held back as a validation dataset [6]. 

A nice cimbination of simple linear and nonlinear algorithms were used; linear discriminant analysis (LDA) and Gaussian Naïve Bayes (NB). These steps were taken to get an idea of the accuracy of the model on our validation dataset. 

So, first I split-out validation dataset. Here we create a set of X and Y arrays of training data in the form of "X train" and "Y train" for training/preparing our two models, and also "X validation" and "Y validation" sets that we can use later.
We also set a "validation size" of 20 which splits the dataset into 80% for our training purposes and 20% for our validation.

We will then use 10-fold cross validation to estimate accuracy. This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits (See Index[12] in project.py file). 

We will use the metric of ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next [6].

It is difficult to know which algorithms would be good on this problem or what configurations to use. We get an idea from the previous plots that some of the Iris classes are partially linearly separable in some dimensions, so we are expecting generally good results. 

Next I evaluated two different algorithms: Linear Discriminant Analysis (LDA) and Gaussian Naive Bayes (NB). T felt this was an appropriate blend of simple linear (LDA) and nonlinear (NB) algorithms.
It was important to reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using precisely the same data splits. 
This ensures the results are directly comparable. We can now build the models (See Index[12] in project.py file) [6].

I programmed Python to evaluate each model in turn by creating a for loop that would begin the 10-fold cross validation process to train and test the Iris data. Following this, it was simply a case of comparing the models to each other and selecting the most accurate (See Index[12] output below).

<img src="https://image.ibb.co/d4QSjn/Index12.png" alt="Index12" border="0" />

(Index[12])

I also generated an algorithm comparison plot of these evaluation results and compared the spread and the mean accuracy of each model (See Index[13] output below).

<img src="https://image.ibb.co/hZirAS/Index13.png" alt="Index13" border="0" />

(Index[13])

From the output I discovered that both models produced the ***same*** accuracy for predicting Iris species (97.5%). 
However, for the final step I choose the LDA model over Naive Bayes to get an idea of the accuracy of the model on my validation set. Indeed, LDA was the method Fisher himself used on his original analyses. This gave me an independent final check on the accuracy of the model [6]. 

It is valuable to keep a validation set in case you made an error during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result. 
Next I ran the model directly on the validation set and printed the results summary as a final accuracy score, a confusion matrix and a classification report (See Index [14] below): 

<img src="https://image.ibb.co/eXQA4n/Index14.png" alt="Index14" border="0" />

(Index[14])

I had now made predictions on the validation dataset and found that the accuracy wass 0.966 or 97%. 
The confusion matrix provided an indication of the three errors made, and the classification report afforded a breakdown of each species by precision, recall, f1-score and support showing excellent results. 
It could be therefore be safely inferred that the LDA model for predicting Iris species based on measurements of anatomy was highly predicitive and reliable.

***Supervised Machine learning on the Iris data*** 

For the final part of the project I used ScikitLearn to generate a decision tree classifier algorithm. This program is a basic example of supervised learning. 

The algorithm works by instructing Python to "learn" all of the exisiting data in the Iris dataset. Essentially, Python takes all of the existing Iris measurements, conducts regressional analyses, and then "matches" its calculations to each of the corresponding flower species. 

Following this, Python will ask the user to input 4 "new" Iris measurements (sepal length, sepal width, petal length and petal width) and then decide what Iris species it "believes" the measurements belong to based on what it has just learned and from what new measurments we have given it. 

Python will output this predicted flower type and generate an accuracy estimate (percentage) and confidence intervals based on its prediction [15] (See Index [15] below and Index[15] in project.py file):

<img src="https://image.ibb.co/hiP1Un/Index15.png" alt="Index15" border="0" />

(Index[15])



**DISCUSSION:**

**REFERENCES:** 

[1]: Marshall, M (1988, July). Iris Data Set. Retrieved from: https://archive.ics.uci.edu/ml/datasets/iris.

[2]: (Author unknown)(2016). Machine Learning With Iris Dataset. Retrieved from: https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset/data.

[3]: Cox, N (2013, November). Iris Data Set. Retrieved from: https://stats.stackexchange.com/questions/74776/what-aspects-of-the-iris-data-set-make-it-so-successful-as-an-example-teaching. 

[4]: (Author unknown)(2018). Multivariate Data Analysis. Retrieved from: http://www.camo.com/multivariate_analysis.html.

[5]: Sayad, S (2018). Linear Discriminant Analysis. Retrieved from: http://chem-eng.utoronto.ca/~datamining/dmc/lda.html. 

[6]: Brownlee, J (2017). Linear Discriminant Analysis for Machine Learning. Retrieved from: https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/.

[7]: Wakefield, K (2018). A guide to machine learning algorithms and their applications. Retrived from: https://www.sas.com/en_ie/insights/articles/analytics/machine-learning-algorithms.html.

[8]: Kelleher, G (2018). The Iris Dataset. Retrieved from: https://gist.github.com/curran/a08a1080b88344b0c8a7.

[9]: (Wikipedia, multiple authors)(2018, February). The Iris Flower Data set. Retrieved from: https://en.wikipedia.org/wiki/Iris_flower_data_set.

[10]: (Author unknown)(2018). The Iris Flower Data Set. Retrieved from: https://www.techopedia.com/definition/32880/iris-flower-data-set.

[11]: Hammer, B (2016). Iris Dataset. Retrieved from: https://www.kaggle.com/uciml/iris/discussion/18365.
  
[12]: Farheen, S (2018). Iris Data Analysis. Retrieved from: https://www.kaggle.com/farheen28/iris-dataset-analysis-using-knn

[13]:  Awal, R (2017). Iris Species Data. Retrieved from: https://github.com/rabiulcste/Kaggle-Kernels-ML/blob/master/Iris%20Species%20Data/Data%20Visualization%20and%20Machine%20Learning%20using%20Iris%20Data.ipynb

[14]: (Muliple authors unknown)(2018). Pandas0.15.2 Documentaion. Retrievd from http://pandas.pydata.org/pandas-docs/version/0.15/visualization.html.

[15]: Gebbie, W (2017). Iris. Retrieved from: https://github.com/will-gebbie.

**APPENDICES(Tables & Figures):**
