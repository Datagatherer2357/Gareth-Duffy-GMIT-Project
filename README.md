## **Gareth-Duffy -  (G00364693) - GMIT - H.Dip Data Analytics**

<img src="https://image.ibb.co/gw4Gen/Index_GMIT.png" alt="Index GMIT" border="0" />

* Project 2018-Programming & Scripting
* Start date: 22-3-2018 End date 29-4-2018

----------------------------------------------------------------------------------------------------------------------------------------

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

----------------------------------------------------------------------------------------------------------------------------------------

**INTRODUCTION**

<img src="https://image.ibb.co/bUBF5S/Index.png" alt="Index" border="0" />

The Iris dataset is a multivariate dataset introduced by the British statistician and biologist Ronald Fisher in his classic 1936 paper, “The Use of Multiple Measurements in Taxonomic Problems” as an example of linear discriminant analysis. The dataset can be found on the UCI Machine Learning Repository [1], [2]. 
The data were originally collected and published by the statistically-minded botanist Edgar S. Anderson [3]. 

Multivariate (Data analysis) refers to any statistical technique used to analyze data which arises from more than one variable [4].

Linear discriminant analysis (LDA) is a classification method originally developed by Fisher. It is simple, mathematically robust and typically produces models whose accuracy is as good as more complex methods. 
LDA is based upon the concept of searching for a linear combination of variables (predictors) that best separates two classes [5]. 
Thus, if you have more than two classes then LDA is a widely preferred linear classification technique. 
Conversely logistic regression for example, is a classification algorithm traditionally limited to only two-class classification problems [6].

<img src="https://image.ibb.co/bFXDQS/Index0.png" alt="Index0" border="0" />

Based on Fisher's linear discriminant model, this dataset became a typical test case for many statistical classification techniques in machine learning such as testing out machine learning algorithms and visualisations (e.g. scatterplots), as well as techniques such as  “support vector machines”, which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 

Machine learning (ML) is often incorrectly synonymized with artificial intelligence (AI), but ML is actually a sub-field and type of AI. ML is also often referred to as predictive analytics, or predictive modelling. Coined by American computer scientist Arthur Samuel in 1959, the term ‘machine learning’ is defined as a “computer’s ability to learn without being explicitly programmed”. 
At its most basic, ML uses programmed algorithms that receive and analyse input data to predict output values within an acceptable range. As new data is fed to these algorithms, they learn and optimise their operations to improve performance, developing ‘intelligence’ over time [7]. I will demonstrate a nice example of "supervised" ML on the Iris data at the end of the project.

The majority of ML uses supervised learning. Supervised learning requires that the algorithm’s possible outputs are already known and that the data used to train the algorithm is already labeled with correct answers. However, when data are not labelled, supervised learning is not possible, and an unsupervised learning approach is needed i.e. a type of ML algorithm used to draw inferences from datasets consisting of input data without labeled responses [6].

The most common unsupervised learning method is cluster analysis, which endeavours to find natural clustering of the data to groups, and then map new data to these formed groups [8]. However, with the Iris dataset the use of cluster analysis is not typical, since the set only contains two clusters with obvious separation. One of the clusters contains Iris setosa which is *linearly separable* from the other two, and I will elaborate on this phenomenon below. The other cluster contains both Iris virginica and Iris versicolor and is not separable without the species information Fisher used [9].

The Iris dataset is an excellent example of a traditional resource that has become a staple of the computing world, especially for testing purposes. New types of sorting models and taxonomy algorithms often use the Iris dataset as an input, to examine how various technologies sort and handle data sets. For example, programmers might download the Iris flower dataset for the purposes of testing a decision tree, or a piece of ML software. For this reason, the Iris dataset is built into some coding libraries, in order to make this process easier (e.g. Python’s "ScikitLearn" module comes preloaded with it) [10].

The Iris dataset is probably the best known dataset to be found in the pattern recognition literature. The dataset is small but relevent, simple but challenging, and the examples (cases) are of real data, useful and of sound analytical quality. 

----------------------------------------------------------------------------------------------------------------------------------------

**METHOD**

***Software and Dependencies***

Software used for this project: Python 3.6, Anaconda Navigator, PyPI, Microsoft Visual Studio Code, Windows Edge, Microsoft Office.

This project requires Python 3.6 and the following Python libraries installed: Pandas, SciPy, ScikitLearn ,Numpy, Matplotlib, and Seaborn


***Qualities and attributes of the Iris dataset***

The Iris dataset contains 150 examples (rows), and 5 variables (columns) named; sepal length, sepal width, petal length, petal width, and species. 

<img src="https://image.ibb.co/kecPX7/IndexA.png" alt="IndexA" border="0" />

There are 3 species of iris flower (Setosa, Virginica, Versicolour), with 50 examples of each type. The number of data points for each class is equal, thus it is a balanced dataset. 

<img src="https://image.ibb.co/jdf05S/IndexB.png" alt="IndexB" border="0" />

Each row of the table represents one Iris flower, including its species and dimensions of its botanical anatomy (sepal length, sepal width, petal length, petal width). 

Each flower measurement is measured in centimetres and is of float data type. The species variables which are of string type. 

As previously mentioned, one flower species, the Iris Setosa, is “linearly separable” from the other two, but the other two are not linearly separable from each other. This refers to the fact that classes of patterns can be separated with a single decision surface, which means we can draw a line on the plot plane between Iris Setosa samples and samples corresponding to the other two species. You will see evidence of this in the figures to follow [11].

----------------------------------------------------------------------------------------------------------------------------------------

**PROCEDURE**

* #1 Python version 3.6 was downloaded via Anaconda Navigator 3 to Windows 10 OS (https://www.anaconda.com/).

* #2 Microsoft Visual Studio Code was downloaded (https://code.visualstudio.com/).

* #3 Microsoft Visual Studio Code was configurated with GitHub (https://github.com/).

* #4 The Iris dataset was imported to Python as a CSV file (see Index[1] of project.py script for various import methods).

  ***(PLEASE NOTE: All Indices e.g. "Index[1]" are reference points for the Python file "project.py" also stored in this repository.
     The "project.py" file contains ALL of the Python code used for this project).***

* #5 Using Python, each version of all the necessary Python libraries were checked and imported (see Index[2] of project.py script). 
     
     <img src="https://image.ibb.co/hgxpqS/Index2.png" alt="Index2" border="0" />
     
     (Index[2])

* #6 Next I used the "shape" method to reveal how many examples (rows) and how many attributes (columns) the Iris dataset contains.
     Shape is essentially a tuple that gives you an indication of the number of dimensions in an array.
   
     I also felt it was a good idea to eyeball the dataset using the "head" method to see the first 30 rows of the dataset. 
     Head returns the first n number of rows e.g. head(n = 30) returns the first 30 rows as seen below.
     
     <img src="https://image.ibb.co/f1vW4n/Index3.png" alt="Index3" border="0" />
     
     (Index[3])
   
* #7 Next I used the "groupby" method in Python to print out the class distribution i.e. the number of instances (rows) that belong to      each species. Groupby basically allows us to group data into buckets by categorical values, in this case species of Iris flower.

   <img src="https://image.ibb.co/kX1bN7/Index4.png" alt="Index4" border="0" />
   
   (Index[4])
 
* #8 Next I went on to run and evaluate some descriptive summary statistics of the Iris data. Following this I conducted some                inferential analyses on the dataset which I considered appropriate (See results in the next section).

----------------------------------------------------------------------------------------------------------------------------------------

**SUMMARY OF INVESTIGATIONS**

***Descriptive summary statistics***

Descriptive statistical analysis was used to establish the means, standard deviations, ranges, skewness/kurtosis and a few other important measurements pertaining to the Iris flower anatomy. Tables (Pandas DataFrames) and figures, i.e. A barplot and histograms were used for graphical representation of descriptive features of the dataset attributes.

Firstly, I established a summary for each Iris flower attribute by using the "describe" method from the Pandas library. This function returns a nice statistical summary of the data including the count, mean, min and max values as well as some upper and lower percentiles. 
For example, looking at the summary output we see that the sepal length mean is highest at 5.84cm while the petal width mean is notably low at 1.19cm.
Another interesting feature is that sepal length ranges from 4.3cm to 7.9cm, while petal width has a much lower range of 0.1cm to 2.5cm.
The median (middle) measurement of each column is represented by the 50th percentile row.

<img src="https://image.ibb.co/hioLFS/Index5.png" alt="Index5" border="0" />

(Index[5])

Next I used Matplotlib to generate a barplot of the anatomical features of the Iris species. This plot shows how the three species of Iris differ distinctly on the basis of their four anatomical features. 

<img src="https://image.ibb.co/jK8YqS/Index5b.png" alt="Index5b" border="0" />

(Index[5b])

Looking at the barplot above, we can see that the Iris Virginia species tends to have the longest sepal length, petal length and petal width, while the Setosa has the longest sepal width.
What also caught my eye about the Setosa is that they seem to exhibit significantly smaller petal widths and petal lengths compared to both other Iris species.
We can also see that all three species tend to have very similar sepal widths.

Following the descriptive summary statistics, I went a little further to analyze the shape of the spread (distribution of the Iris 
data).
I coded Python to establish the skewness and kurtosis of each variable in the dataset (please see Index[6] URL image of output below and Index[6] of project.py in this repository)

The "Skew" of data refers to a distribution that is assumed Gaussian (normal or bell curve) that is shifted or pulled in one direction or another, typically to the left or right end of the spread of data quantities. Many machine learning algorithms assume a Gaussian distribution. Knowing that an attribute has a skew may allow you to perform data preparation to correct the skew (e.g. omit high or low scoring outliers) which can later improve the accuracy of your models [6].

The skewness result will show either a positive (right) or negative (left) skew. Values closer to zero dipict less of a skew.

Kurtosis on the other hand, is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. Contrary to a skewed distribution, kurtosis is evidenced by seeing a very tall or very short distribution line on the spread of data.

<img src="https://image.ibb.co/jqM4QS/Index6.png" alt="Index6" border="0" />

(Index[6])

So, looking at the skewness and kurtosis output we can that sepal length and sepal width are slightly positively skewed, i.e.pulled to the right of the measurement spread.
Conversely, petal length and petal width are negatively skewed, thus also diverging from the normal distribution.
We can see that petal length revealed the highest kurtosis score of -1.4.

Next I endeavoured to illustrate the distributions of each Iris data measurement, in order to further highlight and interpret the collective shape of the data measurments. 
I achieved this by plotting a histogram for each variable in the dataset. Each histogram also displays a distinct distribution curve to emphasize the spread of data. 
The Iris dataset has four numeric (float) type variables, and because I wanted to visualize these distributions *together*, I split the output windows into individual parts for comparison. 

<img src="https://image.ibb.co/ncDYs7/Index7.png" alt="Index7" border="0" />

(Index[7])

Looking at the histogram plots we can see some distinguishing features.
Sepal length shows notable kurtosis, around the 4cm to 7cm range, and sepal width has a very high kurtosis distributed around the 2cm to 4cm range.
Petal length has an unusual distribution with two distinct kurtosis peaks. Looking at petal width, we can see it has a very similar distribution but one that is more tightly packed together. 

These analytical steps allowed me to esatablish a nice descriptive picture of the distribution patterns of the Iris measurements by combining the numeric output of skewness and kurtosis with histograms of the same Iris data. 


***Inferential statistics and figures***

To begin inferential work on the dataset I used the Seaborn library to create two scatterplots of the Iris data. 
The first plot was tailored to illustrate any correlations between sepal length and sepal width (See Index[8] output below) and the second for petal length and petal width [12] (See Index[8] output below).

<img src="https://image.ibb.co/h3Buc7/Index8.png" alt="Index8" border="0" />

(Index[8])

From the regressional sepal scatterplot above we can easily distinguish the Iris setosa data points, but Iris versicolor and Iris verginica cannot be as easily distinguished based on their sepal width and sepal length. 
Indeed, sepal length and sepal width are somewhat correlated but still not greatly so. 
We can also see that the setosa is completely separated since they have small sepal length and small sepal width compared to the other species.

The real issue seems to be that the virgincia and versicolor species are mixed apart. Therefore I flipped to the other side of the Iris scatterplot coin. This petal oriented scatterplot demonstrated a very positive linear correlation between petal length and petal width *across all 3 species*. 
The iris data is also neatly partitioned among the 3 species and forms a very nice strong correlation line (See Index[9] output below).

<img src="https://image.ibb.co/mWmzAS/Index9.png" alt="Index9" border="0" />

(Index[9])

To build on the correlational features demonstrated by the Iris scatterplots, I went a little further with the variables to illustrate their relationships in a different way. 
Here, I used the Seaborn library to program Python to generate a "heatmap". 
The heatmap produces an output matrix of all correlations between the 4 botanical parts of the flowers (See Index[10] output below). 

<img src="https://image.ibb.co/krMGqS/Index10.png" alt="Index10" border="0" />

(Index[10])

Looking at the heatmap, we can see a very strong positive correlation between sepal length and petal length. Interestingly, the strongest negative correlation can be seen between petal length and sepal width.
Sepal length and sepal width also show a rather weak negative correlation, while petal width and petal length reveal another strong positive correlation. 
Based on these relationships, it seems true that species with larger petal lengths tend to also have larger petal widths. 
This analysis corroborated my previous observations from the scatterplots and seemed to suggest that species of iris flower can be more easily identified using petals compared to sepals [13].  

Because the Iris dataset is of the multivariate type, I felt it appropriate to paint a more elaborate graphical picture which would show the multiple clusters of Iris data measurements.
For this task, I used Matplotlib and incorporated "Andrews curves" to plot the multivariate Iris data as a large number of curves. These curves are typically created by using the attributes of samples, in this case Iris measurements, as coefficients. 
A coefficient measures a certain property or characteristic of a data set, phenomenon, or process, given specified conditions. For example, the Pearson’s correlation coefficient(r) tells us the degree of correlation between two variables. 
Essentially, Andrews curves work by mapping each observation, i.e. each Iris measurement onto a *function*. 

It has been shown that these curves are able to preserve means, distance and variances, which means that Andrews curves that are represented by functions close together suggest that the corresponding data points will also be close together.
By coloring the curves differently for each class (species) it is possible to visualize data clustering in the Iris dataset. 
Curves belonging to samples of the same species will usually be closer together and form larger structures (See Index[11] output below) [14].

<img src="https://image.ibb.co/dvcUzn/Index11.png" alt="Index11" border="0" />

(Index[11])

Looking at the Andrews Curves plot we can immediately see its graphical allure. The curves clearly mirror similar trends illustrated by previous plots. 
For example, one of the most salient features of this plot is the fact that the Setosa cluster is notably separated from both the Virginica and Versicolour species. This plot is a fine example of how relatively simple data can be represented in such a way that it paints a surprisingly outspoken picture of the patterns in that data.  


***Evaluating algorithms for Iris data***

After endeavouring to tell an inferential story of the Iris data, I chose to create a couple of models of the data and estimate their accuracy on some unseen data.
This process involved, separating out a validation dataset, setting up a test harness cross validation, building 2 contrasting models to predict Iris species based on flower anatomical measurements, and finally selecting the best model.

For a solid estimate of the accuracy of the best model, some data had to be held back so that the algorithms would not get to see and use this data in order to get an independent idea of how accurate the best model would be. 
To do this, I partitioned the Iris dataset into two, 80% of which would be used to train both models and 20% that would be held back as a validation dataset [6]. 

An appropriate combination of simple linear and nonlinear algorithms were used; linear discriminant analysis (LDA) and Gaussian Naïve Bayes (NB). These steps were taken to get an idea of the accuracy of the model on our validation dataset. 

While linear discriminant analysis has already been briefly described above. A Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature, e.g. sepal width. Gaussian Naive bayes is a classification technique used where features typically follow a normal distribution [16].

First I split-out the validation dataset. I essentially created a set of X and Y arrays of training data in the form of "X train" and "Y train" for training/preparing the two models, and also "X validation" and "Y validation" sets that I could use later.
I set a "validation size" of 20 which split the dataset into 80% for training purposes and 20% for validation.

10-fold cross validation was then used to estimate accuracy. This split the dataset into 10 parts, i.e. train on 9 and test on 1 and repeat for all combinations of train-test splits (See Index[12] in project.py file). 

The measure of ‘accuracy‘ was then to evaluate both models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). The scoring variable would be used when each model is subsequently built and evaluated [6].

Research suggests that it is difficult to know which algorithms are good on this type of problem or what configurations to use. 
However, you can get an idea from the inferential plots that some of the Iris classes are at least partially linearly separable in some dimensions, and this is usually a sign that we can expect generally good results. 

Next the two algorithms (LDA & NB) were evaluated. Research suggested that my choice was a good blend of simple linear (LDA) and nonlinear (NB) algorithms.
It is important to reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using precisely the same data splits. This also ensures results will be directly comparable (See Index[12] in project.py file) [6].

Each model was evaluated in turn by introducing a for loop that would begin the 10-fold cross validation process to train and test the Iris data. Following this, it was a case of comparing the models to each other and selecting the most accurate (See Index[12] output below).

<img src="https://image.ibb.co/d4QSjn/Index12.png" alt="Index12" border="0" />

(Index[12])

I generated an algorithm comparison plot of the evaluation results. This compared the spread and the mean accuracy of each model (See Index[13] output below).

<img src="https://image.ibb.co/hZirAS/Index13.png" alt="Index13" border="0" />

(Index[13])

The output showed that both models had the same accuracy for predicting Iris species based on anatomical features (97.5%). 
Because Fisher himself used LDA for his original analyses, I decided to use it over Naive Bayes for the last step. 
This would give me an idea of how accurate the LDA model was on the validation set, i.e. it would afford a final independent check on the accuracy of the model [6]. 

Research asserts that it is valuable to keep a validation set in case an error is made during training, such as overfitting to the training set or a data leak. Both can result in an overly optimistic result. 
Next I ran the model directly on the validation set and printed the results summary as a final accuracy score, a confusion matrix and a classification report (See Index [14] below): 

<img src="https://image.ibb.co/eXQA4n/Index14.png" alt="Index14" border="0" />

(Index[14])

Looking at the output we can see that the accuracy of the predictions on the validation dataset was 0.966%. 
The confusion matrix provided an indication of errors made, and the classification report afforded a breakdown of each species by precision, recall, f1-score and support, in this case it showed excellent results. 
Based on this output it could be inferred that the LDA model for predicting Iris species based on measurements of anatomy was highly predicitive and reliable.


***Supervised Machine learning on Iris data*** 

To wind down the Iris project, I decided to use ScikitLearn to generate a decision tree classifier algorithm. This program is a basic example of supervised learning. 

The algorithm works by instructing Python to "learn" all of the exisiting data in the Iris dataset. Essentially, Python takes all of the existing Iris measurements, conducts regressional analyses, and then "matches" its calculations to each of the corresponding flower species. 

Python will then prompt the user to input four *new* (unlearned) Iris measurements (sepal length, sepal width, petal length and petal width) and subsequently decide what Iris species it "believes" the measurements belong to, based on what it has just learned and from the new measurments we have given it. 

Python will then output the "predicted" flower type and generate an accuracy estimate (percentage) and confidence intervals based on its prediction [15] (See Index [15] below and Index[15] in project.py file). Pretty cool.

<img src="https://image.ibb.co/hiP1Un/Index15.png" alt="Index15" border="0" />

(Index[15])

----------------------------------------------------------------------------------------------------------------------------------------

**DISCUSSION**

Before I began this project, I had little experience in Python programming, so it took me a bit of time to get my head around some concepts and processes. 
That said, I was still motivated. Combined with my enthusiasm for Python, I also had previous experience in data analtyics, in areas such as statistics, modelling, inferential analysis, SPSS and Rapidminer. My previous academic background helped a lot with managing, formatting and interpretting the project.

Looking back, I can take solace and some pride in the fact that I have learned a vast amount by studying and using Python, and I'm eager to keep learning.
I can now think about problems in a more logical and objective way when I'm using Python's tools and features.
After grasping core concepts such as loops, conditional statements, control flow and logic, I am able to breakdown problems into smaller segments in order to solve them systematically. 
In general, this experience has provided me with valuable tools and accompanying knowledge to add to my data science portfolio.

After conducting a relatively comprehensive analysis, evaluation and interpretation of the Iris dataset it is hoped that some may have gained an appreciation for the value and relevance of data analytics. 
At first glance, the Iris dataset appeared to be little more than a collection of trivial data. The truth however, is that this seemingly typical data, turned out to provide a rich, fascinating and fruitful piece of botanical information, one which allowed me to paint an elaborate analytical picture that reflects something relatively simple in the natural world. 

This is the beauty of data analytics. By using the tools it provides, we can take something seemingly frivolous, and transform it into a fascinating and valuable work of art and science. 
This exciting process lets us better appreciate the relationships and causalities that exist in the natural world, and affords an enormous, ever expanding and untapped repository that anyone can explore. 
We pull apart the initial irrelavent messiness, apply the right type of analytics, and end up with real, scientifically driven patterns and significant relationships to interpret.

Data is without a doubt the new oil, and we are fortunate enough to have an abundance of it at our disposal. 
Fishers dataset is a classic example of why the science is so transformative and relevant.  

----------------------------------------------------------------------------------------------------------------------------------------

**REFERENCES:** 

[1]: Marshall, M. (1988, July). *Iris Data Set*. Retrieved from: https://archive.ics.uci.edu/ml/datasets/iris.

[2]: (Author unknown), (2016). *Machine Learning With Iris Dataset*. Retrieved from: https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset/data.

[3]: Cox, N. (2013, November). *Iris Data Set*. Retrieved from: https://stats.stackexchange.com/questions/74776/what-aspects-of-the-iris-data-set-make-it-so-successful-as-an-example-teaching. 

[4]: (Author unknown), (2018). *Multivariate Data Analysis*. Retrieved from: http://www.camo.com/multivariate_analysis.html.

[5]: Sayad, S. (2018). *Linear Discriminant Analysis*. Retrieved from: http://chem-eng.utoronto.ca/~datamining/dmc/lda.html. 

[6]: Brownlee, J. (2017). *Linear Discriminant Analysis for Machine Learning*. Retrieved from: https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/.

[7]: Wakefield, K. (2018). *A guide to machine learning algorithms and their applications*. Retrived from: https://www.sas.com/en_ie/insights/articles/analytics/machine-learning-algorithms.html.

[8]: Kelleher, G. (2018). *The Iris Dataset*. Retrieved from: https://gist.github.com/curran/a08a1080b88344b0c8a7.

[9]: (Wikipedia, multiple authors)(2018, February). *The Iris Flower Data set*. Retrieved from: https://en.wikipedia.org/wiki/Iris_flower_data_set.

[10]: (Author unknown), (2018). *The Iris Flower Data Set*. Retrieved from: https://www.techopedia.com/definition/32880/iris-flower-data-set.

[11]: Hammer, B. (2016). *Iris Dataset*. Retrieved from: https://www.kaggle.com/uciml/iris/discussion/18365.
  
[12]: Farheen, S. (2018). *Iris Data Analysis*. Retrieved from: https://www.kaggle.com/farheen28/iris-dataset-analysis-using-knn

[13]:  Awal, R. (2017). *Iris Species Data*. Retrieved from: https://github.com/rabiulcste/Kaggle-Kernels-ML/blob/master/Iris%20Species%20Data/Data%20Visualization%20and%20Machine%20Learning%20using%20Iris%20Data.ipynb

[14]: (Authors unknown), (2018). *Pandas0.15.2 Documentaion*. Retrievd from http://pandas.pydata.org/pandas-docs/version/0.15/visualization.html.

[15]: Gebbie, W. (2017). *Iris*. Retrieved from: https://github.com/will-gebbie.

[16]: Sunil, R. (2017). *6 Easy Steps to Learn a Naive Bayes Algorithm*. Retrieved from:  https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/.

----------------------------------------------------------------------------------------------------------------------------------------

**APPENDICES**

***Miscellaneous Python code for Iris dataset***

(Please see Index[A] in project.py file for code for output below)


<img src="https://image.ibb.co/dfQ4X7/IndexA1.png" alt="IndexA1" border="0" />
Output of converting categorical Iris data, i.e. species into to numerical data.

