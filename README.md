# Naive-Bayes-Algorithm-Data-Mining

DM 2

For this project I am Data Mining using the Naïve Bayes algorithm for two different datasets. This is a probabilistic classifier assuming independence between features, The algorthm will calculate the probability of a certain input belonging to a specific class based on its features and training data. Generally, the Naïve Bayes algorithm is pretty efficient and accurate. 

*Please reach out to me on LinkedIn if you would like the datasets sent to you*

**Problem 1**

For this project (DM2) I am Data Mining using the Naïve Bayes algorithm. The first problem we will be using the IRIS dataset introduced in the lecture. K=3 for each of the flowers. It has 150 data points across Sepal length and width, as well as Petal length and width. The three types of flowers are Setosa, Versicolor, Virginica.

*Code output from IanSpAss2P1.sln file - if aiming to duplicate, you will need to replace file path*

<img width="651" alt="2-1" src="https://github.com/ianspetnagel/Naive-Bayes-Algorithm-Data-Mining/assets/62821052/02d20e9c-908b-49f9-bbde-4109a2024cdc">

**Problem 2**

The second problem is also implementing the Naïve Bayes Algorithm, however I am instead using a dataset for wheat seeds. There are 210 data points across 7 features. Our K value will remain 3.
Initially, I experienced issues in compliing because the dataset did not include category names. I have altered the dataset but still received an error.

*Error 1*

<img width="918" alt="err1" src="https://github.com/ianspetnagel/Naive-Bayes-Algorithm-Data-Mining/assets/62821052/e4cc4ecd-a69d-4192-92d1-7493a693c6eb">

This was happening becasue it was still failing to recognize the 'species' column in the dataset. I chnaged the code to index the fifth column of the dataframe and filter the rows by value in the 'species' column.

*Previous code:*
<pre>
#---assemble the data by categories i.e., classes
  df1 = df2[df2 == 1]
  print(df1)
  df2 = dfrandom[dfrandom['species'] == 2]
  print(df2)
  df3 = dfrandom[dfrandom['species'] == 3]
  print(df3)   
</pre>

*New code:*
<pre>
# ---assemble the data by categories i.e., classes
  df1 = dfrandom[dfrandom.iloc[:, 4] == 1]
  print(df1)
  df2 = dfrandom[dfrandom.iloc[:, 4] == 2]
  print(df2)
  df3 = dfrandom[dfrandom.iloc[:, 4] == 3]
  print(df3)
</pre>

I ended up being able to classfiy the seed, but was experiencing a 0.0 classification accuracy. This was likely caused due to errors in the dataset, model complexity, or feature engineering. 

<img width="659" alt="outwzero" src="https://github.com/ianspetnagel/Naive-Bayes-Algorithm-Data-Mining/assets/62821052/aabd43f7-be2f-41db-8502-79f3a0fbd517">













