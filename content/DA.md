---
title: "Data Analytics"
date: 2020-05-10T23:12:23-04:00
draft: true
---

Given a dataset of census income data, cleaned data, built models, performed k fold validations, and compared accuracies to predict whether people earn over 80k or under 80k. 

---
# CS381 Data Analytics Final Project

### Due on 5/13/2020 23:59 pm


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
```


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
```


```python
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
```

### Dataset is based on an census income data
https://archive.ics.uci.edu/ml/datasets/census+income


Data Set Information:

Extraction was done by Barry Becker from the 1994 Census database. 


Attribute Information:

Listing of attributes:

* The last column >50K, <=50K is the target variable indicating whether the people earn less than or larger than 50K per year

* age: continuous.
* workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* fnlwgt: continuous.
* education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* education-num: continuous.
* marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
* relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
* race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
* sex: Female, Male.
* capital-gain: continuous.
* capital-loss: continuous.
* hours-per-week: continuous.
* native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.



### However, I have modified the dataset and so you will need to load the dataset by reading a csv file I provided. In particular, I changed 50K to 80K just to reflect the inflation 


```python
df = pd.read_csv("adult_income_modified.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=80K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=80K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=80K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=80K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=80K</td>
    </tr>
  </tbody>
</table>
</div>



### Your task is to build a model that can predict whether a people will earn <= 80K or > 80K

* Use any one of the models (Logistic, SVM, Naive Bayes, Decision Tree and Random Forecast) that we have covered in class.
* Do not use any models that we have not covered in class.
*
* The best performance model will have an extra 5 points, but the whole project will still be capped at 20 points for the whole final project



### Make sure your work include the following steps

* EDA (chekcing missing values, removing outliers)
* performed basic exploration of relationship, with plots and graphs
* separated data set into training and testing
* setup dummy variables to take care categorical variables
* normalize numerical features if needed
* tried at least two models and checked their model performance
* performed cross-validations


First change the target variable salary to 0 and 1


```python
df['salary'] = df['salary'].apply(lambda x: 0 if x == '<=80K' else 1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.countplot(df['salary'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f16908e12b0>




![png](/../img/output_14_1.png)


### Good Luck !!!

Show all your work below

# EDA


```python
df.shape
```




    (32571, 15)



Check for unknown values


```python
for i,j in zip(df.columns,(df.values.astype(str) == '?').sum(axis = 0)):
    if j > 0:
        print(str(i) + ': ' + str(j) + ' records')
```

    workclass: 1836 records
    occupation: 1843 records
    native-country: 583 records


Data with unknown values seems relatively few, so I'll just remove them


```python
df.replace('?', np.nan, inplace=True)
```


```python
df.dropna(inplace=True)
```


```python
df.shape
```




    (30172, 15)



Now, lets see if we have any outliers


```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>-0.076033</td>
      <td>0.042992</td>
      <td>0.079221</td>
      <td>0.058706</td>
      <td>0.097945</td>
      <td>0.238747</td>
    </tr>
    <tr>
      <th>fnlwgt</th>
      <td>-0.076033</td>
      <td>1.000000</td>
      <td>-0.045184</td>
      <td>0.000375</td>
      <td>-0.008402</td>
      <td>-0.018298</td>
      <td>-0.009112</td>
    </tr>
    <tr>
      <th>education-num</th>
      <td>0.042992</td>
      <td>-0.045184</td>
      <td>1.000000</td>
      <td>0.124404</td>
      <td>0.079385</td>
      <td>0.150591</td>
      <td>0.335303</td>
    </tr>
    <tr>
      <th>capital-gain</th>
      <td>0.079221</td>
      <td>0.000375</td>
      <td>0.124404</td>
      <td>1.000000</td>
      <td>-0.032255</td>
      <td>0.079535</td>
      <td>0.221190</td>
    </tr>
    <tr>
      <th>capital-loss</th>
      <td>0.058706</td>
      <td>-0.008402</td>
      <td>0.079385</td>
      <td>-0.032255</td>
      <td>1.000000</td>
      <td>0.057228</td>
      <td>0.149698</td>
    </tr>
    <tr>
      <th>hours-per-week</th>
      <td>0.097945</td>
      <td>-0.018298</td>
      <td>0.150591</td>
      <td>0.079535</td>
      <td>0.057228</td>
      <td>1.000000</td>
      <td>0.226693</td>
    </tr>
    <tr>
      <th>salary</th>
      <td>0.238747</td>
      <td>-0.009112</td>
      <td>0.335303</td>
      <td>0.221190</td>
      <td>0.149698</td>
      <td>0.226693</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.boxplot(column="capital-gain")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f168d4c89d0>




![png](/../img/output_27_1.png)


100,000 seems like an abitrary number, perhaps it was used as a placeholder. Regardless its an obvious outlier, lets remove it


```python
df = df[df['capital-gain'] < 60000]
df.shape
```




    (30024, 15)




```python
df.boxplot(column="capital-loss")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f168e16c1f0>




![png](/../img/output_30_1.png)


There are a few outliers but it doesn't seem too bad so we'll leave them


```python
df.boxplot(column="hours-per-week")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f168e147700>




![png](/../img/output_32_1.png)


There are only 168 hours in a week, so clearly these should be removed


```python
df = df[df['hours-per-week'] < 168]
df.shape
```




    (30021, 15)



Lets plot some relationships; first the text based ones.


```python
sns.heatmap(pd.crosstab(df['occupation'], df['salary']).div(pd.crosstab(df['occupation'], df['salary']).apply(sum,1),0))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f168e016700>




![png](/../img/output_36_1.png)



```python
sns.heatmap(pd.crosstab(df['workclass'], df['salary']).div(pd.crosstab(df['workclass'], df['salary']).apply(sum,1),0))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f168dfc8400>




![png](/../img/output_37_1.png)



```python
sns.heatmap(pd.crosstab(df['marital-status'], df['salary']).div(pd.crosstab(df['marital-status'], df['salary']).apply(sum,1),0))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f168df4ad30>




![png](/../img/output_38_1.png)



```python
sns.heatmap(pd.crosstab(df['race'], df['salary']).div(pd.crosstab(df['race'], df['salary']).apply(sum,1),0))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f168deb2760>




![png](/../img/output_39_1.png)



```python
sns.heatmap(pd.crosstab(df['sex'], df['salary']).div(pd.crosstab(df['sex'], df['salary']).apply(sum,1),0))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f168ddd49d0>




![png](/../img/output_40_1.png)


Conclusions:
1. Managerial and specialties tend to make more money
2. Self employed workers tend to make more money
3. Married people tend to make more money
4. Men slightly tend to make more money than females.

Now, for the numeric correlations


```python
sns.heatmap(df[list(df.dtypes[df.dtypes != 'object'].index)].corr(),annot = True,square = True);
```


![png](/../img/output_43_0.png)


Conclusions:
1. Education, age, capital gain, hours per week all highly contribute to a person's likelihood of making more money
2. Highly educated people tend to work more hours, highly educated people tend to have a higher capital gain

# Data Transformation

- Education and education num are the same thing, so lets drop one.
- Relationship is basically sex combined with marital status, so lets drop it
- Fnlwgt was used to represent the weight of each type of person. Since we're trying to predict whether a single person's salary, its irrelevant to us so lets drop that too.


```python
df.drop(inplace=True, columns=['education', 'relationship', 'fnlwgt'])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Normalize Numerical Features


```python
from sklearn.preprocessing import MinMaxScaler
num_col = df.dtypes[df.dtypes != 'object'].index
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(df, columns=[num_col])
df[num_col] = scaler.fit_transform(df[num_col])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.156</td>
      <td>State-gov</td>
      <td>0.800000</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>White</td>
      <td>Male</td>
      <td>0.052626</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.200</td>
      <td>Self-emp-not-inc</td>
      <td>0.800000</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.122449</td>
      <td>United-States</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.152</td>
      <td>Private</td>
      <td>0.533333</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.212</td>
      <td>Private</td>
      <td>0.400000</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.112</td>
      <td>Private</td>
      <td>0.800000</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Black</td>
      <td>Female</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>Cuba</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



One hot encoding


```python
df = pd.get_dummies(df, prefix=['workclass', 'marital-status', 'occupation', 'race', 'sex', 'native-country'], drop_first=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>salary</th>
      <th>workclass_Local-gov</th>
      <th>workclass_Private</th>
      <th>workclass_Self-emp-inc</th>
      <th>workclass_Self-emp-not-inc</th>
      <th>...</th>
      <th>native-country_Portugal</th>
      <th>native-country_Puerto-Rico</th>
      <th>native-country_Scotland</th>
      <th>native-country_South</th>
      <th>native-country_Taiwan</th>
      <th>native-country_Thailand</th>
      <th>native-country_Trinadad&amp;Tobago</th>
      <th>native-country_United-States</th>
      <th>native-country_Vietnam</th>
      <th>native-country_Yugoslavia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.156</td>
      <td>0.800000</td>
      <td>0.052626</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.200</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.122449</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.152</td>
      <td>0.533333</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.212</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.112</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 76 columns</p>
</div>



Split our data into training and test so we can evaluate our model after training


```python
y = df['salary']
df.drop('salary', axis=1, inplace=True)
X = df
```

# Training

## SVM


```python
from sklearn import metrics
from statistics import mean 
import random
svm_res =  {'accuracy':[],'f1':[],'precision':[],'recall':[]}
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random.randint(1,1000))
    model = svm.SVC(kernel="linear")
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    svm_res['accuracy'].append(metrics.accuracy_score(y_test,y_pred))
    svm_res['f1'].append(metrics.f1_score(y_test,y_pred))
    svm_res['precision'].append(metrics.precision_score(y_test,y_pred))
    svm_res['recall'].append(metrics.recall_score(y_test,y_pred))
    if i==0:
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
             0.0       0.87      0.93      0.90      4541
             1.0       0.72      0.58      0.64      1464
    
        accuracy                           0.84      6005
       macro avg       0.80      0.75      0.77      6005
    weighted avg       0.84      0.84      0.84      6005
    
    [[4217  324]
     [ 617  847]]


## Logistic Regression


```python
logreg_res =  {'accuracy':[],'f1':[],'precision':[],'recall':[]}
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random.randint(1,1000))
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    logreg_res['accuracy'].append(metrics.accuracy_score(y_test,y_pred))
    logreg_res['f1'].append(metrics.f1_score(y_test,y_pred))
    logreg_res['precision'].append(metrics.precision_score(y_test,y_pred))
    logreg_res['recall'].append(metrics.recall_score(y_test,y_pred))
    if i==0:
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
             0.0       0.87      0.93      0.90      4496
             1.0       0.74      0.57      0.64      1509
    
        accuracy                           0.84      6005
       macro avg       0.80      0.75      0.77      6005
    weighted avg       0.83      0.84      0.83      6005
    
    [[4194  302]
     [ 652  857]]


## Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB

bayes_res =  {'accuracy':[],'f1':[],'precision':[],'recall':[]}
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random.randint(1,1000))
    model = GaussianNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    bayes_res['accuracy'].append(metrics.accuracy_score(y_test,y_pred))
    bayes_res['f1'].append(metrics.f1_score(y_test,y_pred))
    bayes_res['precision'].append(metrics.precision_score(y_test,y_pred))
    bayes_res['recall'].append(metrics.recall_score(y_test,y_pred))
    if i==0:
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
             0.0       0.95      0.21      0.35      4588
             1.0       0.27      0.97      0.43      1417
    
        accuracy                           0.39      6005
       macro avg       0.61      0.59      0.39      6005
    weighted avg       0.79      0.39      0.37      6005
    
    [[ 975 3613]
     [  47 1370]]


## Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier

dtree_res =  {'accuracy':[],'f1':[],'precision':[],'recall':[]}
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random.randint(1,1000))
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    dtree_res['accuracy'].append(metrics.accuracy_score(y_test,y_pred))
    dtree_res['f1'].append(metrics.f1_score(y_test,y_pred))
    dtree_res['precision'].append(metrics.precision_score(y_test,y_pred))
    dtree_res['recall'].append(metrics.recall_score(y_test,y_pred))
    if i==0:
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
             0.0       0.85      0.95      0.90      4489
             1.0       0.77      0.51      0.61      1516
    
        accuracy                           0.84      6005
       macro avg       0.81      0.73      0.75      6005
    weighted avg       0.83      0.84      0.83      6005
    
    [[4265  224]
     [ 750  766]]


## Random Forest


```python
rforest_res =  {'accuracy':[],'f1':[],'precision':[],'recall':[]}
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random.randint(1,1000))
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    rforest_res['accuracy'].append(metrics.accuracy_score(y_test,y_pred))
    rforest_res['f1'].append(metrics.f1_score(y_test,y_pred))
    rforest_res['precision'].append(metrics.precision_score(y_test,y_pred))
    rforest_res['recall'].append(metrics.recall_score(y_test,y_pred))
    if i==0:
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
             0.0       0.88      0.91      0.90      4552
             1.0       0.70      0.62      0.65      1453
    
        accuracy                           0.84      6005
       macro avg       0.79      0.77      0.78      6005
    weighted avg       0.84      0.84      0.84      6005
    
    [[4159  393]
     [ 556  897]]


Calculate average scores for the K fold validation


```python
results = {'Support Vector Machine':svm_res,'Logistic Regression':logreg_res,'Naive Bayes':bayes_res,'Decision Tree':dtree_res,'Random Forest':rforest_res}
for algo in results:
    for score in results[algo]:
        results[algo][score] = mean(results[algo][score])
```


```python
pd.DataFrame(results).transpose().sort_values(by=['accuracy'], ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>f1</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Support Vector Machine</th>
      <td>0.846461</td>
      <td>0.653227</td>
      <td>0.727248</td>
      <td>0.592986</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.845845</td>
      <td>0.651503</td>
      <td>0.730976</td>
      <td>0.587738</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.843764</td>
      <td>0.658334</td>
      <td>0.709411</td>
      <td>0.614295</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.838052</td>
      <td>0.605000</td>
      <td>0.761951</td>
      <td>0.501822</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.387194</td>
      <td>0.435770</td>
      <td>0.281097</td>
      <td>0.969358</td>
    </tr>
  </tbody>
</table>
</div>



Almost all of the models are quite close, however naive bayes doesn't seem like a good fit for this data.

The support vector machine model seems to give the best results 

-----------------------------------------------------------

Final project for CS381 Data Analytics