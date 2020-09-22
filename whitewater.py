#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objects as go
import datetime
import plotly.tools as tls
import plotly.express as px
import os
import chart_studio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


#read CSV
whitewater_accidents= pd.read_csv('/Users/AnnaD/Desktop/accident-2/accidents.csv')


# I dropped accidents that didn't have a date (there weren't that many of them so I decided it was ok to drop them. As the date was key for analysis I decided that dropping the date, as opposed to imputing was sensible
# Also, I dropped duplicate columns, as these were present and unecessary

# In[3]:


print(whitewater_accidents.shape)
whitewater_accidents = whitewater_accidents.dropna(subset=['accidentdate'])
print(whitewater_accidents.shape)
#drop duplicate columns
whitewater_accidents = whitewater_accidents.T.drop_duplicates().T
print(whitewater_accidents.shape)


# In[4]:


#change date to date time
whitewater_accidents['accidentdate'] =  pd.to_datetime(whitewater_accidents['accidentdate'])


# In[5]:


whitewater_accidents.isna().sum()
#there are a lot of missing values


# In[6]:


whitewater_accidents.isna().sum()
#drop columns I don't need & columns with high amounts of nan
whitewater_accidents.drop(columns = ['contactname', 'contactphone','contactemail','othervictimnames','waterlevel','groupinfo','location'], inplace=True)


# In[7]:


whitewater_accidents.info()


# In[8]:


whitewater_accidents


# In[9]:


#filter USA Accidents to map accidents
whitewater_accidents_us = whitewater_accidents[whitewater_accidents['countryabbr']=='US']
whitewater_accidents_us


# In[10]:


count_state = pd.DataFrame(whitewater_accidents_us['state'].value_counts()).reset_index()
count_state = count_state.rename(columns={"index": "state","state": "count"})
count_state.head()


# In[11]:


import plotly.express as px  # Be sure to import express

fig = px.choropleth(count_state,  # Input Pandas DataFrame
                    locations="state",  # DataFrame column with locations
                    color="count",  # DataFrame column with color values
                    color_continuous_scale='darkmint',
                    locationmode = 'USA-states') # Set to plot as US States
fig.update_layout(
    title_text = 'Whitewater Accidents by State', # Create a Title
    geo_scope='usa',  # Plot only the USA instead of globe
)
fig.show()  # Output the plot to the screen
#colorado had the highest number of accidents


# In[12]:


count_date = pd.DataFrame(whitewater_accidents_us['accidentdate'].value_counts()).reset_index()
count_date = count_date.rename(columns={"index": "date","accidentdate": "count"})
count_date['date']=pd.to_datetime(count_date.date)
count_date = count_date[(count_date['date'] > '1980-01-01') & (count_date['date'] < '2020-09-14')]
count_date = count_date.sort_values(['date'])
count_date


# In[13]:



#plot
fig = px.line(count_date, x='date', y="count")
fig.show()
#what happened in June 1989/1990?


# In[14]:


### import datetime as dt
whitewater_accidents['accidentdate'] = pd.to_datetime(whitewater_accidents['accidentdate'])
whitewater_accidents[(whitewater_accidents['accidentdate'] > '1990-06-01') &
                     (whitewater_accidents['accidentdate'] < '1990-06-30')]
#there were just a lot of accidents in June 1990, no one single event with more than 1 victim


# In[15]:


whitewater_accidents.sort_values(by=['numvictims'], ascending = False)
#8 is the maximum number of victims in one accident


# In[16]:


whitewater_accidents['age'] = whitewater_accidents['age'].fillna(method='ffill')
#I wanted to make sure that there were no nans for age so I used the ffill method
whitewater_accidents.info()


# In[17]:


fig = px.histogram(whitewater_accidents, x="age")
fig.show()




# In[18]:


#impute Nan for 0
#babies don't go on whitewater very often, it must be that 0 was imputed for unknown ages
cols=['age']
whitewater_accidents[cols] = whitewater_accidents[cols].replace({0:np.nan, 0:np.nan})
whitewater_accidents['age'] = whitewater_accidents['age'].fillna(method='ffill')


# In[19]:


whitewater_accidents.isna().sum()
#make sure most nans were removed.


# In[20]:


whitewater_accidents.dropna(subset=['age'],inplace = True)
#one row was removed, lacked age even after imputing
whitewater_accidents.info()


# In[21]:


##### visualize age of victims

fig = px.histogram(whitewater_accidents, x="age", range_x = (0,80), nbins = 20)

fig.show()
#25-29 most deaths
#relatively normally distributed age of victims


# In[22]:


whitewater_accidents['type'].value_counts()


# In[23]:


#Wanted to change fatal and near miss to binary death or non death
# Create a list to store the data
death = []

# For each row in the column,
for row in whitewater_accidents['type']:
    # if equals F (Fatal),
    if row == 'F':
        # Append
        death.append(1)
    # else, M (Near Miss),
    elif row == 'M':
        # Append
        death.append(0)
    # else I (Injury) ,
    elif row == 'I':
        # Append a letter grade
        death.append(0)
    else:
        death.append(0)


# Create a column from the list
whitewater_accidents['death'] = death
# View the new dataframe
whitewater_accidents


# In[24]:


# Create a list to store the data
numeric_difficulty = []
#encoed difficulty from 1 - 11

# For each row in the column,
for row in whitewater_accidents['difficulty']:
    # if equals I
    if row == 'I':
        # Append
        numeric_difficulty.append(1)
    # else, II
    elif row == 'II':
        # Append
        numeric_difficulty.append(2)
     # else, II+
    elif row == 'II+':
        # Append
        numeric_difficulty.append(3)
    # else, III
    elif row == 'III':
        # Append
        numeric_difficulty.append(4)
    # else III
    elif row == 'III+':
        # Append a letter grade
        numeric_difficulty.append(5)
     # else IV-,
    elif row == 'IV-':
        # Append a letter grade
        numeric_difficulty.append(6)
     # else IV
    elif row == 'IV':
        # Append a letter grade
        numeric_difficulty.append(7)
     # else IV+
    elif row == 'IV+':
        # Append a letter grade
        numeric_difficulty.append(8)
     # else V
    elif row == 'V':
        # Append a letter grade
        numeric_difficulty.append(9)
      # else V+
    elif row == 'V+':
        # Append a letter grade
        numeric_difficulty.append(10)
      # else VI
    elif row == 'VI':
        # Append a letter grade
        numeric_difficulty.append(11)
    else:
        numeric_difficulty.append('NA')

# Create a column from the list
whitewater_accidents['numeric_difficulty'] = numeric_difficulty
# View the new dataframe
whitewater_accidents


# In[25]:


whitewater_accidents['death'].value_counts()


# In[26]:


#check for class imbalance
count_deaths = len(whitewater_accidents[whitewater_accidents['death']==0])
count_no_death = len(whitewater_accidents[whitewater_accidents['death']==1])
pct_of_no_death = count_no_death/(count_no_death+count_deaths)
print("percentage of no deaths is", pct_of_no_death*100)
pct_of_death = count_deaths/(count_no_death+count_deaths)
print("percentage of deaths", pct_of_death*100)
#Classes arent balanced but they aren't terrible


# In[27]:


#created a column of month
whitewater_accidents['month'] = pd.DatetimeIndex(whitewater_accidents['accidentdate']).month


# In[28]:


# Import label encoder
from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
# Encode labels
#experience
whitewater_accidents['experience'] = whitewater_accidents['experience'].astype(str)
whitewater_accidents['encoded_experience'] = label_encoder.fit_transform(whitewater_accidents['experience'])

#cause
whitewater_accidents['cause'] = whitewater_accidents['cause'].astype(str)
whitewater_accidents['cause'] = label_encoder.fit_transform(whitewater_accidents['cause'])
#privcomm
whitewater_accidents['privcomm'] = whitewater_accidents['privcomm'].astype(str)
whitewater_accidents['privcomm'] = label_encoder.fit_transform(whitewater_accidents['privcomm'])

#state
whitewater_accidents['state'] = whitewater_accidents['state'].astype(str)
whitewater_accidents['state'] = label_encoder.fit_transform(whitewater_accidents['state'])

#age
whitewater_accidents['age'] = whitewater_accidents['age'].astype(int)
whitewater_accidents['age'] = label_encoder.fit_transform(whitewater_accidents['age'])

#age
whitewater_accidents['death'] = whitewater_accidents['death'].astype(int)





# In[29]:


whitewater_accidents.info()


# In[30]:


#selected numeric columns to use as predictors
whitewater_numeric_columns = whitewater_accidents[['age','privcomm','cause','month','encoded_experience','death']]


# In[31]:


whitewater_numeric_columns


# In[32]:


pd.crosstab(whitewater_numeric_columns.month,whitewater_numeric_columns.death).plot(kind='bar')
plt.title('Frequency of Deaths by Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Death')


# In[33]:


whitewater_numeric_columns = whitewater_numeric_columns.rename(columns ={"death": "y"})


# The idea was to try to see if the following columns (age','privcomm','cause','month','encoded_experience')  were a good predictor of fatalities. It seems like they are not a terrible predictor of fatalities. Thus, if we know the age of the participant, whether it was a private or commercial trip, the month and the experience of the participant we might be able to predict whether the accident is a fatal accident or not.

# In[34]:


#set X and y for analysis
X = whitewater_numeric_columns.drop('y', axis = 1)
y = whitewater_numeric_columns['y']


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[37]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
#ensure p values are low


# In[38]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set:{:.2f}'.format(logreg.score(X_test, y_test)))


# In[39]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# The result is telling us that we have 89+463 correct predictions and 23+181 incorrect predictions

# In[40]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Decision Tree Accuracy
# I looked at decision trees briefly to see if the accuracy was different. The test accuracy wasn't very different

# In[41]:


from sklearn.tree import DecisionTreeClassifier
# Instantiate & fit the DT
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, y_train)



# Evaluate its classification accuracy (Just on the training set for now)
print(f"DT training set accuracy: {DT_model.score(X_train, y_train)}")
print(f"DT test set accuracy: {DT_model.score(X_test, y_test)}")


# In[42]:


# Create a scaled version of X data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[43]:


# KNN - unscaled data
from sklearn.neighbors import KNeighborsClassifier

train_accs = []
test_accs = []
k_values = list(range(1,30))

# Loop over different number of neighbors
for k in k_values:

    # Instantiate & fit
    my_knn = KNeighborsClassifier(n_neighbors = k)
    my_knn.fit(X_train, y_train)

    # Evaluate on train & test data
    train_accs.append( my_knn.score(X_train, y_train) )
    test_accs.append( my_knn.score(X_test, y_test) )



# KNN - scaled data

train_accs_s = []
test_accs_s = []
k_values = list(range(1,30))

# Loop over different number of neighbors
for k in k_values:

    # Instantiate & fit
    my_knn = KNeighborsClassifier(n_neighbors = k)
    my_knn.fit(X_train_scaled, y_train)

    # Evaluate on train & test data
    train_accs_s.append( my_knn.score(X_train_scaled, y_train) )
    test_accs_s.append( my_knn.score(X_test_scaled, y_test) )



# Plot the results
plt.subplot(211) # plot unscaled classifier results
plt.plot(k_values, train_accs, label='train')
plt.plot(k_values, test_accs, label='test')
plt.legend()
plt.xlabel('number of neighbors')
plt.ylabel('accuracy')
plt.title('KNN Accuracy (UNscaled data)')

plt.subplot(212) # plot scaled classifier results
plt.plot(k_values, train_accs_s, label='train')
plt.plot(k_values, test_accs_s, label='test')
plt.legend()
plt.xlabel('number of neighbors')
plt.ylabel('accuracy')
plt.title('KNN Accuracy (SCALED data)')

plt.tight_layout()
plt.show()
