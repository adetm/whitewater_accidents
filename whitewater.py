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
#drop columns I don't need & columns with high amounts of nan
whitewater_accidents.drop(columns = ['contactname', 'contactphone','contactemail','othervictimnames','waterlevel','groupinfo','location'], inplace=True)


# In[6]:


whitewater_accidents.isna().sum()


# In[7]:


whitewater_accidents.info()


# In[8]:


whitewater_accidents


# In[9]:


#filter USA Accidents
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
#8 is the maximum number of victims


# In[16]:


trainingData = whitewater_accidents.iloc[:, :].values
dataset = whitewater_accidents.iloc[:, :].values


# In[ ]:





# In[17]:


whitewater_accidents['age'] = whitewater_accidents['age'].fillna(method='ffill')
whitewater_accidents.info()


# In[18]:



fig = px.histogram(whitewater_accidents, x="age")
fig.show()
#impute Nan for 0
#babies don't go on whitewater very often, it must be that 0 was imputed for unknown ages
cols=['age']
whitewater_accidents[cols] = whitewater_accidents[cols].replace({0:np.nan, 0:np.nan})
whitewater_accidents['age'] = whitewater_accidents['age'].fillna(method='ffill')



# In[19]:


whitewater_accidents


# In[ ]:





# In[ ]:





# In[20]:


whitewater_accidents.isna().sum()


# In[21]:


whitewater_accidents.dropna(subset=['age'],inplace = True)
whitewater_accidents.info()


# In[22]:


##### visualize age of victims

fig = px.histogram(whitewater_accidents, x="age", range_x = (0,80), nbins = 20)

fig.show()
#25-29 most deaths


# In[23]:


whitewater_accidents['type'].value_counts()


# In[24]:


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


# In[25]:


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


# In[26]:


whitewater_accidents['death'].value_counts()


# In[27]:


count_deaths = len(whitewater_accidents[whitewater_accidents['death']==0])
count_no_death = len(whitewater_accidents[whitewater_accidents['death']==1])
pct_of_no_death = count_no_death/(count_no_death+count_deaths)
print("percentage of no deaths is", pct_of_no_death*100)
pct_of_death = count_deaths/(count_no_death+count_deaths)
print("percentage of deaths", pct_of_death*100)


# In[28]:


whitewater_accidents['age'].isna().sum()


# In[ ]:





# In[29]:


whitewater_accidents['month'] = pd.DatetimeIndex(whitewater_accidents['accidentdate']).month


# In[30]:


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




'''
#difficulty
whitewater_accidents['numeric_difficulty'] = whitewater_accidents['numeric_difficulty'].astype(int)
whitewater_accidents['encoded_difficulty'] = label_encoder.fit_transform(whitewater_accidents['numeric_difficulty'])

'''


# In[31]:


whitewater_accidents.info()


# In[32]:


whitewater_numeric_columns = whitewater_accidents[['age','privcomm','cause','month','encoded_experience','death']]


# In[33]:


whitewater_numeric_columns


# In[34]:


pd.crosstab(whitewater_numeric_columns.month,whitewater_numeric_columns.death).plot(kind='bar')
plt.title('Frequency of Deaths by Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Death')


# In[35]:


whitewater_numeric_columns = whitewater_numeric_columns.rename(columns ={"death": "y"})


# In[51]:


X = whitewater_numeric_columns.drop('y', axis = 1)
y = whitewater_numeric_columns['y']


# In[52]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns


# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[54]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# In[56]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set:{:.2f}'.format(logreg.score(X_test, y_test)))


# In[57]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# The result is telling us that we have 89+463 correct predictions and 23+181 incorrect predictions

# In[58]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
