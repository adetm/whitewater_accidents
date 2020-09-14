import numpy as np
import pandas as pd
import math
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objects as go
import datetime
import plotly.tools as tls
import plotly.express as px
import os
import chart_studio

#read CSV
whitewater_accidents= pd.read_csv('/Users/AnnaD/Desktop/accident-2/accidents.csv')

print(whitewater_accidents.shape)
whitewater_accidents = whitewater_accidents.dropna(subset=['accidentdate'])
print(whitewater_accidents.shape)
#drop duplicate columns
whitewater_accidents = whitewater_accidents.T.drop_duplicates().T
print(whitewater_accidents.shape)

whitewater_accidents.info()

#change date to date time
whitewater_accidents['accidentdate'] =  pd.to_datetime(whitewater_accidents['accidentdate'])

whitewater_accidents.isna().sum()
#drop columns I don't need & columns with high amounts of nan
whitewater_accidents.drop(columns = ['contactname', 'contactphone','contactemail','othervictimnames','waterlevel','groupinfo','location'], inplace=True)

whitewater_accidents.isna().sum()

#filter USA Accidents
whitewater_accidents_us = whitewater_accidents[whitewater_accidents['countryabbr']=='US']
whitewater_accidents_us

#count of accidents by state
count_state = pd.DataFrame(whitewater_accidents_us['state'].value_counts()).reset_index()
count_state = count_state.rename(columns={"index": "state","state": "count"})
count_state.head()

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
fig.show()
#colorado has the highest number of whitewater Accidents
# Output the plot to the screen

#visualize years
count_date = pd.DataFrame(whitewater_accidents_us['accidentdate'].value_counts()).reset_index()
count_date = count_date.rename(columns={"index": "date","accidentdate": "count"})
count_date['date']=pd.to_datetime(count_date.date)
count_date = count_date[(count_date['date'] > '1980-01-01') & (count_date['date'] < '2020-09-14')]
count_date = count_date.sort_values(['date'])
count_date


#plot
fig = px.line(count_date, x='date', y="count")
fig.show()

### import datetime as dt
whitewater_accidents['accidentdate'] = pd.to_datetime(whitewater_accidents['accidentdate'])
whitewater_accidents[(whitewater_accidents['accidentdate'] > '1990-06-01') &
                     (whitewater_accidents['accidentdate'] < '1990-06-30')]
#there were just a lot of accidents in June 1990, no one single event with more than 1 victim

fig = px.histogram(whitewater_accidents, x="age")
#babies don't go on whitewater very often, it must be that 0 was imputed for unknown ages
fig.show()

##### visualize age of victims

fig = px.histogram(whitewater_accidents, x="age", range_x = (5,80), nbins = 20)
fig.show()
#25-29 most deaths
