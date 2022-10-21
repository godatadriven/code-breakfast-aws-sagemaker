from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import plot_importance, plot_tree
from german_holidays import get_german_holiday_calendar
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
sns.set_style('whitegrid')
plt.rcParams['figure.figsize']=(20,10) # for graphs styling
plt.style.use('tableau-colorblind10') # for graph stying




# Importing German holidays
cal_cls = get_german_holiday_calendar('NW')
cal = cal_cls()
holidays = [
    h.date() for h in pd.to_datetime(cal.holidays(start='2012', end='2020'))
]


main_df = pd.read_excel('SalesData.xlsx', parse_dates=True)
main_df.head()


main_df['From'] = pd.to_datetime(main_df['From']).dt.date
main_df = pd.DataFrame(main_df.groupby(main_df['From'])['Sold Units'].sum())
test_df['Date'] = pd.to_datetime(test_df['Date']).dt.date
main_df.head()


training_df = main_df.groupby(['From'])['Sold Units'].sum().reset_index()
training_df['Date'] = pd.to_datetime(training_df['From']).dt.date

training_df = training_df[training_df['Date'].isin(holidays) == False]
training_df['From'] = pd.to_datetime(training_df['From'])


training_df.set_index('From', inplace=True)
training_df = training_df['2016-06-01':]
training_df.head()