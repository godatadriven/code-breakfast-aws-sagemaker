import pandas as pd

data = pd.read_csv("input/supermarket_sales.csv")

list_1 = list(data.columns)

list_cate = []
for i in list_1:
    if data[i].dtype=='object':
        list_cate.append(i)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for i in list_cate:
    data[i]=le.fit_transform(data[i])

y=data['Gender']
x=data.drop('Gender',axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
print('end')