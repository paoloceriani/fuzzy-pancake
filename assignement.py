# -*- coding: utf-8 -*-

# i have put all the datasets into the same folder

import numpy as np
import pandas as pd

x=np.array([1,1,1])
y=np.array([0, -1, 1,5])


def inner(x,y):
    return x.dot(y)

# or equally, but implemented even with normal list -> will be slower 
    
def inner2(x,y):
    sum=0
    if(len(x) != len(y)):
        raise Exception('the two vectors have different lenghts, i cannot execute the inner product')
    for i in range(len(x)):
        sum=sum+x[i]*y[i]
    return sum


def Mae(x,y):
    return 1/len(x)*(sum(abs(y-x)))

#pair distances 
import math
def pair1(X,y):
    sum=0
    l=[]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            sum=sum+(X[i][j]-y[j])**2
        l.append(math.sqrt(sum))
    return l 

def pair2(X, y):
    l=[]
    n=X.shape[0]
    for i in range(0, n): # x[i, :]
        l.append(np.linalg.norm(X[i]-y))
    return l

def pair3(X,y):
    return np.linalg.norm(X-y,1)


def lead(x,n):
    ## devo fare shifting delle componenti oltre la n 
    l=len(x)
    ## voglio l-n componenti all'inizio
    for i in range(l-n):
        x[i]=x[n+i]
    print(x)
    ## voglio nan da l-n a l
    for i in range(l-n,l):
        x[i]=np.nan
    return x

def lag(x,n):
    l=len(x)
    for i in range(0,l-n):
        x[l-1-i]=x[l-n-i-1]
        print(x)
    for i in range(n):
        x[i]=np.nan
    return x


## PANDAS EXERCISES 
name=["origin","year","month","day","hour","temp","dewp","humid","wind_dir","wind_speed","wind_gust","precip","pressure","visib","time_hour"]

data = pd.read_csv('nycflights13_weather.csv', sep=',', names=name, skiprows=43)
data['temp']=(data['temp']-32)*(5/9)
 


# dataset with origin=JFK
data=data[data.origin=="JFK"]

#first dumb way to interpolate
data[pd.isna(data.temp)]

data.temp[5596]=(data.temp[5595]+data.temp[5597])/2
# i could have used 
data.temp=data.temp.interpolate()


# now i want to create a new vector that contains inly the means 
daily_mean=[]

# creo una nuova variabile nel dataset con piccola modifica, tanto costa O(n) assumendo criterio di costo uniforme
data['new_data']=data['day']+35*data['month']
avg_temp=data.groupby('new_data', as_index=False)['temp'].mean()
avg_temp.plot(x='new_data', y='temp')

#or, if i want more precise label, i create a new date
 
avg_temp['date']=pd.to_datetime("2013" +"-" +((avg_temp.new_data)//35).map(str)+ "-"+ ((avg_temp.new_data)%35).map(str))
avg_temp.plot(x="date",y="temp")


#per una roba più pulita che si può fare? potrei creare un nuovo indice giorno, o sistemare la chiave

# choose days with greater mean temperature than in the preceding day
prec=[True]
for i in range(avg_temp.shape[0]-1):
    if avg_temp['temp'][i+1]>avg_temp['temp'][i]:
        prec.append(True)
    else:
        prec.append(False)

#alternativamente dovrebbe funzionare
l=[True]
for i in range(26129):
   l.append(avg_temp.temp[i+1]>avg_temp.temp[i])




#find the top five
#i could have ordered the entire array and take the first(last) five, but it will take O(n log(n)) at the best
# better to use this, O(n)

ind=np.argpartition(avg_temp, -5)[-5:]


# EXERCISE NUMBER 2

clear 

data_2=pd.read_csv("nycflights13_flights.csv", sep=',', skiprows=54)
name=list(data_2)

if data_2.loc[:,'year':'day'].shape[1]:
    res1=data_2.loc[:, 'year':'day']
else:
    res1=data_2.loc[:, 'day':'year']
    
i=name.index('year')
j=name.index('day')
if i<j:
    name2=name[0:i]+name[(j+1):]
else:
    name2=name[0:j]+name[(i+1):]
data_2[name2]


#EXERCISE NUMBER 3
A=pd.read_csv("some_birth_dates1.csv")
B=pd.read_csv("some_birth_dates2.csv")
C=pd.read_csv("some_birth_dates3.csv")
frames=[A,B,C]
aux=pd.merge(A,B, on='Name', how='outer')
aux2=pd.merge(aux,C, on='Name', how='outer')
#to be sure it's correct
A['Name'].isin(B['Name']).value_counts()
pd.merge(A,B, on='Name', how='inner')
pd.merge(A,C, on='Name', how='inner')

A[~B['Name'].isin(A['Name'])]

