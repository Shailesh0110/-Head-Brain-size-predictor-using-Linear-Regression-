import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('MarvellousHeadBrain.csv')
#print(df)
#print(df.columns)
['Gender', 'Age Range', 'Head Size(cm^3)', 'Brain Weight(grams)']
print('shape of data',df.shape)

X=df['Head Size(cm^3)'].values
y=df['Brain Weight(grams)'].values
#print(X)

#Least Square Method
mean_x=np.mean(X)
mean_y=np.mean(y)

#print(mean_y)

n=(len(X))

numrator=0
denamintor=0

#NISHA Equation y=mx +c
for i in range(n):
     numrator +=(X[i]-mean_x)*(y[i]-mean_y)
     denamintor +=(X[i]-mean_x)**2
m=numrator/denamintor
c=mean_y-(m*mean_x)

print('slope of line',m)
print('intercept',c)

max_x=np.max(X)+100
min_x=np.min(X)+100
#Display the plotting above points
x=np.linspace(min_x,max_x,n)

y_=m*x+c

plt.plot(x,y_,color='red',label='Regression line')
plt.scatter(X,y,label='Scatter line')
plt.legend()
plt.show()

#find goodness of fit line R2 square

ss_t=0
ss_r=0

for i in range(n):
    y_pred= c+m*X[i]
    ss_t+=(y[i]-mean_y)**2
    ss_r+=(y[i]-y_pred)**2

r2=1-(ss_r/ss_t)

print('R2 score',r2)


