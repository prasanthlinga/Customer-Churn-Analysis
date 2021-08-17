#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv(r"D:\Prasanth\customer attrition\WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()


# In[3]:


data


# In[4]:


data.tail()


# In[5]:


data.head(50)


# In[6]:


# data.head(50:100)   invalid syntax


# In[7]:


data.shape


# In[8]:


data.size


# In[9]:


data.dtypes


# In[10]:


data.columns


# In[11]:


data.columns[0:19]


# In[12]:


data.iloc()


# In[13]:


data.iloc[0]  # prints zeroth row


# In[14]:


data.iloc[0:3] # prints first 3 rows


# In[15]:


a=data.loc[3]
a


# In[16]:


b=data.loc[3]
b


# In[17]:


a==b


# In[18]:


#data.loc["gender"]  #error 
#for loc we need to sepcify names of both rows and columns
#data.loc[Male]


# In[19]:


data.loc[data.gender=="Male"]


# In[20]:


data.loc[data.gender=="Female"].shape


# In[21]:


data.loc[data.gender=="Female"].size


# In[22]:


data.iloc[[0,1]]


# In[23]:


data.iloc[[2,7],[1,2]]     # first list selects rows and second list selects columnsS


# In[24]:


data.describe()
 


# In[25]:


data.describe().astype("int")


# In[26]:


data.describe().astype("str")


# In[27]:


data.describe().astype("float")


# In[28]:


data.MonthlyCharges.describe()   # describe method is applied to single column too


# In[29]:


data.describe(include="all")   #includes all columns


# In[30]:


data.gender.value_counts().plot(kind="bar")


# In[31]:


data.gender.value_counts().plot(kind="barh")    # bar to barh    horizontal


# In[32]:


x=data["Churn"].value_counts()    # here is he data imbalanced or not?
x


# In[33]:


x*100/len(data)


# # data is highly imbalanced     

# In[34]:


data.info()  # prints a concise summary of dataframe


# In[35]:


data.isnull()  # blank values afre not considered as null values


# In[36]:


data.empty


# In[37]:


data.isnull().sum()


# # Data Cleaning
# 

# In[38]:


data2=data.copy()
data2


# In[39]:


# Total charges should be numeric amount
# we need to convertit into numeric data type
data2.TotalCharges=pd.to_numeric(data2.TotalCharges,errors="coerce")


# In[40]:


data2.isnull().sum()


# In[41]:


# There are 11 rows with total charges as blank
#lets find out what they are
data2.loc[data2.TotalCharges.isnull()==True]


# In[42]:


data2.TotalCharges.iloc[753]


# In[43]:


data2.isnull().sum()


# In[44]:


data2=data2.dropna(how="any") #the 11 rows with no data are removed,if how =all then the row will be dropped only if all values ae Nan


# In[45]:


data2.shape


# In[46]:


data2.size


# In[47]:



labels=["{}-{}" .format(i,i+11) for i in range(1,72,12)]
labels


# In[48]:



data2["tenure_group"]=pd.cut(data2.tenure,range(1,80,12),labels=labels,right=False)
data2


# In[49]:


# Drop unnnecessary columns
data2.drop(columns=["customerID","tenure"],axis=1,inplace=True)
data2


# # Univariate analysis

# In[50]:


import seaborn as sns
#for i,X in enumerate(data2.columns):
 #   plt.figure(i)
 #   sns.countplot(data=data2,hue="Churn",x=X)


# In[51]:


plt.figure(figsize=(25,6))
plt1=data.gender.value_counts().plot(kind='bar')  #returns count of unique values  in dataframe.CompamnyNmae
plt1.set(xlabel="gender", ylabel="count")
plt.title("gender histogram")
plt.show()


# In[52]:


sns.countplot(data=data2,hue="Churn",x=data2.gender)


# In[53]:


a={"age":[10,20],"sex":["m","f"]}
b=pd.DataFrame(a)
b


# In[54]:


b["sex"].replace({"m":1,"f":2},inplace=True)
b


# In[55]:


data2["Churn"].replace({"Yes":1,"No":0},inplace=True)
data2


# # CONVERT ALL CATEGORICAL VARIABLES into dummy  variables

# In[56]:


data2_new=pd.get_dummies(data2)


# In[57]:


data2_new


# In[58]:



data2_new.corr()


# In[59]:


plt.figure(figsize=(20,10))
data2_new.corr()["Churn"].sort_values(ascending=False).plot(kind="bar")


#  #High Churn seen in Month to month contracts, no online security,and no tech support 
#  #where as low churn seen in lon term contract,no internet service and tenure roup of more than 5 years
#  #gender phone service does not have any impact on churn

# In[60]:


plt.figure(figsize=(40,40))
sns.heatmap(data2_new.corr(),annot=True,cmap="Paired")


# In[61]:


#data3=pd.get_dummies(data2_new.Churn,drop_first=False,)


# In[62]:


#data4=pd.merge(left=data2_new,right=data3,left_index=True,right_index=True)
#data4


# In[63]:


#data4.describe(include="all").transpose()


# In[64]:


churn0=data2.loc[data2["Churn"]==0]
churn0


# In[65]:


churn1=data2.loc[data2["Churn"]==1]
churn1


# # Bivariate Analysis

# In[66]:


def relation(df,col,title,hue):
    fig,ax=plt.subplots()
    plt.xticks(rotation=45)
    plt.yscale("log")
    plt.title(title)
    ax=sns.countplot(data=df,x=col,order=df[col].value_counts().index,hue=hue)
    plt.show()


# In[67]:


# to show two categorical variables, we need to use hue.


# In[68]:


relation(churn1,col="Partner",title="Distribution of gender for churned customers",hue="gender")


# In[69]:


relation(churn0,col="Partner",title="Distribution of gender for non churn customer",hue="gender")


# In[70]:


relation(churn1,col="PaymentMethod",title='Distribution of gender for churned customers',hue="gender")


# In[71]:


relation(churn0, col="PaymentMethod",title="Distribution of Non churned customers", hue="gender")


# In[72]:


relation(churn1,col="Contract",hue="gender",title="distribution of gender for churned customers")


# # Insights
# 1.Electronic check medium are the highest churners
# <br>2.Contract Type - Monthly customers are more likely to churn because of no contract terms, as they are free to go customers.

# In[73]:


data2_new.to_csv("final.csv")


# In[74]:


import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[75]:


df=pd.read_csv("final.csv")
df


# In[76]:


df=df.drop(columns=["Unnamed: 0"])


# In[77]:


df.columns


# In[78]:


df


# In[79]:


x=df.drop(columns=["Churn"],axis=1)
x


# In[80]:


y=df["Churn"]
y


# In[81]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40)


# In[82]:


x_train


# In[83]:


print(x_train.dtypes)
x_train.shape


# In[84]:


print(y_train.shape)


# In[85]:


from sklearn.tree import DecisionTreeClassifier


# In[86]:


dt_model=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=4,min_samples_leaf=8)
dt_model.fit(x_train,y_train)


# In[87]:


y_predict=dt_model.predict(x_test)


# In[88]:


y_predict


# In[89]:


accuracy_score(y_test,y_predict)


# In[90]:


dt_model.score(x_test,y_test)


# In[91]:


print(classification_report(y_test,y_predict))


# 1.Accuracy is  low  but we should take it into0n the account since our datset is imbalanced.
# 

# In[92]:


print(metrics.confusion_matrix(y_test,y_predict))


# In[93]:


print("training accuracy:", accuracy_score(y_train,dt_model.predict(x_train)))


# In[94]:


from sklearn import tree
tree.plot_tree(dt_model)


# In[95]:


tree.plot_tree(dt_model,filled=True)


# In[96]:


import os
os.environ["PATH"]+=os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
import graphviz
b=tree.export_graphviz(dt_model,out_file=None,feature_names=None,class_names=None,
                       filled=True,rounded=True,special_characters=True)
graph=graphviz.Source(b)
graph


# In[97]:


from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=100,criterion="gini",random_state = 100,max_depth=6, min_samples_leaf=8)
rf_model


# In[98]:


rf_model.fit(x_train,y_train)


# In[99]:


y_predicted=rf_model.predict(x_test)


# In[100]:


print(classification_report(y_test,y_predicted))


# # Random forest with SMOTE 

# In[101]:


pip install imbalanced-learn


# In[102]:


from collections import Counter
counter=Counter(y_train)
print("before",counter)

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
sm=SMOTEENN()
sm


# In[103]:


x_train_sm,y_train_sm=sm.fit_resample(x_train,y_train)
counter=Counter(y_train_sm)
print("after",counter)


# In[104]:


print(x_train_sm)


# In[105]:


x_train_new,x_test_new,y_train_new,y_test_new=train_test_split(x_train_sm,y_train_sm,random_state=100,test_size=0.2)


# In[106]:


print(x_train_new.shape)
print (x_test_new.shape)
print(y_train_new.shape)
print(y_test_new.shape)


# In[107]:


rf_model_smote=RandomForestClassifier(n_estimators=10,criterion="gini",max_depth=6,min_samples_leaf=8)


# In[108]:


rf_model_smote


# In[109]:


rf_model_smote.fit(x_train_new,y_train_new)


# In[110]:


y_predicted_new=rf_model_smote.predict(x_test_new)
y_predicted_new


# In[111]:


accuracy_score(y_test_new,y_predicted_new)


# In[112]:


print(classification_report(y_test_new,y_predicted_new))


# In[113]:


print(confusion_matrix(y_test_new,y_predicted_new))


# # We got an 94% accuracy (almost a balanced dataset) with Random Forest with SMOTEENN (syntehtic minority oversampling techinique with edited neraest neighbours)

# # Pickle the MOdel

# In[114]:


import pickle


# In[115]:


a=pickle.dump(rf_model_smote,open("saved_model.pkl",'wb'))
a


# In[116]:


load_model=pickle.load(open("saved_model.pkl","rb"))


# In[117]:


load_model.predict(x_test_new)


# In[ ]:





# In[ ]:




