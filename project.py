#!/usr/bin/env python
# coding: utf-8

# Goal It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable.
# 
# Metric Submissions are evaluated on Mean-Squared-Error (MSE). Submission File Format The file should contain a header and have the following format:
# 
# Id,SalePrice 1461,169000.1 1462,187724.1233 1463,175221
# 
# You can see an example submission file (sample_submission.csv)

# # FILTER WARNINGS

# In[1]:


from warnings import filterwarnings
filterwarnings("ignore")


# # READING DATASET

# In[2]:


import pandas as pd
train = pd.read_csv("C:/Users/hp/OneDrive/Desktop/DS/project1/training_set.csv")
test = pd.read_csv("C:/Users/hp/OneDrive/Desktop/DS/project1/testing_set.csv")


# In[3]:


train.head(2)


# # MISSING DATA TREATMENT

# In[4]:


train.isna().sum()


# In[5]:


train.Alley=train.Alley.fillna("No alley access")
train.BsmtQual=train.BsmtQual.fillna("No Baseme")
train.BsmtCond =train.BsmtCond .fillna("No Basement")
train.BsmtExposure=train.BsmtExposure.fillna("No Basements")
train.BsmtFinType1=train.BsmtFinType1.fillna("No Basement")
train.BsmtFinType2=train.BsmtFinType2.fillna("No Basement")
train.FireplaceQu=train.FireplaceQu.fillna("No Fireplace")
train.GarageType=train.GarageType.fillna("No Garage")
train.GarageFinish=train.GarageFinish.fillna("No Garage")
train.GarageQual=train.GarageQual.fillna("No Garage")
train.GarageCond=train.GarageCond.fillna("No Garage")
train.PoolQC=train.PoolQC.fillna("No Pool")
train.Fence=train.Fence.fillna("No Fence")
train.MiscFeature=train.MiscFeature.fillna("None")

test.Alley=train.Alley.fillna("No alley access")
test.BsmtQual=train.BsmtQual.fillna("No Baseme")
test.BsmtCond =train.BsmtCond .fillna("No Basement")
test.BsmtExposure=train.BsmtExposure.fillna("No Basements")
test.BsmtFinType1=train.BsmtFinType1.fillna("No Basement")
test.BsmtFinType2=train.BsmtFinType2.fillna("No Basement")
test.FireplaceQu=train.FireplaceQu.fillna("No Fireplace")
test.GarageType=train.GarageType.fillna("No Garage")
test.GarageFinish=train.GarageFinish.fillna("No Garage")
test.GarageQual=train.GarageQual.fillna("No Garage")
test.GarageCond=train.GarageCond.fillna("No Garage")
test.PoolQC=train.PoolQC.fillna("No Pool")
test.Fence=train.Fence.fillna("No Fence")
test.MiscFeature=train.MiscFeature.fillna("None")


# In[6]:


from function import replacer
replacer(train)
replacer(test)


# # Define X , Y and remove all columns which are not required

# In[69]:


cat = []
con = []
for i in train.columns:
    if(train[i].dtypes == "object"):
        cat.append(i)
    else:
        con.append(i)


# In[9]:


train.corr()["SalePrice"].sort_values()


# In[10]:


train.isna().sum()


# In[11]:


X = train.drop(labels=["Id","SalePrice"],axis=1)
Y = train[["SalePrice"]]


# In[12]:


X.isna().sum()


# # REMOVE OUTLIERS

# In[13]:


from function import standardize,outliers
X1 = standardize(X)
OL = outliers(X1)


# In[14]:


X1.isna().sum()


# In[15]:


X = X.drop(index=OL,axis=0)
Y = Y.drop(index=OL,axis=0)


# In[16]:


X.shape


# In[17]:


X.index = range(0,1021,1)
Y.index = range(0,1021,1)


# # PREPROCESSING

# In[18]:


from function import preprocessing
Xnew = preprocessing(X)


# In[19]:


Xnew.isna().sum()


# # SPLITING DATASET INTO TEST AND TRAIN

# In[20]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# # Exploratory Data Analysis(EDA)

# In[71]:


train.corr()["SalePrice"].sort_values()


# In[21]:


from statsmodels.api import OLS,add_constant
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst)
model = ol.fit()
rsq = model.rsquared_adj
col_to_drop = model.pvalues.sort_values().index[-1]
print("Dropped: column:",col_to_drop,"\tRsquared:",round(rsq,4))
Xnew = Xnew.drop(labels=col_to_drop,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst)
model = ol.fit()
rsq = model.rsquared_adj


# In[74]:


for i in range(0,10):
    from statsmodels.api import OLS,add_constant
    xconst = add_constant(xtrain)
    ol = OLS(ytrain,xconst)
    model = ol.fit()
    rsq = model.rsquared_adj
    col_to_drop = model.pvalues.sort_values().index[-1]
    print("Dropped: column:",col_to_drop,"\tRsquared:",round(rsq,4))
    Xnew = Xnew.drop(labels=col_to_drop,axis=1)
    xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)
    xconst = add_constant(xtrain)
    ol = OLS(ytrain,xconst)
    model = ol.fit()
    rsq = model.rsquared_adj


# # Overfitting

# In[75]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err = mean_absolute_error(ytest,ts_pred)


# In[76]:


model


# In[24]:


tr_err


# In[25]:


ts_err


# In[26]:


Xnew.corr()


# # Linerization

# In[27]:


lambdas = []
q = 8
for i in range(0,4000,1):
    q = q + 0.001
    q = round(q,4)
    lambdas.append(q)


# # Ridge implementation

# In[28]:


from sklearn.linear_model import Ridge
tr = []
ts = []
for i in lambdas:
    rr = Ridge(alpha=i)
    model = rr.fit(xtrain,ytrain)
    tr_pred = model.predict(xtrain)
    ts_pred = model.predict(xtest)
    from sklearn.metrics import mean_absolute_error
    tr_err = mean_absolute_error(ytrain,tr_pred)
    ts_err = mean_absolute_error(ytest,ts_pred)
    tr.append(tr_err)
    ts.append(ts_err)


# In[29]:


t = range(0,4000,1)


# In[30]:


import matplotlib.pyplot as plt
plt.plot(t,tr,c="red")
plt.plot(t,ts,c="blue")


# In[31]:


lambdas[-1]


# In[32]:


tr_err


# In[33]:


ts_err


# In[34]:


model


# In[35]:


rr = Ridge(alpha=12.0)
model = rr.fit(xtrain,ytrain)


# In[36]:


xtrain.columns


# # Making testing data set ready

# In[40]:


xtest = test.drop(labels=["Id"],axis=1)
xtest_new = preprocessing(xtest)


# In[78]:


#xtest_new[list(xtrain.columns)]


# In[41]:


X2=pd.read_csv("C:/Users/hp/OneDrive/Desktop/DS/project1/testing_set.csv")


# In[42]:


replacer(X2)


# In[43]:


X2new=preprocessing(X2)


# In[45]:


X2new[list(Xnew.columns)]


# In[53]:


q=['Alley_No alley access', 'Condition2_RRNn', 'Exterior1st_ImStucc', 'Exterior2nd_Other', 'BsmtQual_No Baseme', 'BsmtCond_No Basement', 'BsmtExposure_No Basements', 'BsmtFinType1_No Basement', 'BsmtFinType2_No Basement', 'Heating_Floor', 'FireplaceQu_No Fireplace', 'GarageType_No Garage', 'GarageFinish_No Garage', 'GarageQual_Ex', 'GarageQual_No Garage', 'GarageCond_No Garage', 'PoolQC_No Pool', 'Fence_No Fence', 'MiscFeature_None']


# In[57]:


for i in q:
    X2new[i]=0


# In[58]:


X2new.shape


# In[59]:


final_result = X2new[list(xtrain.columns)]


# # Make Prediction

# In[60]:


pred = model.predict(final_result)


# In[62]:


Q = test[["Id"]]


# In[63]:


Q['SalePrice']=pred


# In[65]:


Q.to_csv("Desktop/final_submissions.csv")


# In[ ]:




