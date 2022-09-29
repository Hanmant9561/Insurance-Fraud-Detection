#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# The goal of this project is to build a model that can detect auto insurance fraud. The challenge behind fraud detection in machine learning is that frauds are far less common as compared to legit insurance claims. This type of problems is known as imbalanced class classification.
# 
# Frauds are unethical and are losses to the company. By building a model that can classify auto insurance fraud, I am able to cut losses for the insurance company. Less losses equates to more earning.

# In[1]:


# import libraries

import numpy as np
import pandas as pd


# In[2]:


# load the dataset

df = pd.read_csv('insurance_claims.csv')
df.head()                                                    # observe the dataset


# In[3]:


# observe the dimensions of dataset

df.shape


# In[4]:


# check the characteristic of data set

df.info()


# In[5]:


# check basic statstical characteristic of data set

df.describe()


# In[6]:


object_columns = list(df.select_dtypes(include=['object']).columns)
object_columns


# In[7]:


df.head()


# In[8]:


# let's check whether the data has any null values or not.

# but there are '?' in the datset which we have to remove by NaN Values


df = df.replace('?',np.NaN)

df.isnull().any().any()


# In[9]:


# missing value treatment using fillna

# we will replace the '?' by the most common collision type as we are unaware of the type.
df['collision_type'].fillna(df['collision_type'].mode()[0], inplace = True)

# It may be the case that there are no responses for property damage then we might take it as No property damage.
df['property_damage'].fillna('NO', inplace = True)

# again, if there are no responses fpr police report available then we might take it as No report available
df['police_report_available'].fillna('NO', inplace = True)

df.isnull().any().any()


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


df.shape


# In[13]:


df.head()


# In[14]:


df.columns


# # Visualization

# In[15]:


import plotly.express as px
fig = px.histogram(df, x="fraud_reported",title='Distribution of Fraud',color='fraud_reported')
fig.update_layout(bargap=0.3)
fig.show()


# In[16]:


# lets see sex wise fraud reported

fig = px.histogram(df, x="fraud_reported", color="insured_sex", title="sex vs fraud_reported")
fig.update_layout(bargap=0.3)
fig.show()


# In[17]:


# see in which state and how many fraud reported

fig = px.histogram(df, x="fraud_reported",color='incident_state')
fig.update_layout(bargap=0.3)
fig.show()


# In[18]:


fig = px.histogram(df, x="fraud_reported", color="insured_sex", pattern_shape="age")
fig.show()


# In[19]:


# in which city fraud reported


fig = px.bar(df, x="fraud_reported", y="incident_city", color="incident_city", title="Fraud reported in City")
fig.update_layout(bargap=0.3)
fig.show()


# In[20]:


import plotly.express as px
fig = px.pie(df, values='witnesses', names='incident_city', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()


# In[21]:


import plotly.express as px
fig = px.box(df, x="fraud_reported", y="total_claim_amount", points="all")
fig.show()


# In[22]:


import plotly.express as px



fig = px.box(df, x="fraud_reported", y="capital-loss", color="incident_state")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()


# In[23]:


import plotly.express as px



fig = px.box(df, x="fraud_reported", y="capital-gains", color="incident_state")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()


# In[24]:


import plotly.express as px

fig = px.histogram(df, x="policy_annual_premium")
fig.update_layout(bargap=0.2)
fig.show()


# In[25]:


import plotly.express as px

fig = px.histogram(df, x="insured_hobbies")
fig.update_layout(bargap=0.2)
fig.show()


# In[26]:


import plotly.express as px

fig = px.histogram(df, x="insured_occupation")
fig.update_layout(bargap=0.2)
fig.show()


# In[27]:


import plotly.express as px

fig = px.violin(df, y="incident_date", x="fraud_reported", color="insured_sex", box=True, points="all",
          hover_data=df.columns)
fig.show()


# In[28]:


import plotly.express as px
import numpy as np
fig = px.scatter_3d(df, x='age', y='incident_location', z='witnesses', size='number_of_vehicles_involved', color='incident_type',
                    hover_data=['incident_state'])
fig.update_layout(scene_zaxis_type="log")
fig.show()


# In[29]:


import plotly.express as px

fig = px.scatter(df, x="policy_annual_premium", y="total_claim_amount",
	         size="capital-gains", color="property_claim",
                 hover_name="insured_education_level", log_x=True, size_max=60)
fig.show()


# In[30]:


import plotly.express as px
fig = px.scatter(df, x="property_claim", y="vehicle_claim", color="insured_zip",
                 size='auto_year', hover_data=['policy_number'])
fig.show()


# In[31]:


# check property damage and total calim amount

fig = px.bar(df, x='property_damage', y='total_claim_amount',color='total_claim_amount')
fig.show()


# In[32]:


# check no. of vehicles involved in incident in a particular city

fig = px.bar(df, x='incident_city', y='number_of_vehicles_involved',color='number_of_vehicles_involved')
fig.show()


# In[33]:


import plotly.graph_objects as go

labels = df['insured_education_level']


# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels,  pull=[0.1, 0.2, 0.3, 0.4,0.5,])])
fig.show()


# In[34]:


import plotly.graph_objects as go


fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="count", y=df['incident_city'], x=df['incident_type'], name="incident_city"))

fig.show()


# In[35]:


import plotly.express as px
fig = px.funnel(df, x='incident_hour_of_the_day', y='witnesses')
fig.show()


# In[36]:


import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Bar(
  x = df['fraud_reported'],
  y = df['incident_hour_of_the_day'],
  name = "incident_hour_of_the_day",
))

fig.add_trace(go.Bar(
  x = df['fraud_reported'],
  y = df['number_of_vehicles_involved'],
  name = "'number_of_vehicles_involved",
))

fig.add_trace(go.Bar(
  x = df['fraud_reported'],
  y = df['witnesses'],
  name= 'witnesses'))

fig.update_layout(title_text="Multi-category axis")

fig.show()


# In[37]:


import plotly.express as px

fig = px.strip(df, x='property_damage', y='property_claim')
fig.show()


# # Preprocessing

# In[38]:


object_columns = list(df.select_dtypes(include=['object']).columns)
object_columns


# In[39]:


# let's extrat days, month and year from policy bind date

df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors = 'coerce')


# In[40]:


# let's encode the fraud report to numerical values

df['fraud_reported'] = df['fraud_reported'].replace(('Y','N'),(0,1))


# In[41]:


# let's check the correlation of authorities_contacted with the target

df[['auto_model','fraud_reported']].groupby(['auto_model'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[42]:


# let's perform target encoding for auto make

df['auto_model'] = df['auto_model'].replace(('3 Series','RSX','Malibu','Wrangler','Pathfinder','Ultima','Camry',
                'Corolla','CRV','Legacy','Neon','95','TL','93','MDX','Accord','Grand Cherokee','Escape','E4000',
            'A3','Highlander','Passat','92x','Jetta','Fusion','Forrestor','Maxima','Impreza','X5','RAM','M5','A5',
                'Civic','F150','Tahaoe','C300','ML350','Silverado','X6'),
                (0.95,0.91, 0.90,0.88,0.87,0.86,0.855,0.85,0.85,0.84,0.83,0.81,0.80,0.80,0.78,0.77,0.76,0.75,0.74,
                 0.73,0.72,0.72,0.71,0.71,0.71,0.71,0.70,0.70,0.69,0.67,0.66,0.65,0.64,0.63,0.62,0.61,0.60,0.59,0.56))


# In[43]:


# let's check the correlation auto make with the target

df[['auto_make','fraud_reported']].groupby(['auto_make'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[44]:


# let's perform target encoding for auto make

df['auto_make'] = df['auto_make'].replace(('Jeep','Nissan','Toyota','Accura','Saab','Suburu',
                                'Dodge','Honda','Chevrolet','BMW','Volkswagen','Audi','Ford','Mercedes'),
                                              (0.84,0.82,0.81,0.80,0.77,0.76,0.75,0.74,0.73,0.72,0.71,0.69,0.69,0.66))


# In[45]:


# let's check the correlation of authorities_contacted with the target

df[['police_report_available','fraud_reported']].groupby(['police_report_available'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[46]:


# let's perform target encoding for property damage

df['police_report_available'] = df['police_report_available'].replace(('NO','YES'),(0.77,0.74))


# In[47]:


# let's check the correlation of authorities_contacted with the target

df[['property_damage','fraud_reported']].groupby(['property_damage'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[48]:


# target encoding for policy_csl

df['property_damage'] = df['property_damage'].replace(('YES','NO'),(0.74,0.75,))


# In[49]:


# let's check the correlation of authorities_contacted with the target

df[['incident_city','fraud_reported']].groupby(['incident_city'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[50]:


# let's do target encoding for incident city

df['incident_city'] = df['incident_city'].replace(('Northbrook','Riverwood','Northbend','Springfield',
                                    'Hillsdale','Columbus','Arlington'),(0.78,0.77,0.76,0.75,0.74,0.73,0.71))


# In[51]:


# let's check the correlation of authorities_contacted with the target

df[['incident_state','fraud_reported']].groupby(['incident_state'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[52]:


# let's perform target encoding for incident state

df['incident_state'] = df['incident_state'].replace(('WV','NY','VA','PA','SC','NC','OH'),
                                                        (0.82,0.77,0.76,0.73,0.70,0.69,0.56))


# In[53]:


# let's check the correlation of incident_severity with the target

df[['incident_severity','fraud_reported']].groupby(['incident_severity'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[54]:


# let's check the correlation of authorities_contacted with the target

df[['authorities_contacted','fraud_reported']].groupby(['authorities_contacted'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[55]:


# let's perform target encoding for authorities contacted

df['authorities_contacted'] = df['authorities_contacted'].replace(('None','Police','Fire','Ambulance','Other'),
                                                                      (0.94,0.79,0.73,0.70,0.68))


# In[56]:


# let's check the correlation of incident_severity with the target

df[['incident_severity','fraud_reported']].groupby(['incident_severity'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[57]:


# let's perform target encoding for incident severity

df['incident_severity'] = df['incident_severity'].replace(('Trivial Damage','Minor Damage','Total Loss',
                                                              'Major Damage'),(0.94,0.89,0.87,0.39))


# In[58]:


# let's check the correlation of collision_type with the target

df[['collision_type','fraud_reported']].groupby(['collision_type'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[59]:


# let's perform target encoding for collision type

df['collision_type'] = df['collision_type'].replace(('Rear Collision', 'Side Collision', 'Front Collision'),
                                                        (0.78,0.74,0.72))


# In[60]:


# let's check the correlation of incident_type with the target

df[['incident_type','fraud_reported']].groupby(['incident_type'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[61]:


# let's perform target encoing for incident type

df['incident_type'] = df['incident_type'].replace(('Vehicle Theft','Parked Car','Multi-vehicle Collision',
                                'Single Vehicle Collision'),(0.91, 0.90, 0.72,0.70))


# In[62]:


df['incident_date'] = pd.to_datetime(df['incident_date'], errors = 'coerce')

# extracting days and month from date
df['incident_month'] = df['incident_date'].dt.month
df['incident_day'] = df['incident_date'].dt.day


# In[63]:


# let's know the relation between insured_relationship and fraud reported

df[['insured_relationship','fraud_reported']].groupby(['insured_relationship'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[64]:


# let's do target encoding for insured relationship

df['insured_relationship'] = df['insured_relationship'].replace(('husband','own-child','unmarried',
                                        'not-in-family','wife','other-relative'),(0.79,0.78,0.75,0.74,0.72,0.70))


# In[65]:


# let's know the relation between insured_hobbies and fraud reported

df[['insured_hobbies','fraud_reported']].groupby(['insured_hobbies'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[66]:


# let's know the relation between insured_occupation and fraud reported

df[['insured_occupation','fraud_reported']].groupby(['insured_occupation'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[67]:


# let's perform target encoding for insured_hobbies

df['insured_hobbies'] = df['insured_hobbies'].replace(('camping', 'kayaking', 'golf','dancing',
        'bungie-jumping','movies', 'basketball','exercise','sleeping','video-games','skydiving','paintball',
            'hiking','base-jumping','reading','polo','board-games','yachting', 'cross-fit','chess'),(0.91, 0.90,
                0.89, 0.88,0.84,0.83,0.82,0.81,0.805,0.80,0.78,0.77,0.76,0.73,0.73,0.72,0.70,0.69,0.25,0.17))


# In[68]:


# let's know the relation of insured_education_level with faud_reported

df[['insured_education_level','fraud_reported']].groupby(['insured_education_level'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[69]:


# let's perform target encoding for insured_occupation

df['insured_occupation'] = df['insured_occupation'].replace(('other-service','priv-house-serv',
                        'adm-clerical','handlers-cleaners','prof-specialty','protective-serv',
                'machine-op-inspct','armed-forces','sales','tech-support','transport-moving','craft-repair',
                    'farming-fishing','exec-managerial'),(0.84, 0.84,0.83, 0.79,0.78,0.77,0.76,0.75,0.72,0.71,
                                                          0.705,0.70,0.69,0.63))


# In[70]:


# let's know the relation of insured_education_level with faud_reported

df[['insured_education_level','fraud_reported']].groupby(['insured_education_level'], 
                as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)


# In[71]:


# let's perform target encoding

df['insured_education_level'] = df['insured_education_level'].replace(('Masters', 'High School','Associate',
                                        'JD','College', 'MD','PhD'),(0.78,0.77,0.76,0.74,0.73,0.72,0.71))


# In[72]:


# lets know the relation of insured sex and fraud reported

df[['insured_sex','fraud_reported']].groupby(['insured_sex'], as_index = False).mean().sort_values(
    by = 'fraud_reported', ascending = False)


# In[73]:


# target encoding for sex

df['insured_sex'] = df['insured_sex'].replace(('FEMALE','MALE'),(0.76,0.73))


# # csl - combined single limit
# CSL is a single number that describes the predetermined limit for the combined total of the Bodily Injury Liability coverage and Property Damage Liability coverage per occurrence or accident.

# In[74]:


# lets know the relation of policy state and fraud reported

df[['policy_csl','fraud_reported']].groupby(['policy_csl'], as_index = False).mean().sort_values(
    by = 'fraud_reported', ascending = False)


# In[75]:


# target encoding for policy_csl

df['policy_csl'] = df['policy_csl'].replace(('500/1000','100/300','250/500'),(0.78,0.74,0.73))


# In[76]:


# lets know the relation of policy state and fraud reported

df[['policy_state','fraud_reported']].groupby(['policy_state'], as_index = False).mean().sort_values(
    by = 'fraud_reported', ascending = False)


# In[77]:


# target encoding for policy_csl

df['policy_state'] = df['policy_state'].replace(('IL','IN','OH'),(0.77,0.745,0.74))


# In[78]:


# let's delete unnecassary columns

df = df.drop(['policy_number','policy_bind_date', 'incident_date','incident_location','auto_model'], axis = 1)

# let's check the columns after deleting the columns
df.columns


# In[79]:


# let's split the data into dependent and independent sets

x = df.drop(['fraud_reported'], axis = 1)
y = df['fraud_reported']

print("Shape of x :", x.shape)
print("Shape of y :", y.shape)


# In[80]:


# let's split the dataset into train and test sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)


# In[81]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (15, 10)
sns.heatmap(x_train.corr(), cmap = 'copper')
plt.title('Heat Map for Correlations', fontsize = 20)
plt.show()


# # Feature Selection

# In[82]:


# Using the ANOVA Test to check the best features
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  # naming the dataframe columns
print("These are the best 10  features of our data")
print(featureScores.nlargest(10,'Score'))  # print 10 best features


# # PCA
# Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

# # Note
# NO Need to Use the PCA Data I performed it just to get the idea, Use Best 10 Features for the model building Use the train_test_split method.

# # Model Building

# In[83]:


# split the data into independent and dependent variable

x = df[['incident_severity','insured_hobbies', 'incident_type', 'vehicle_claim', 'total_claim_amount',
       'authorities_contacted','property_claim','insured_occupation','incident_state','auto_make']]
y = df['fraud_reported']

print("Shape of x :", x.shape)
print("Shape of y :", y.shape)


# In[84]:


x.head()


# In[85]:


y.head()


# In[86]:


# split data into train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[87]:


x_train


# In[88]:


y_train,len(y_train)


# In[89]:


y.value_counts()


# # Over Sampling

# In[90]:


import warnings
warnings.filterwarnings("ignore")


# In[91]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter


# In[92]:


os = RandomOverSampler(0.8)
x_train_os, y_train_os = os.fit_resample(x_train,y_train)
print('The number of Classes before fit {}'.format(Counter(y_train)))
print('The number of Classes after fit {}'.format(Counter(y_train_os)))


# Now we've got our data split into traing and test sets,it's time to build a machine learning model.
# 
# we'll train it(find the patterns)on the training set.
# 
# And we'll test it(use the patterns)on the test set
# 
# we're going to try 6 different machine learning models
# 
# 1.Logistic Regression
# 
# 2.Support Vector Machine
# 
# 3.K- Nearest Neighbors Classifier
# 
# 4.Random Forest Classifier
# 
# 5.XGBoost
# 
# 6.Naive Bayes

# # Logististic Regression

# In[93]:


# fit the model

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression() 
LR.fit(x_train_os,y_train_os)  


# In[94]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[95]:


y_pred_test = LR.predict(x_test)   
y_pred_train = LR.predict(x_train)


# In[96]:


from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(y_test,y_pred_test)
accuracy_train = accuracy_score(y_train,y_pred_train)
print('test accuracy is:', accuracy_test)
print('train accuracy is:',accuracy_train)


# In[97]:


y_pred = LR.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[100]:


log_model = LogisticRegression()

param_grid = [
              {'penalty' : ['l1','l2','elasticnet', 'none'],
     'C' : np.logspace(-4,4,20),
     'solver' : [ 'newton-cg','lbfgs','liblinear','sag','saga'],
     'max_iter' : [100,1000,2500,5000]
              }
    

]

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(log_model,param_grid=param_grid,cv=3,verbose=True,n_jobs=-1)
best_clf = clf.fit(x_train,y_train)
print("Best Hyper Parameters:\n",clf.best_params_)
prediction=clf.predict(x_test)
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

print("Accuracy:",metrics.accuracy_score(prediction,y_test))
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y_test))
print('Classification_report:\n',metrics.classification_report(prediction,y_test))


# # Support Vector Machine

# In[101]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train_os,y_train_os)


# In[102]:


y_pred_test = svm.predict(x_test)   
y_pred_train = svm.predict(x_train)


# In[103]:


from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(y_test,y_pred_test)
accuracy_train = accuracy_score(y_train,y_pred_train)
print('test accuracy is:', accuracy_test)
print('train accuracy is:',accuracy_train)


# In[104]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))


# # K- Nearest Neighbors Classifier

# In[105]:


# fit the model

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train_os,y_train_os)


# In[106]:


y_pred_test = knn.predict(x_test)
y_pred_train = knn.predict(x_train)


# In[107]:


from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(y_test,y_pred_test)
accuracy_train = accuracy_score(y_train,y_pred_train)
print('test accuracy is:',accuracy_test)
print('train accuracy is:',accuracy_train)


# In[108]:


# k-fold cross validation

from sklearn.model_selection import KFold
knn = KNeighborsClassifier()
kfold_validation = KFold(10)


# In[109]:


from sklearn.model_selection import cross_val_score
results = cross_val_score(knn,x,y, cv = kfold_validation)
print(results)
print(np.mean(results))


# In[110]:


# stratified k-fold cross validation

from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5)
knn = KNeighborsClassifier()
scores = cross_val_score(knn,x,y, cv = skfold)
print(np.mean(scores))


# In[111]:


# leave one out cross validation (voocv)

from sklearn.model_selection import LeaveOneOut
knn = KNeighborsClassifier()
leave_validation = LeaveOneOut()
results = cross_val_score(knn,x,y, cv = leave_validation)
print(np.mean(results))


# In[112]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))


# # Random Forest Classifier

# In[113]:


# fit the model

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train_os,y_train_os)


# In[114]:


y_pred_test = rf.predict(x_test)
y_pred_train = rf.predict(x_train)


# In[115]:


from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(y_test,y_pred_test)
accuracy_train = accuracy_score(y_train,y_pred_train)
print('test accuracy is:',accuracy_test)
print('train accuracy is:',accuracy_train)


# In[116]:


# k-fold cross validation

from sklearn.model_selection import KFold
rf = RandomForestClassifier()
kfold_validation = KFold(10)


# In[117]:


from sklearn.model_selection import cross_val_score
results = cross_val_score(rf,x,y, cv = kfold_validation)
print(results)
print(np.mean(results))


# In[121]:


# stratified k-fold cross validation

from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5)
rf = RandomForestClassifier()
scores = cross_val_score(rf,x,y, cv = skfold)
print(np.mean(scores))


# In[122]:


# leave one out cross validation (voocv)

from sklearn.model_selection import LeaveOneOut
rf = RandomForestClassifier()
leave_validation = LeaveOneOut()
results = cross_val_score(rf,x,y, cv = leave_validation)
print(np.mean(results))


# In[123]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))


# # XGBoost

# In[124]:


# fit the model

from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(x_train_os,y_train_os)


# In[125]:


y_pred_test = clf.predict(x_test)
y_perd_train = clf.predict(x_train)


# In[126]:


from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(y_test,y_pred_test)
accuracy_tarain = accuracy_score(y_train,y_pred_train)
print('test accuracy is:',accuracy_test)
print('train accuracy is:',accuracy_train)


# In[127]:


# k-fold cross validation

from sklearn.model_selection import KFold
clf = XGBClassifier()
kfold_validation = KFold(10)


# In[128]:


from sklearn.model_selection import cross_val_score
results = cross_val_score(clf,x,y, cv = kfold_validation)
print(results)
print(np.mean(results))


# In[131]:


# stratified k-fold cross validation

from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5)
clf = XGBClassifier()
scores = cross_val_score(clf,x,y, cv = skfold)
print(np.mean(scores))
import warnings
warnings.filterwarnings('ignore')


# In[132]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))


# # Naive Bayes

# In[133]:


# fit the model

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train_os,y_train_os)


# In[134]:


y_pred_test = nb.predict(x_test)
y_perd_train = nb.predict(x_train)


# In[135]:


from sklearn.metrics import accuracy_score
accuracy_test = accuracy_score(y_test,y_pred_test)
accuracy_tarain = accuracy_score(y_train,y_pred_train)
print('test accuracy is:',accuracy_test)
print('train accuracy is:',accuracy_train)


# In[136]:


# k-fold cross validation

from sklearn.model_selection import KFold
nb = GaussianNB()
kfold_validation = KFold(10)


# In[137]:


from sklearn.model_selection import cross_val_score
results = cross_val_score(nb,x,y, cv = kfold_validation)
print(results)
print(np.mean(results))


# In[138]:


# stratified k-fold cross validation

from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5)
nb = GaussianNB()
scores = cross_val_score(nb,x,y, cv = skfold)
print(np.mean(scores))


# In[139]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))


# # Hypertuning Parameter
# A hyperparameter is a parameter whose value is set before the learning process begins.

# In[140]:


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()

params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB = GridSearchCV(estimator=nb_classifier, 
                 param_grid=params_NB, 
                    # use any cross validation technique 
                 verbose=1, 
                 scoring='accuracy') 
gs_NB.fit(x_train, y_train)
print("Best Hyper Parameters:\n",gs_NB.best_params_)
prediction=gs_NB.predict(x_test)


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(prediction,y_test))

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y_test))
print('Classification_report:\n',metrics.classification_report(prediction,y_test))


# In[141]:


#importing modules
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#making the instance
knn_classifier = KNeighborsClassifier(n_jobs=-1)
#Hyper Parameters Set
params = {'n_neighbors':[5,6,7,8,9,10],
          'leaf_size':[1,2,3,5],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
model1 = GridSearchCV(knn_classifier, param_grid=params, n_jobs=1)
#Learning
model1.fit(x_train,y_train)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(x_test)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y_test))
print('Classification_report:\n',metrics.classification_report(prediction,y_test))


# In[142]:


#importing modules
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#making the instance
rf_classifier=RandomForestClassifier()
#hyper parameters set
params = {'criterion':['gini','entropy'],
          'n_estimators':[10,15,20,25,30],
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[3,4,5,6,7], 
          'random_state':[123],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
model = GridSearchCV(rf_classifier, param_grid=params, n_jobs=-1)
#learning
model.fit(x_train,y_train)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(x_test)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y_test))
print('Classification_report:\n',metrics.classification_report(prediction,y_test))


# In[143]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

xgb_classifier = XGBClassifier()

param = {'learning_rate': [0.001,0.01,0.1,0.25],
         'max_depth': [1,2,3,4,5,6],
         'max_features': [1,2,3,4,5,6],
         'n_estimators': [20,40,50,70,100]}

grid_search = GridSearchCV(xgb_classifier, param_grid=params, n_jobs=-1)
grid_search.fit(x_train,y_train)

print("Best Hyper Parameters:\n",grid_search.best_params_)
prediction=grid_search.predict(x_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(prediction,y_test))

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y_test))
print('Classification_report:\n',metrics.classification_report(prediction,y_test))


# In[144]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


# In[145]:


# Prediction probabilities
model1.fit(x_train,y_train)

r_probs = [0 for _ in range(len(y_test))]
lr_probs = best_clf.predict_proba(x_test)
knn_probs = model1.predict_proba(x_test)
rf_probs = model.predict_proba(x_test)
xgb_probs = grid_search.predict_proba(x_test)
nb_probs = gs_NB.predict_proba(x_test)


# In[146]:


lr_probs = lr_probs[:,1]
knn_probs = knn_probs[:,1]
rf_probs = rf_probs[:,1]
xgb_probs = xgb_probs[:,1]
nb_probs = nb_probs[:,1]


# In[147]:


# claculate AUROC

lr_auc = roc_auc_score(y_test,lr_probs)
knn_auc = roc_auc_score(y_test,knn_probs)
rf_auc = roc_auc_score(y_test,rf_probs)
xgb_auc = roc_auc_score(y_test,xgb_probs)
nb_auc = roc_auc_score(y_test,nb_probs)


# In[148]:


print('Logistic Regression: AUROC = % 0.3f' %(lr_auc))
print('KNNeighbors: AUROC = % 0.3f' %(knn_auc))
print('Random Forest: AUROC = % 0.3f' %(rf_auc))
print('XGBoost: AUROC = % 0.3f' %(xgb_auc))
print('Naive Bayes: AUROC = % 0.3f' %(nb_auc))


# In[149]:


# Predicting test set results

lr_fpr, lr_tpr, _ = roc_curve(y_test,lr_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test,knn_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test,rf_probs)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test,xgb_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test,nb_probs)


# In[150]:


def plot_roc_curve(rf_fpr,rf_tpr):
    plt.plot(lr_fpr, lr_tpr,marker = '.', label='Logistic Regression: AUROC = % 0.3f' %(lr_auc))
    plt.plot(knn_fpr, knn_tpr,marker = '.', label='KNNeighbor: AUROC = % 0.3f' %(knn_auc))
    plt.plot(rf_fpr, rf_tpr,marker = '.', label='Random Foret: AUROC = % 0.3f' %(rf_auc))
    plt.plot(xgb_fpr, xgb_tpr,marker = '.', label='XGBoost: AUROC = % 0.3f' %(xgb_auc))
    plt.plot(nb_fpr, nb_tpr,marker = '.', label='Naive Bayes: AUROC = % 0.3f' %(nb_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reveiver Operating Characteristics (ROC) Curve')
    plt.legend()
    plt.show()


# In[151]:


plot_roc_curve(rf_fpr,rf_tpr)

