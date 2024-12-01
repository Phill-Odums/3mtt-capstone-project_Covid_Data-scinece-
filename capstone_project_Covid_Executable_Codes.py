#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing required libraries
import pandas as pd
import numpy as np
import matplotlib_inline as plot
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#importing the dataset 
covid = pd.read_csv("covid_19_clean_complete.csv")


# In[3]:


# displaying the information on the dataset 
covid


# In[4]:


#using describe  method to check about the dataset 
covid.describe()


# In[5]:


#checking count values of dataset using count method
covid.count()


# In[6]:


# using value_counts method to check number of occurance   
covid.value_counts()


# In[7]:


# checking for number of rows and columns in the dataset using shape method
covid.shape


# In[8]:


# checking for category of columns in the dataset
covid.columns


# In[9]:


# checking for dublicated information in the dataset 
covid.duplicated().sum()


# In[10]:


# checking for empty values in the dataset
covid.isna().sum()


# In[11]:


# dropping irrelevant column in the dataset
covid.drop(["Province/State"],axis=1,inplace= True)


# In[12]:


# reviewing the dataset if columns has been dropped
covid


# In[13]:


# renaming Country/Region column to Country for easy assesing 
covid.rename(columns=  {"Country/Region":"Country"},inplace=True) 


# In[14]:


#converting date column to datetime stamp for extraction
covid["Date"] = pd.to_datetime(covid["Date"])


# In[15]:


# extracting Month from datetime in numerics for modelling using dt.month
covid["Month"] = covid["Date"].dt.month


# In[16]:


# extracting Month from datetime in string for visualization using dt.month_name()
covid["Months"] = covid["Date"].dt.month_name()


# In[17]:


# extracting day from datetime in numerics for modelling using dt.day
covid["Day"] = covid["Date"].dt.day


# In[18]:


# extracting day from datetime in strings for visualization using dt.day_name
covid["Days"] = covid["Date"].dt.day_name()


# In[19]:


# displaying the dataset to check if new features are included
covid


# In[20]:


# Merging longitude and latitude columns into a single column
covid['Coordinates'] = covid.apply(lambda row: (row['Lat'], row['Long']), axis=1)


# In[21]:


# sorting date in ascending order 
covid["Date"] = covid["Date"].sort_values()

#setting date to index 
covid.set_index("Date",inplace=True)


# In[22]:


# sorting country names alphabetically 
covid.sort_values(by="Country",ascending=True,inplace=True)


# In[23]:


#sorting countries based on date index
covid.sort_values(by=["Country","Date"],ascending=True,inplace=True)


# In[24]:


# displaying dataset to see new added features 
covid


# In[25]:


# extracting_daily_growth rate in percentage from the given information in dataset
covid["Daily growth rate"] = ((covid["Confirmed"] - covid["Confirmed"].shift(1))/(covid["Confirmed"].shift(1))*100)


# In[26]:


# displaying information to see new features 
covid


# In[27]:


# extracting Total_population from the given information in the dataset 
covid["Total_Population"] = (covid["Confirmed"]+covid["Deaths"]+covid["Recovered"]+covid["Active"])


# In[28]:


# extracting Mortality_rate from the given information in the dataset
covid["Mortality rate"] = (covid["Deaths"] / covid["Total_Population"])*100 


# In[29]:


# displaying the dataset to see the new features 
covid


# In[30]:


covid["Daily Recovery"] = (covid["Recovered"] - covid["Recovered"].shift(1))/(covid["Recovered"].shift(1))*100


# In[31]:


covid.tail()


# In[32]:


covid["Daily Active"] = (covid["Active"] - covid["Active"].shift(1))/(covid["Active"].shift(1))*100


# In[33]:


covid.tail()


# In[34]:


# grouping confirmed cases per country population using groupby method
case_per_population = covid.groupby("Country")["Confirmed"].sum().reset_index()


# In[35]:


# displaying grouped cases base on country 
case_per_population


# In[36]:


# grouping countries base on their Regions 
WHO_Regions = covid.groupby("Country")["WHO Region"].sum().reset_index()


# In[37]:


# displaying grouped countries base on their regions 
WHO_Regions


# In[38]:


# checking for missing values in the dataset 
covid.isna().sum()


# In[39]:


# filling missing values with 0 instead of NAN 
covid.fillna({"Daily growth rate":0},inplace=True)

covid["Daily growth rate"]


# In[40]:


# checking if missing values are filled
covid.isna().sum()


# In[41]:


# filling the missing values with 0  
covid.fillna({"Mortality rate": (int(0))},inplace=True)
covid["Mortality rate"]


# In[42]:


#checcking dataset if missing values has been filled 
covid.isna().sum()


# In[43]:


# filling the missing values with 0
covid.fillna({"Daily Recovery": (int(0))},inplace=True)
covid.fillna({"Daily Active":   (int(0))},inplace=True) 


# In[44]:


covid.isna().sum()


# In[45]:


covid.duplicated().sum()


# In[46]:


covid.plot(kind = "scatter",x ="Mortality rate",  y ="WHO Region",
           title = "visualizing disease mortality rate per Region".title(),
           figsize = (10,6) )


# In[47]:


covid.plot(kind = "scatter",x ="Daily growth rate",  y ="WHO Region",
           title = "visualizing daily disease growth rate per Region".title(),
           figsize = (10,6) )


# In[48]:


covid.plot(kind = "scatter",x ="Months" ,  y ="Daily growth rate" ,
           title = "visualizing daily growth rate per month".title(),
           figsize = (10,6) )


# In[49]:


# A bar plot to vi
sns.barplot(x ="WHO Region", y ="Mortality rate", data = covid, orient ='vertical')
plt.title("chart to show the death rate in each region".title())


# In[50]:


sns.lineplot(data = covid, x ="Months",y = "Mortality rate",marker="o")   
plt.title = "visualizing daily growth rate per month".title()


# In[51]:


sns.lineplot(data = covid, x ="Months",y = "Mortality rate" ,marker="o",hue = "WHO Region")


# In[52]:


from sklearn.preprocessing import LabelEncoder
encorder = LabelEncoder()
covid["EncCountry"] = encorder.fit_transform(covid["Country"]) 
covid["EncWHO Region"] = encorder.fit_transform(covid["WHO Region"]) 
sns.lmplot(data = covid,
           x = "Mortality rate",y = "EncWHO Region",markers=["*"],row = "Months")


# In[ ]:





# In[53]:


sns.lineplot(data = covid, x =  "Months",y = "Mortality rate",marker="o")


# In[54]:


covid.plot(kind = "scatter",x ="Daily Recovery",  y ="Month" ,
           title = "visualizing disease daily recovery rate per WHO Region".title(),
           figsize = (10,6) )


# In[55]:


covid.plot(kind = "scatter",x ="Mortality rate",  y ="WHO Region",
           title = "visualizing disease mortality rate per Region".title(),
           figsize = (10,6) )
#... other plotting code 


# In[56]:


covid1 = covid.groupby("WHO Region")["Daily growth rate"].count()
fig, ax = plt.subplots(figsize=(7, 12)) # create figure and axes objects
ax.pie(covid1, labels=covid1.index, autopct="%1.0f%%", startangle=90, shadow=True)
ax.set_title("Mortality Rate in All Regions") #set title on axes object
ax.axis("equal")
ax.legend(loc="upper left") 
plt.show()


# In[57]:


#importing labelEncoder for convertint string values to numerics
from sklearn.preprocessing import LabelEncoder


# In[58]:


#creating a new column with converted country and Region to numerics
encorder = LabelEncoder()
covid["EncCountry"] = encorder.fit_transform(covid["Country"]) 
covid["EncWHO Region"] = encorder.fit_transform(covid["WHO Region"]) 


# In[59]:


# dropping duplicates 
covid.drop_duplicates(inplace=True)


# In[60]:


#checking for duplicated values 
covid.duplicated().sum()


# In[61]:


# replace all infinty values with non-numerics and drop all non numerics 
#covid = covid.replace([np.inf, -np.inf], np.nan)dropna()
covid = covid.replace([np.inf, -np.inf], np.nan).fillna(int(0))


# In[62]:


#spliting the dataset into train and test 
x = covid[["Long","Lat","Confirmed","Deaths","Recovered","Active",
           "Daily Active","Daily Recovery","Total_Population","Mortality rate","Day","Month","EncWHO Region","EncCountry"]]
y = round(covid["Daily growth rate"])


# In[63]:


# checking for the shape of the dataset, number of columns and rows
covid.shape


# In[64]:


# converting y and x values to descreate values kusing round function to round up figures
x = np.array((x.round()))
y = np.array((y.round()))


# In[65]:


# displaying dataset to see featues 
covid


# In[66]:


# importing necessary libraries for assigning splitied values into features and target column for machine learning model 
import sklearn      
from sklearn.model_selection import train_test_split


# In[67]:


# assigning splited dataset to test and train
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[68]:


# importing classification model for model building 
#from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier


# In[69]:


#using estimator to improve model performance assigning and assigning to a variable 
#model = RandomForestClassifier(n_estimators=100, max_depth=10)#n_estimators=100, max_depth=10

model = DecisionTreeClassifier() 


# In[70]:


# fitting model 
model.fit(x_train,y_train)


# In[71]:


# predicting model performance 
y_pred = model.predict(x_test).astype(int)
y_pred


# In[72]:


#evaluating model accuracy
from sklearn.metrics import accuracy_score


# In[73]:


# printing model accuracy 
accuracy = accuracy_score(y_test,y_pred)
f"Model Accuracy: {accuracy:.2f}%"


# In[74]:


from sklearn.metrics import classification_report, accuracy_score, precision_score,confusion_matrix,f1_score,recall_score


# In[75]:


# Confusion Matrix
report = confusion_matrix(y_test, y_pred)
report


# In[76]:


# Print classification report
classification_report(y_test,y_pred)


# In[77]:


#printing model precsion score
precision = precision_score(y_test,y_pred,average="weighted")
f"precision score : {precision:.4f}"


# In[78]:


# printing model F1 score"
F1_score = f1_score(y_test,y_pred,average="weighted")
f"F1 Score :{F1_score : 2f}"


# In[79]:


#printing model recall score
recall = recall_score(y_test,y_pred, average="micro")
f"recall : {recall:.2f} " 


# In[ ]:





# In[ ]:




