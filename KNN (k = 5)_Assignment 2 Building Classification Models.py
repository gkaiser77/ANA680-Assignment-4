#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


from sklearn.metrics import accuracy_score, confusion_matrix
from ucimlrepo import fetch_ucirepo 
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle


# In[3]:


# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features 
y = breast_cancer_wisconsin_original.data.targets 
  
# variable information 
print(breast_cancer_wisconsin_original.variables) 


# In[4]:


X['Bare_nuclei'].unique()


# In[5]:


imputer = SimpleImputer(strategy='median')

X.loc[:, 'Bare_nuclei'] = imputer.fit_transform(X[['Bare_nuclei']])


# In[6]:


X['Bare_nuclei'].unique()


# In[7]:


y = np.where(y == 2, 0, 1)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[9]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[10]:


y_train = y_train.ravel() 


# In[11]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)


# In[12]:


y_pred = model.predict(X_test)


# In[13]:


accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("KNN (k=5)")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)


# In[14]:


# Save the trained model as a pickle file
model_filename = 'knn_breast_cancer_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved as {model_filename}")

