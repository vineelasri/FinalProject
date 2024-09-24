#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score


# ###### Load the dataset from the CSV file

# In[2]:


df = pd.read_csv('AnomaData.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# ##### Exploratory Data Analysis (EDA)

# In[5]:


df.isnull().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


df.describe()


# In[16]:


#df.fillna(df.mean(), inplace=True)


# In[8]:


# Visualize the distribution of each column
df.hist(figsize=(15, 10), bins=20)
plt.show()


# ###### Convert Time Column to Correct Datatype

# In[10]:


if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Extract features like hour, day, month from the 'time' column
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month

    # Drop the 'time' column if no longer necessary
    df.drop(columns=['time'], inplace=True)

# Preview changes
print(df.head())


# ###### Feature Engineering & Feature Selection

# In[11]:


X = df.drop(columns=['y'])
y = df['y']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ##### Train/Test Split with Sampling Distribution

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Check the distribution of the target variable in the train/test sets
print(f'Train set distribution:\n{y_train.value_counts()}')
print(f'Test set distribution:\n{y_test.value_counts()}')


# ##### Metrics for Model Evaluation
# For anomaly detection, accuracy might not always be the best metric, especially for imbalanced datasets. You can also use precision, recall, F1-score, or AUC-ROC.

# In[13]:


def evaluate_model(y_test, y_pred):
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))


# ###### Model Selection, Training, Predicting, and Assessment

# In[14]:


rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
evaluate_model(y_test, y_pred)


# ###### Hyperparameter Tuning and Model Improvement

# In[15]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=2, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit grid search
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
evaluate_model(y_test, y_pred_best)


# ###### Model Validation

# In[17]:


from sklearn.model_selection import cross_val_score

# Cross-validate the best model using 5-fold cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')

# Print cross-validation results
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {np.mean(cv_scores):.2f}')
print(f'Standard Deviation of CV Scores: {np.std(cv_scores):.2f}')


# In[18]:


y_test_pred = best_model.predict(X_test)

# Evaluate model performance on the test set
evaluate_model(y_test, y_test_pred)


# In[19]:


cm = confusion_matrix(y_test, y_test_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Anomaly', 'Anomaly'], yticklabels=['No Anomaly', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# As from above metrics we can see that the success metrics of the accuracy of the model on the test data set is > 75%

# In[ ]:




