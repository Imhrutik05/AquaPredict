# %% [markdown]
# ## Importing Libraries

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
import pickle
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# %%
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Importing Data

# %%
data = pd.read_csv(r"D:\01_Projects\Water Project\Final Project\water_potability.csv")
data.head()


# %%
columns_name = data.columns
for index, col_name in enumerate(columns_name):
    print(index, col_name)

# %%
# Function to detect outliers using Tukey's method
def detect_outliers_tukey(data, features):
    outliers = pd.DataFrame()
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        outliers = pd.concat([outliers, feature_outliers])
    return outliers

# List of features for outlier detection
features_for_outliers = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# Detect and remove outliers from the entire dataset
outliers = detect_outliers_tukey(data, features_for_outliers)
data_no_outliers = data[~data.index.isin(outliers.index)]

# Display information about the removed outliers
print(colored(f"Removed {len(outliers)} outliers from the dataset.", 'blue'))

# Data after removing outliers
data_no_outliers.head()


# %% [markdown]
# ### Data Informatics

# %%
data_no_outliers.info()

# %%
data_no_outliers.describe().loc[['min', '50%', 'mean', 'max', 'std']].T.style.background_gradient(axis=1)



# %%
columns_name = data_no_outliers.columns
for index, col_name in enumerate(columns_name):
    print(index, col_name)

# %%
data_no_outliers.isnull().sum()

# %%
data_no_outliers.head(10)

# %% [markdown]
# # Cleaning the dataset

# %% [markdown]
# ## Modifying columns which contain null values and Inserting mean in place of the null values

# %%
data_no_outliers['ph'].fillna(value=data_no_outliers['ph'].mean(), inplace=True)
data_no_outliers['Sulfate'].fillna(value=data_no_outliers['Sulfate'].mean(), inplace=True)
data_no_outliers['Trihalomethanes'].fillna(value=data_no_outliers['Trihalomethanes'].mean(), inplace=True)


# %%
data.isnull().sum()


# %%
X=data_no_outliers.iloc[:,0:9]
Y=data_no_outliers.iloc[:,-1]

# %%
X.shape , Y.shape

# %% [markdown]
# ## Splitting the dataset in training and testing data

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# %%
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
X = scaler.fit_transform(X)
X

# %% [markdown]
# ## 4. Random Forest Classifier

# %%
# Creating model object
model_rf = RandomForestClassifier(random_state=42)

# Training model object
model_rf.fit(X_train, y_train)


# %%
#Making Predictions
pred_rf = model_rf.predict(X_test)

#accuracy score
rf_accuracy = accuracy_score(y_test, pred_rf)
print(rf_accuracy)

# %%
#classifiction report
print(classification_report(y_test,pred_rf))

# confusion Maxtrix
cm4 = confusion_matrix(y_test, pred_rf)
sns.heatmap(cm4/np.sum(cm4), annot = True, fmt=  '0.2%', cmap = 'Reds')



# %% [markdown]
# The shows that 'Sulfates' Contribute most to the result followed by pph, hardness
# and we can work on it to make the model more accurate.

# %% [markdown]
# Now, Lets go ahead and make predictions to see how the model performs

# %% [markdown]
# ## Lets try it!

# %%
pickle.dump(model_rf,open("model.pkl","wb"))

model=pickle.load(open('model.pkl','rb'))






