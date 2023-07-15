# CVIP
import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
df=pd.read_csv("C:\\Users\\sai\\Downloads\\diabetes_prediction_dataset.csv")
df

df.shape

df.dtypes

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder

en=LabelEncoder()
df["gender"]=en.fit_transform(df["gender"])
df["smoking_history"]=en.fit_transform(df["smoking_history"])
df.dtypes


df

df["diabetes"].value_counts()

X=df.iloc[:,:8]
X.shape

Y=df.iloc[:,8]
Y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.40,random_state=5000)
x_train.shape

x_test.shape

y_train.shape

y_test.shape

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)

y_predicted_values=model.predict(x_test)
y_predicted_values

y_predicted_values.shape

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print("confusion matrix\n",confusion_matrix(y_test,y_predicted_values))
print("classification report\n",classification_report(y_test,y_predicted_values))
print("accuracy score:",accuracy_score(y_test,y_predicted_values))

mlt.plot(y_test,color="red")
mlt.plot(y_predicted_values,color="blue")
mlt.show()
