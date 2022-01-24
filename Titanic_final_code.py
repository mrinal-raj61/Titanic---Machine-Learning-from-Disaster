

#------------------------------------------------------------------------------
#Titanic - Machine Learning from Disaster
#------------------------------------------------------------------------------

#------------------------------------------
# Importing and Preprocessing the data
#------------------------------------------
import pandas as pd
data = pd.read_csv('C:/Users/Mrinal Raj/soting alogorithm/train.csv')


#checking for null values
data.isnull().sum()


#removing cabin column because it has 687 null values out of 891.
data_1 = data.drop(['Cabin'], axis = 1)


#Age has 177 nan values and replacing it with mean values
data_1['Age']=data_1['Age'].fillna(int(data_1['Age'].mean()))

#dropping 2 rows wich have nan values in 'Embarked' 
data_1 = data_1.dropna()

#finally checking null values
data_1.isnull().sum()
# no null values :)

#Dropping unnecessary columns
data_1 = data_1.drop(['PassengerId','Name','Ticket'],axis =1)

#Plotting & Nomalising 'Age' and 'Fare' column

import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.hist(data_1['Age'])
plt.subplot(2,2,2)
plt.hist(data_1['Fare'])

#'Face' is not normal distribution





#checking data types
data_1.dtypes
#changing Sex and Pclass to category
cols = ['Survived', 'Pclass', 'Sex', 'SibSp','Parch','Embarked']
data_1[cols] = data_1[cols].astype('category')
data_1.dtypes

#creating dummy variables

data_2 = pd.get_dummies(data_1 , drop_first= True)

#creating X and Y train

X_train = data_2.drop(['Survived_1'], axis = 1)

Y_train = data_2.iloc[:,2]


#Importing and Processing test data
Test_data = pd.read_csv('C:/Users/Mrinal Raj/soting alogorithm/test.csv')
Test_data.isnull().sum() #cabin has 327 & Age has 86 & fare has 1
Tdata = Test_data.drop(['Cabin'], axis = 1)
Tdata['Age']=Tdata['Age'].fillna(int(Tdata['Age'].mean()))
Tdata['Fare']=Tdata['Fare'].fillna(int(Tdata['Fare'].mean()))
Tdata.isnull().sum()
#No null values
Tdata = Tdata.drop(['PassengerId','Name','Ticket'],axis =1)
colst = [ 'Pclass', 'Sex', 'SibSp','Parch','Embarked']
Tdata[colst] = Tdata[colst].astype('category')
Tdata.dtypes
Tdatad = pd.get_dummies(Tdata , drop_first= True)
Tdatad = Tdatad.drop(['Parch_9'], axis = 1)
X_test = Tdatad

#------------------------------------------------------------------------------
#Model Training and Making Prediction(Random Forest Classifier is used.)
#------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
rfc =  RandomForestClassifier()
rfc.fit(X_train, Y_train)

Y_predict = rfc.predict(X_test)
Y_predict = pd.Series(Y_predict)

print(Y_predict)

sub_id = Test_data['PassengerId']


datasub =  {'PassengerID' : sub_id, 'Survived' : Y_predict}
sub_file = pd.concat(datasub,axis =1)
sub_file.to_csv(r'C:/Users/Mrinal Raj/Downloads/SubmissionFile.csv' , index = False)
