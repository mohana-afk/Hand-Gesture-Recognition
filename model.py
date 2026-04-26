import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle



# read data with help of pandas
data = pd.read_csv("landmarks.csv",header = None)

# data split 

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

#model 
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size= 0.2)

model = RandomForestClassifier(n_estimators=100)

model.fit(x_train,y_train)

predictions =  model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as gesture_model.pkl")

"""
LARGER THE DATA EASIER THE ANSWER

"""