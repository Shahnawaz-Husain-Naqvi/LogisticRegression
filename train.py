import dataProcess
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
warnings.filterwarnings("ignore")


path = "C:/credit card fraud/archive (7)/creditcardcsvpresent.csv"

x_train , y_train , _ ,_  = dataProcess.get_data(path)

model = LogisticRegression()

model.fit(x_train, y_train)

with open ("saved/model.pkl" , 'wb') as f:
    pickle.dump(model , f)
    print("Model Saved")