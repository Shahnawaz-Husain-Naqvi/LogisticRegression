import dataProcess
import pickle
from sklearn.metrics import accuracy_score


path = "C:/credit card fraud/archive (7)/creditcardcsvpresent.csv"

x_train , y_train , x_test , y_test = dataProcess.get_data(path)

with open("saved/model.pkl" , 'rb') as f:
    model = pickle.load(f)

pred_test = model.predict(x_test)
pred_train = model.predict(x_train)

test_acc_score = accuracy_score(pred_test , y_test)
train_acc_score = accuracy_score(pred_train , y_train)
print(f"Test accuracy = {test_acc_score:.5f} \nTrain accuracy = {train_acc_score:.5f}")

