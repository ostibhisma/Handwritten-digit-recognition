import os
import csv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Model:
    """
    This Class Train the model and save the best model according to validation
    aaccuracy in the form of .pkl file
    Written By : Bhisma
    
    """
    def __init__(self,X_train,X_test,y_train,y_test,logging_obj):
        self.logging_obj = logging_obj
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def ml_model(self):
        self.logging_obj.info("Initially Started Creating Model")
        """
        Method Name: ml_model
        Description : This method describe the Support Vector machine classifier 
                       Because It has given better result than other in my test
        """

        try:
            self.svm_model = svm.SVC(C = 1.5)
            return self.svm_model
           
	

            self.logging_obj.info("Successfully Completed Creating Model")

        except Exception as e:
            self.logging_obj.error(f"Error Occurs as : {e}")
            
            
    def training(self, model):
        self.model = model
        self.logging_obj.info("Started Training the model")
        """
        Method Name: training
        Description : Now the model that is pass from above method started to train
                      and the accuracy and classification report is written in model.txt file
                      dynamically
        """
        try:
            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)
           
            acc = accuracy_score(self.y_test, y_pred)
            confusion_mat = confusion_matrix(self.y_test, y_pred)
            clasify_report = classification_report(self.y_test, y_pred)
            
            f = open(os.path.join("Report",'model_report.txt'), 'w')
            f.write(f"\n Accuracy = {str(acc)} \n")
            f.write(f"\n Confusion Matrix = \n{str(confusion_mat)}\n")
            f.write(f"\n Classification Report = \n{str(clasify_report)}")
            f.close()
            self.logging_obj.info("Successfully Created text file")


            # Save the model as a pickle in a file
            joblib.dump(self.model, os.path.join("Models","svm_model.pkl"))
            
            self.logging_obj.info("Successfully Completed Training the Model")
            

        except Exception as e:
            self.logging_obj.error(f"Error Occurs as : {e}")

