import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Dataloader:
    """
    Class : Dataloader
    
    input : path_to_data = (images.csv) and path_to_image = (handwritten_digit_corpus folder)
    
    Description : This class shall obtain the data as input and transform 
                these data in the training format.
    """
    def __init__(self,path_to_data,path_to_image,logging_obj):
        # reading input files
        self.obj = logging_obj
        self.obj.info("Data Loading and manipulations starts initially ")
        self.path_to_data = path_to_data
        self.path_to_image = path_to_image
        # pandas dataframe for train.csv files
        self.df = pd.read_csv(os.path.join(self.path_to_data,"images.csv"))
        self.df.columns = ["images", "target"]
        # initializing additional attributes
        self.images = []
        self.labels = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def imageData(self):
        """
        Method Name : imageData
        Description : This method reads all the 'images' column of 
                      dataframe(df) and convert each image to (28,28)
                      size and append it on the images list.
        """
        try:
            # reading all the images from dataframe 
            self.obj.info("Started Reading the 'images' columns of dataframe")


            for image_name, label in zip(self.df['images'], self.df["target"]):
                image = os.path.join(self.path_to_data, str(image_name))
                # reading images using opencv
                image = cv2.imread(image)
                # resizing the image as height 28 and width 28
                image_resized = cv2.resize(image,(28,28))
                # appending image to the images list
                self.images.append(np.array(image_resized,"float32"))
            return self.images
        
        except Exception as e:
            self.obj.error(f"Error occurs :{e}")
        self.obj.info("Successfully completed reading 'images' columns and all images are appended to list ")
    
    def getData(self,images):
        """
        Method Name : getData
        Input : Images  list from the 'imageData' method
        Description :  It then splits the images and
                       target column using train_test_split from sklearn with 
                       the test_size of 0.2. It then convert all of the four 
                       data to numpy array and returns all of them.
        """
        try:
            self.images = images
            # target column
            labels = self.df['target'].values
            # splitting the data
            self.obj.info("Start Splitting train and test data")
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(
                        self.images,labels,test_size=0.2,random_state=42
            )
            self.obj.info("Successfully Completed Splitting train and test data")
            # convert to numpy array
            self.X_train = np.array(self.X_train)/255.0
            self.X_test = np.array(self.X_test)/255.0
            self.y_train = np.array(self.y_train)
            self.y_test = np.array(self.y_test)
            # reshape the X_train and X_test data


            nsample, nx, ny, _ = self.X_train.shape
            self.X_train = self.X_train.reshape((nsample, nx*ny*3))

            ntsample, ntx, nty, _t = self.X_test.shape
            self.X_test = self.X_test.reshape((ntsample, ntx*nty*3))
            
            # returning the four values
            return self.X_train,self.X_test,self.y_train,self.y_test
        
        except Exception as e:
            self.obj.error(f"Error occusrs {e}")

        self.obj.info("Successfully Completed Loading and manipulating all data")
