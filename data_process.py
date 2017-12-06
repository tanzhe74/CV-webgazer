import numpy as np
import os
import csv
import cv2
from regression import regression

class data_process:
    def __init__(self, filename, eye_size, regression, size):
        self.reg = regression 
        #true using lasso regresion 
        #false using cnn
        
        self.labels = []
        #store the true labels for 1 image 4 numbers
        
        self.data = []
        #training data 
        #for regression the dimension is 5000
        #for cnn the dimension is 100 * 50 * 3 * len
        if not self.reg:
            self.cnn = np.zeros((2*eye_size, eye_size, 3, size), dtype=np.float )

        self.get_data(filename)
        print(self.get_len())

    def get_per_data(self, frameFilename):
        #print(frameFilename)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        img = cv2.imread(frameFilename )   

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eye_person = []
        if len(faces) == 0: return eye_person
        
        x,y,w,h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray,1.3,10)
       
        if len(eyes) != 2: return eye_person
        
        for (ex,ey,ew,eh) in eyes:
            
            if self.reg:
                eye = roi_gray[ex:ex+ew, ey:ey+eh]
                resize = cv2.resize(eye, (50,50))
                l = [item for sublist in resize for item in sublist]
                eye_person.extend(l)
            else:
                eye = roi_color[ex:ex+ew, ey:ey+eh]
                resize = cv2.resize(eye, (50,50))
                eye_person.extend(resize)
        
        if self.reg:
            return eye_person
        else:

            #return np.reshape(eye_person, [eye_size*2, eye_size, 3])
            return np.array(eye_person)


    def get_data(self, filename):
        f = open(filename)
        lines = f.readlines()

        dataFile = "/gazePredictions.csv"
        count = 0
        for line in lines:
        
            print (line)
            try:
                with open(line.strip().replace("\\", "/") + dataFile) as f:
                    readCSV = csv.reader(f, delimiter=',')
                    for row in readCSV:
                        
                        frameFilename = row[0]
                        
                        eye_feature = self.get_per_data(frameFilename)
                        
                        if len(eye_feature) == 0:
                            continue
                        if self.reg:
                            self.data.append(eye_feature)
                        else:
                            self.cnn[:,:,:,count] = eye_feature
                            count += 1

                        label = []
                        label.append(float(row[2]))
                        label.append(float(row[3]))
                        label.append(float(row[4]))
                        label.append(float(row[5]))
                        self.labels.append(label)
            except:
                pass

    def get_len(self):
        
        return len(self.labels)

       
if __name__ == '__main__':

    dp_train = data_process("train_1430_1.txt", 50, False, 44081)
    np.save('train_cnn_x.npy', dp_train.cnn)
    # np.save('train_cnn_y.npy', dp_train.labels)

    dp_test = data_process("test_1430_1.txt", 50, False, 13451)
    np.save('test_cnn_x.npy', dp_test.cnn)
    # np.save('test_cnn_y.npy', dp_test.labels)
    # dp_test = data_process("test_1430_1.txt", 50, True)
  
