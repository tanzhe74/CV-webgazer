#
# CSCI 1430 Webgazer Project
# james_tompkin@brown.edu
#

import os
import csv
import cv2
import math
class computeDistance:
#class to compute the original web gazer prediction accuracy
#using the distance of the web gazer and tobii as the score
    
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
            
         
            eye = roi_gray[ex:ex+ew, ey:ey+eh]
            resize = cv2.resize(eye, (50,50))
            l = [item for sublist in resize for item in sublist]
            eye_person.extend(l)
            
        return eye_person
       
    def distance(self, pre_x, pre_y, label_x, label_y):    
        dist = math.sqrt((label_x-pre_x)**2 + (label_y-pre_y)**2)
        return dist


    def res(self):
        filename = "test_1430_1.txt"
        f = open(filename)
        lines = f.readlines()
        dataFile = "/gazePredictions.csv"
        dis_sum = 0.0
        count = 0
        for line in lines:
            print(line)
            try:
                with open(line.strip().replace("\\", "/") + dataFile) as f:
                    readCSV = csv.reader(f, delimiter=',')
                    for row in readCSV:

                        frameFilename = row[0]
                        eye_feature = self.get_per_data(frameFilename)
                        if len(eye_feature) == 0:
                            continue

                        tobiiLeftEyeGazeX = float( row[2] )
                        tobiiLeftEyeGazeY = float( row[3] )
                        tobiiRightEyeGazeX = float( row[4] )
                        tobiiRightEyeGazeY = float( row[5] )
                        webgazerX = float( row[6] )
                        webgazerY = float( row[7] )
                        tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
                        tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2
                        dis_sum += self.distance(webgazerX, webgazerY, tobiiEyeGazeX, tobiiEyeGazeY)
                        count += 1
            except:
                pass
        print(count)
        return dis_sum / count


if __name__ == '__main__':

    compute = computeDistance()
    print(compute.res())
   
    # count:            24838
    # distance_average: 0.2854903746016776

    # count:            13451
    # distance_average: 0.293238937006042
