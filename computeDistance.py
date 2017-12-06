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
            try:
                with open(line.strip().replace("\\", "/") + dataFile) as f:
                    readCSV = csv.reader(f, delimiter=',')
                    for row in readCSV:

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
