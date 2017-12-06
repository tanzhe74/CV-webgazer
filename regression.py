from sklearn import linear_model
import numpy as np 
import math
class regression:
    def __init__(self, train_X, train_y, test_X, test_y, type_of_reg):

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
 		
        if type_of_reg == "lasso":
        	self.reg = linear_model.Lasso(alpha = 0.1)
        elif type_of_reg == "lars":
        	self.reg = linear_model.LassoLars(alpha=0.01)
        elif type_of_reg == "net":
        	self.reg = linear_model.ElasticNet(random_state = 0)
        elif type_of_reg == "ridge":
        	self.reg = linear_model.Ridge()
        
        self.train()


    def train(self):
        self.reg.fit(self.train_X, self.train_y)

    def predict(self, test_x):
        return self.reg.predict(test_x)[0]

    def distance(self, predict, label):
        predict_x = (predict[0] + predict[2]) / 2
        predict_y = (predict[1] + predict[3]) / 2
        label_x = (label[0] + label[2]) / 2
        label_y = (label[1] + label[3]) / 2
        dist = math.sqrt((label_x-predict_x)**2 + (label_y-predict_y)**2)
        return dist

    def dis_avg(self):
        dis_sum = 0.0
        for i in range(len(self.test_X)):
            pre = self.predict(self.test_X[i])
            dis = self.distance(pre, self.test_y[i])
            dis_sum += dis
        return dis_sum / len(self.test_X)

if __name__ == '__main__':

    train_X = np.load('train_reg_x.npy') 
    train_y = np.load('train_reg_y.npy')
    test_X = np.load('test_reg_x.npy')  
    test_y = np.load('test_reg_y.npy')
   
    regression = regression(train_X, train_y, test_X, test_y, "lasso") #0.16694475862513625
    # regression = regression(train_X, train_y, test_X, test_y, "lars") #0.16920453630748747
    # regression = regression(train_X, train_y, test_X, test_y, "net") #0.16900103092369337
    # regression = regression(train_X, train_y, test_X, test_y, "ridge") #0.18331017007622946
    
    print(regression.dis_avg())

	#train size: 44081
	#test size : 13451

	#method 	dis_avg
	#WebGazer   0.2854903746016776
	#Lasso 		0.16694475862513625
	#LassoLars  0.16920453630748747
	#ElasticNet 0.16900103092369337
	#Ridge 		0.18331017007622946
