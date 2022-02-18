import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
import random
from helper_functions import *
import argparse
import warnings
warnings.filterwarnings("ignore")

def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--path', default='./Data/Ball_travel_10fps.mp4', help='Video path , Default:./Data/Ball_travel_10fps.mp4')

    Args = Parser.parse_args()
    path = Args.path
    
    
    print(" ########################################## PROBLEM 2 - Compute Ball Trajectory ##########################################")
    imgs = videoRead(path)
    
    # get centroid positions of the ball along all 28 frames
    locations, _ = getLocations(imgs)
    x = locations[:,0].astype(np.float64)
    Y = np.array(locations[:,1]).astype(np.float64)
    print("Obtained X and Y ")
    
    ########################################## Fit  Least Squares ########################################
    print("Performing Least Squares Fit....")

    B = LeastSquares(x,Y)
    x_ = np.linspace(0,800,800)    
    y_pred = getModel(x_,B) 

    fig = plt.figure(figsize=(10,4))
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.ylim([640, 0])
    plt.scatter(x,Y,c='red')
    plt.text(300,500,'LeastSquares Fit')
    plt.plot(y_pred,'k')
    plt.savefig('./Results/LeastSquaresFit.png')
    plt.show()
    
    print("Least Squares Output saved in ./Results Folder ")
    print("------------------------------------------------")
    #################################### Fit Total Least Squares ########################################
    print("Performing Total Least Squares Fit....")
    
    B = TotalLeastSquares(x,Y)
    d = (B[0])*np.mean(x**2) + (B[1])*np.mean(x) + (B[2])*np.mean(Y)

    x_ = np.linspace(0,800,800)
    y_pred = getTLSModel(x_,B,d)

    fig = plt.figure(figsize=(10,4))
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.ylim([640, 0])
    plt.scatter(x,Y,c='red') # true data
    plt.text(300,500,'Total LeastSquares Fit')
    plt.plot(y_pred,'b') # predictions
    plt.savefig('./Results/TotalLeastSquaresFit.png')
    plt.show()

    print("Total Least Squares Output saved in ./Results Folder ")
    print("------------------------------------------------")
    ######################################### Fit RANSAC ####################################################
    print("Performing RANSAC Fit....")
    
    B,  predictions = Ransac(x,Y)

    x_ = np.linspace(0,800,800)
    y_pred = getModel(x_,B)

    fig = plt.figure(figsize=(10,4))
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.ylim([640, 0])
    plt.scatter(x,Y,c='red')
    plt.text(350,500,'RANSAC')
    plt.plot(y_pred,'k')
    plt.savefig('./Results/RansacFit.png')
    plt.show()
    
    print("RANSAC Output saved in ./Results Folder ")
    print("------------------------------------------------")
    
    print(" ########################################## PROBLEM 3 - Compute Homography ##########################################")


    X = np.array([5,150,150,5])
    Y = np.array([5,5,150,150])

    Xp = np.array([100,200,220,100])
    Yp = np.array([100,80,80,200])

    """ To compute Homography matrix:
        1) Compute the A matrix with 8x9 dimensions. Need to solve for Ax = 0 
        2) perform Singular Value decomposition on A matrix to find the solution x 
        3) reshape x into a 3x3 homography matrix
    """
    ############################ Generate A matrix ############################
    startFlag=1

    for (x,y,xp,yp) in zip(X,Y,Xp,Yp):

        if (startFlag == 1) :
            A = np.array([[-x,-y,-1,0,0,0, x*xp, y*xp,xp], [0,0,0,-x,-y,-1, x*yp, y*yp, yp]])
        else:
            tmp = np.array([[-x,-y,-1,0,0,0, x*xp, y*xp,xp], [0,0,0,-x,-y,-1, x*yp, y*yp, yp]])
            A = np.vstack((A, tmp))

        startFlag+=1    



    print("The A matrix to be solved:")
    print(A)    

    U,S,Vt = SVD(A.astype(np.float32))
    

    H = Vt[8,:]/Vt[8][8]
    H = H.reshape(3,3)
    print("\n")
    print("The estimated Homography matrix is:  \n")
    print(H)
    print("\n")

    H1 = cv2.getPerspectiveTransform(np.dstack((X,Y)).astype(np.float32), np.dstack((Xp,Yp)).astype(np.float32))
    print("The estimated Homography matrix using opencv implementation:  \n")
    print(H1)
    
    
if __name__ == '__main__':
    main()