import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
import random


def SVD(A):
    """ Reference: https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm 
        U is eigen vectors of A.AT
        V is eigen vectors of AT.A
        sig is a diagonal matrix with eigen values of ATA or AAT along the diagonals
        
        all are sorted in descending order of the eigen values
    """
    
    # get eigen values and vectors of AAT
    eigenval,eigenvec=np.linalg.eig(np.dot(A,A.T))
    # make sure they are sorted in descending order
    sorted_idx = eigenval.argsort()[::-1] 
    eigenval = eigenval[sorted_idx]
    U = eigenvec[:,sorted_idx]

    # get the sigma Matrix component
    sig = np.zeros_like(A) # define a zero matrix of shape A
    
    sigval = (eigenval)**0.5 # get the diagonal matrix's values (no idea why i get negative values)
    diagMatrix = np.diag(sigval) # create diagonal matrix
    h,w = diagMatrix.shape[:2]
    sig[:h,:w] = diagMatrix # create the sigma matrix

    # get eigen values and vectors of ATA
    eigenval,eigenvec = np.linalg.eig(np.dot(A.T,A))
    #make sure they are sorted in descending order
    sorted_idx = eigenval.argsort()[::-1] 
    V = eigenvec[:,sorted_idx] 
    
    return U,sig,V.T


def videoRead(path = './Data/Ball_travel_10fps.mp4'):
    imgs = []
    cap = cv2.VideoCapture(path)
    
    while(True):
        ret, frame = cap.read()
        if ret:
            frame  = cv2.resize(frame, (800,640))
            imgs.append(frame)
        else:
            break
    cap.release()
    
    return imgs

def getLocations(imgs):
    """ 
    Binary mask [red = HIGH; white = LOW]  was obtained using thresholding and inversion. 
    pixel points of HIGH(red) region calculated by finding maxima 
    
    Returns,
        locations = Centroid pixel point of the ball in [x,y] in the space
        all_pts = all pixel positiona of the ball in [x,y]
    """
    
    locations = []
    startFlag = 1
    all_pts = []
    for im in imgs:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # get grayscale version
        ret,thresh1 = cv2.threshold(im,200,255,cv2.THRESH_BINARY_INV) # get binary mask that red ball is HIGH
        coordinates = peak_local_max(thresh1, min_distance=20) # find the location of peak regions ie HIGH red ball regions
        if startFlag == 1:
            all_pts = coordinates
            startFlag= 0
        else:    
            all_pts = np.vstack((all_pts,coordinates))
        
        locations.append(np.mean(coordinates, axis=0).astype(np.uint16))  # get mean location
    return np.flip(np.array(locations),-1), np.flip(all_pts,-1) # return x,y 

def LeastSquares(x,Y):
    """
    Solves the least squares condition: B = XT_X_inv.dot(XT_Y)
    
    Returns,
        B =  the coefficients of the parabolic trajectory
    """
    
    X = np.squeeze(np.dstack((x**2, x, np.ones(x.shape))))
    XT_X = (X.T).dot(X)
    XT_X_inv = np.linalg.inv(XT_X) 
    XT_Y = (X.T).dot(Y)
    
    B = XT_X_inv.dot(XT_Y)
    
    return B

def getModel(x,B):
    y_ = B[0]*(x**2) + B[1]*x + B[2]
    return y_

def getTLSModel(x,B,d):
    y_ = 1/B[2] * (d - B[0]*(x**2) - B[1]*x)  
    return y_

def TotalLeastSquares(x,Y): 
    """"
    Fits a parabolic polynomial using Total least Squares with formula ax**2 + bx + cy = d
    
    Returns,
        B =  the coefficients of the parabolic trajectory
    """
   
    U = np.squeeze(np.dstack((x**2, x, Y)))
    D = U.mean(axis=0) # d was found to be the constant term
    U = U - D 
    UtU = (U.T).dot(U)
    _,s,V = SVD(UtU)
    
    return V[2,:]


# RANSAC  using Least Squares.
def Ransac(x,Y_true, s = 3, thresh = 20, n_iterations = 50):
    """
    s = no. of data points sampled per iteration
    thresh = threshold for considering whether a value is inlier or outlier 
            *chosen based on visual inspection in Least Squares Method
    n_iterations =   number of iterations to run the algoorithm 
    """
    predictions = []
    
    max_inliers = 0 
    B_best = np.zeros([3])
    print("Number of Iterations: ", n_iterations )
    for i in range(n_iterations):
        # choose random pair of 3 data points
        random_idxs = random.sample(range(0,len(x)), s) 
        x_r,Y_r =  x[random_idxs], Y_true[random_idxs]


        # estimate polynomial using least Squares.
        B = LeastSquares(x_r,Y_r) 
        # predict the curve using the polynomial
        y_pred = getModel(x,B)
        
        predictions.append(y_pred)

        Error = abs(Y_true - y_pred)  

        Error[Error <= thresh] = 1
        Error[Error>thresh] = 0
        inliers = int(Error.sum())

        if inliers >= max_inliers:
            max_inliers = inliers
            B_best = B
    
    print("Max number of inliers: " + str(max_inliers)+'/'+ str(len(Y_true)))
    return B_best, predictions
