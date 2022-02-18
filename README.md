for dependancies , 

`pip3 install -r requirements.txt `



To test run for  Problem 2 and 3 :
    Run 

```python3 main.py --path <path_for_video> ```

Check `helper_functions.py` for the code written for all functions for this assignment. It includes code to perform the following.

#  Linear Least Squares

Coefficients/parameters of the polynomial is obtained by solving the analytical solution of Squared Error cost function which is given as  $ E =  || y - Bx ||^{2} $

The optimization is given as $ argmin_{B} E$. Since derivative of the cost function at extrema is equal to zero, we can differentiate E with respect to B and set $\frac{dE}{dB} = 0 $, to solve for B. This analytical solution is computed as $B =  (X^{T}X)^{-1}X^{T}Y $

# Total Least Squares:

For Total Least Squares, we need to find a line $ax_i + by_i = d$ that minimizes the cost function $E = \sum (ax_i + by_i -d)^2$ 

This is equivalent to minimizing the cost function: 
	``` $ E = \sum_{i=1}^{n} (a(x_i - \bar{x})  + b(y_i - \bar{y})  )^2 $``` 
 which is rewritten in matrix form as 
	``` $E =(UN)^T (UN)$ ```

where ``` $U = \begin{bmatrix}
x_1 - \bar{x} & y_1 - \bar{y} \\
 . &  .\\
 . &  .\\
x_n - \bar{x} & y_1 - \bar{y} \\
\end{bmatrix} , N = \begin{bmatrix}
a \\ b
\end{bmatrix} ; \bar{x} = \frac{1}{n} \sum_{i=0}^n x_i, \bar{y} = \frac{1}{n} \sum_{i=0}^n y_i  $```

We can find the analytical solution of the cost function by setting the derivative = 0, which results in this relationship $ (U^TU)N=0$.

N are the parameters of the model found by solving th homogenous system of equations. . We can solve these system of equations using Singular Value Decomposition. 


# RANSAC

- We use the least squares approach to solve for the coefficients.
- We also obtained slightly better improvements using RANSAC.

The algorithm for RANSAC is given below
![ransac](''./Data/ransac.png')

# Singular Value Decomposition

This repo also contains code to compute singular value decomposition of a given $m \times n$ matrix.





