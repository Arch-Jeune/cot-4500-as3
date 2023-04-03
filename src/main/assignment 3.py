import numpy as np 
np.set_printoptions(precision=7, suppress=True, linewidth=100)

#Question 1 
def function(t,y):
    return t-y**2
def euler(initial_point,a,b,n):
    h=(b-a)/n
    t,y=a,initial_point
    
    for i in range(1,n+1):
        y+=h*function(t,y)
        t+=h 
    return y 
initial_point=1
a,b=0,2
n=10
print(euler(initial_point,a,b,n))
print()

#Question 2 
def runge_kutta(initial_point,a,b,n):
    h=(b-a)/n
    t,y=a,initial_point
    for i in range(n):
        k1=h*function(t,y)
        k2=h*function(t+h/2, y+k1/2)
        k3=h*function(t+h/2, y+k2/2)
        k4=h*function(t+h,y+k3)
        y+=(k1+2*k2+2*k3+k4)/6
        t+=h 
    return y 
print(runge_kutta(initial_point,a,b,n))
print()

#Question 3 
A = np.array([[2,-1,1,6], [1,3,1,0], [-1,5,4,-3]])

n = A.shape[0]

# Gaussian elimination with pivoting
for i in range(n):
    max_index = np.argmax(abs(A[i:,i])) + i 
    A[[i,max_index]] = A[[max_index,i]]
    for j in range(i+1,n):
        factor = A[j,i]/A[i,i]
        A[j,i:] = A[j,i:]-factor*A[i,i:]

# Backward substitution
x = np.zeros(n)
for i in range(n-1,-1,-1):
    x[i] = (A[i,n] - np.dot(A[i,i+1:n],x[i+1:n]))/A[i,i]

print(x)
print()

#Question 4
A =np.array ([[1, 1, 0, 3],[2, 1, -1, 1],[3, -1, -1, 2],
			  [-1, 2, 3, -1]]) 
			  
def lu_factorization(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for j in range(n):
        U[0, j] = A[0, j]
        L[j, 0] = A[j, 0] / U[0, 0]

    for i in range(1, n):
        for j in range(i, n):
            s1 = sum(U[k, j] * L[i, k] for k in range(i))
            U[i, j] = A[i, j] - s1

            s2 = sum(U[k, i] * L[j, k] for k in range(i))
            L[j, i] = (A[j, i] - s2) / U[i, i]
    return L,U

L,U = lu_factorization(A)
determinant = round (np.linalg.det (A))
print (f"Matrix determinant: {determinant}")
print ()
print (f"L matrix:\n{L}")
print ()
print (f"U matrix:\n{U}")
print ()

#Question 5
def dia_dominant (A):
    for i in range (len (A)): 
        dia = abs (A[i][i])
        non_dia = sum (abs (A[i][j]) for j in range (len (A)) if j != i)
        if dia<=non_dia:
            return False
    return True
A =[[9, 0, 5, 2, 1],
	[3, 9, 1, 2, 1],
	[0, 1, 7, 2, 3],
	[4, 2, 3, 12, 2],
	[3, 2, 4, 0, 8]]
print (dia_dominant (A))
print ()

#Question 6
def pos_definite (A):
  eigenvalues = np.linalg.eigvals (A)
  return all (eigenvalues > 0)
A =[[2, 2, 1],[2, 3, 0],[1, 0, 2]] 
print (pos_definite (A))


