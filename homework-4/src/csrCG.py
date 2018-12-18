import numpy as np
from ILU0 import ILU0


def csrCG(sA, iA, jA, b, rtol, prec):
    n = len(b)
    rtol /= 10  # To make sure we don't exceed the tolerance
    x = np.zeros(n, dtype = np.float64)
    res = np.copy(b)
    itercount = 0
    if not prec:
        p = np.copy(res)
        while np.linalg.norm(res)/np.linalg.norm(b) > rtol:
            Ap = product_CSR(sA, iA, jA, p)
            dot_res = np.dot(res, res)
            alpha = dot_res / np.dot(p, Ap)
            x += alpha*p
            res -= alpha*Ap
            beta = np.dot(res, res) / dot_res
            p = res + beta*p
            itercount += 1
    else:
        sM, iM, jM = ILU0(sA, iA, jA, remove_zeros=False)
        res_tilde = ILUcsrsolve(sM, iM, jM, res)
        p = np.copy(res_tilde)
        while np.linalg.norm(res)/np.linalg.norm(b) > rtol:
            dot_res = np.dot(res, res_tilde)
            Ap = product_CSR(sA, iA, jA, p)
            alpha = dot_res / np.dot(p, Ap)
            x += alpha*p
            res -= alpha*Ap
            res_tilde = ILUcsrsolve(sM, iM, jM, res)
            beta = np.dot(res, res_tilde) / dot_res
            p = res_tilde + beta*p
            itercount += 1
            
    return x, res

        
def product_CSR(sA, iA, jA, p):
    """
    Compute `Ap`, the product of a matrix in CSR format and a vector.
    
    Parameters
    ----------
    sA : ndarray
        Vector containing the nonzero entries of `A`.
    iA : ndarray
        Vector containing the indices of the first nonzero element
        on each row of `A`, as well as the expected number
        on the "(n+1)-th row" of `A`.
    jA : ndarray
        Vector containing the columns of the nonzero entries of `A`.
    p : ndarray
        Vector.
        
    Returns
    -------
    product : ndarray
        Vector containing the product of `A` and `p`.
    """
    n = len(iA) - 1
    product = np.zeros(n, dtype=np.float64)
    for i in range(n):
        product[i] = np.dot(sA[iA[i]:iA[i+1]], p[jA[iA[i]:iA[i+1]]])
    return product


def ILUcsrsolve(sA, iA, jA, b):
    """
    Solves the linear system ``Ax = b`` where `A` is given in CSR format.
    
    The system is solved by successive forward and back substitution.
    
    Parameters
    ----------
    sA : ndarray
        Vector containing the nonzero entries of `A`.
    iA : ndarray
        Vector containing the indices of the first nonzero element
        on each row of `A`, as well as the expected number
        on the "(n+1)-th row" of `A`.
    jA : ndarray
        Vector containing the columns of the nonzero entries of `A`.
    b : ndarray
        Vector of the linear system.
    Returns
    -------
    x : ndarray
        Solution to the matrix equation.
    """
    n = len(iA) - 1  # Sparse decomposition, while keeping zero elements on the band
    
    right = iA[1] - iA[0]
    left = iA[-1] - iA[-2]
    bandwidth = left + right - 1
    
    def get_element(i, j):
        """
        Returns the value element `(i, j)` in `A` by looking in sA.
        
        Parameters
        ----------
        i : int
            Index of the row of the element.
        j : int
            Index of the column of the element.
                
        Returns
        -------
        elem : int
            Element in sA.
        """
        iright = i + right
        if j >= iright or j <= i - left:
            return 0
        
        index = i*bandwidth + (j-i)
        left2 = left-2
        ng = left2*(left2+1)/2
        left2i = left2 - i
        if left2i > 0:
            ng -= (left2i)*(left2i+1)/2
            
        nd = 0
        if iright > n:
            delta = iright - n
            nd = delta*(delta-1)/2
            
        return sA[int(index - ng - nd)]
    
    x = np.zeros(n, dtype = np.float64)
    y = np.zeros(n, dtype = np.float64)

    # Forward substitution to solve Ly = b for y
    y[0] = b[0]
    for i in range(1, n):
        s = 0
        for k in range(max(0, i - left + 1), i):
            s += get_element(i, k) * y[k]
        y[i] = b[i] - s
        
    # Back substitution to solve Ux = y for x
    x[-1] = y[-1] / get_element(n-1, n-1)
    for i in range(n-2, -1, -1):
        s = 0
        for k in range(i+1, min(i + right, n)):
            s += get_element(i, k) * x[k]
        x[i] = (y[i] - s) / get_element(i, i)
    return x