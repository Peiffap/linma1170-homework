import numpy as np


def ILU0(sA, iA, jA, remove_zeros=True):
    """
    Apply incomplete LU decomposition (ILU0) to a sparse matrix.
    
    The matrix is given in CSR format, and is square.
    
    Parameters
    ----------
    sA : ndarray
        Vector containing the nonzero entries of `A`.
    iA : ndarray
        Vector containing the indices of the first nonzero element
        on each row of A, as well as the expected number
        on the "(`n+1`)-th row" of `A`.
    jA : ndarray
        Vector containing the columns of the nonzero entries of `A`.
    remove_zeros : boolean, optional
        * True  : removes all zeros left on the band after fill-in (default).
        * False : the zeros are left in the sparse vectors, which is easier to manipulate in some cases.
        
    Returns
    -------
    sLU : ndarray
        Vector containing the nonzero entries
        of the in-place ILU0 decomposition of `A`.
    iLU : ndarray
        Vector containing the indices of the first nonzero element
        on each row of the in-place ILU0 decomposition of `A`,
        as well as the expected number
        on the "(`n+1`)-th row" of `A`.
    jLU : ndarray
        Vector containing the columns of the nonzero entries
        of the in-place ILU0 factorization of `A`.
    """
    n = len(iA) - 1

    left, right, bandwidth = compute_bandwidth(iA, jA)
    
    def get_index(i, j):
        """
        Returns the index in sLU of element `(i, j)` in `A`.
        
        Parameters
        ----------
        i : int
            Index of the row of the element.
        j : int
            Index of the column of the element.
                
        Returns
        -------
        ind : int
            Index of the element in sLU.
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
            
        return int(index - ng - nd)
    
    def get_element(i, j):
        """
        Returns element `(i, j)` in `A` in sLU.
        
        Parameters
        ----------
        i : int
            Index of the row of the element.
        j : int
            Index of the column of the element.
                
        Returns
        -------
        val : float
            Element in sLU.
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
            
        return sLU[int(index - ng - nd)]
    
    nleft = n - left
    nright = n - right
    size = int(n*n - (nleft*(nleft + 1) + (nright*(nright + 1)))/2)
    sLU = np.zeros(size, dtype = np.float64)
    iLU = np.zeros(len(iA), dtype = int)
    jLU = np.zeros(size)
    curr_index = 0
    
    for i in range(n):
        iright = i + right
        for j in range(iA[i], iA[i+1]):
            ind = get_index(i, jA[j])
            sLU[ind] = sA[j]
        iLU[i] = curr_index
        if i >= left and iright <= n:
            jLU[curr_index:curr_index + bandwidth] = range(i-left+1, iright)
            curr_index += bandwidth
        elif i <= left and iright <= n:
            jLU[curr_index:curr_index + iright] = range(0, iright)
            curr_index += iright
        elif i >= left and iright >= n:
            part = left + n - i - 1
            jLU[curr_index:curr_index + part] = range(n - part, n)
            curr_index += part
        else:
            jLU[curr_index:curr_index + n] = range(n)
            curr_index += n
    iLU[-1] = size
    
    
    # Incomplete LU decomposition
    for i in range(1, n):
        for l in range(iLU[i], get_index(i, i)):
            if sLU[l] != 0:
                k = jLU[l]
                sLU[l] /= get_element(k, k)
                for jj in range(l+1, iLU[i+1]):
                    if sLU[jj] != 0:
                        j = jLU[jj]
                        sLU[jj] -= sLU[l]*get_element(k, j)
    
    # Remove zeros from the precomputed band
    if remove_zeros:
        zeros = np.where(sLU == 0.0)[0]
        if len(zeros) > 0:                  
            count = 0
            line = 1
            for i in range(size - 1):
                if sLU[i] != 0:
                    count+=1
                if jLU[i] > jLU[i+1]:
                    iLU[line] = count
                    line += 1
            iLU[-1] = count+1
            sLU = np.delete(sLU, zeros)
            jLU = np.delete(jLU, zeros)
        
    return sLU.T, iLU.T, jLU.T


def compute_bandwidth(iA, jA):
    """
    Compute the bandwidth of a matrix in CSR format.
    
    Parameters
    ----------
    iA : ndarray
        Vector containing the indices of the first nonzero element
        on each row of A, as well as the expected number
        on the "(`n+1`)-th row" of `A`.
    jA : ndarray
        Vector containing the columns of the nonzero entries of `A`.
        
    Returns
    -------
    left : int
        Lower bandwidth of `A`.
    right : int
        Upper bandwidth of `A`.
    bandwidth : int
        Bandwidth of `A`.
    """
    n = len(iA) - 1
    right = 1
    left = 1
    
    # Find the upper and lower bandwidth by checking each column
    for i in range(n):
        curr_right = jA[iA[i+1] - 1] - i + 1
        if curr_right > right:
            right = curr_right
            
        curr_left = i - jA[iA[i]] + 1
        if curr_left > left:
            left = curr_left
        
    return left, right, left + right - 1