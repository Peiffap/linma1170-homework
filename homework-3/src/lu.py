# The function mysolve(A, b) is invoked by ccore.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ccore.py -clscale float
# where the line argument 'clscale' allows the global coarsening of the mesh
# -clscale 1  fine mesh
# -clscale 10 coarse mesh


# Homework 3 : Direct solvers
# @author Gilles Peiffer, 24321600
# gilles.peiffer@student.uclouvain.be
# Dec. 10, 2018

# Mathematical imports
import numpy as np
from collections import deque  # Needed for deque used in RCMK

# SciPy is only used for tests and plots
import scipy.sparse
import scipy.sparse.linalg

# ccore imports
import sys
import gmsh

# Imports for plots and timing
import matplotlib.pyplot as plt
import timeit


SolverType = 'scipy'


def mysolve(A, b):
    """
    Solve the matrix equation ``Ax = b`` for `x`.
    
    According to which solver is selected by the global variable "SolverType",
    this solver uses either:
        
        * scipy.sparse.linalg's built-in sparse solver;
        * a solver based on QR factorization;
        * a solver based on LU factorization;
        * a solver based on the GMRES algorithm.
    
    Parameters
    ----------
    A : ndarray
        Square (`n x n`) coefficient matrix for the system ``Ax = b``.
    b : ndarray
        `n`-element vector for the system.
            
    Returns
    -------
    success : boolean
        Indicates whether the solver was succesful in its computation.
    x : ndarray
        Exact solution of the matrix equation.
    """
    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b)
    #elif SolverType == 'QR':
        # write here your code for the QR solver
    elif SolverType == 'LU':
        return True, LUsolve(A, b)
    #elif SolverType == 'GMRES':
        # write here your code for the LU solver
    else:
        return False, 0
    
    
# Dense matrices
    

def LUfactorize(A):
    """
    Factorize a square matrix in-place using LU decomposition.
    
    Does not take sparsity into account, but gains significant speedups
    due to vectorisation and use of highly optimized NumPy routines.
    
    While these properties are not used and are thus not necessary,
    the problem statement also specifies that the matrix has a low density
    and is hermitian.
    Indirectly, the matrix is the result of a finite element program,
    and is thus also positive definite.
    
    Parameters
    ----------
    A : ndarray
        Square matrix to decompose in-place.
            
    Returns
    -------
    None
    """
    lA = len(A)
    for k in range (0, lA-1):
        A[k+1:, k] = A[k+1:, k] / A[k, k]
        A[k+1:, k+1:] -= np.outer(A[k+1:, k],A[k, k+1:])

def LUsolve(A, b):
    """
    Solve the matrix equation ``Ax = b`` for `x`.
    
    The algorithm used for this is straightforward;
    the matrix `A` is first factorized in-place,
    then forward and back substitution are used
    to find the exact solution to the system.
    
    Parameters
    ----------
    A : ndarray
        Coefficient matrix for the system.
    b : ndarray
        Vector for the system.
            
    Returns
    -------
    x : ndarray
        Exact solution to the system.
    """
    
    # Factorize the matrix in-place
    LUfactorize(A)
    
    lA = len(A)
    x = np.zeros(lA, dtype = complex)
    y = np.zeros(lA, dtype = complex)
    y[0] = b[0]
    # Forward substitution to solve Ly = b for y
    for i in range(1, lA):
        y[i] = b[i] - np.dot(A[i,:i], y[:i])
        
    # Back substitution to solve Ux = y for x
    for j in range(lA):
        k = lA - (j+1)
        x[k] = (y[k] - np.dot(x, A.T[:, k])) / A[k, k]
        
    return x.T


# Sparse matrices

    
def CSRformat(A):
    """
    Compresses a square matrix `A`.
    
    Compression follows the CSR (compressed sparse row) format,
    where a matrix is represented by three vectors.
    
    Parameters
    ----------
    A : ndarray
        Square matrix to compress.
            
    Returns
    -------
    sA : ndarray
        Vector containing the nonzero entries of A.
    iA : ndarray
        Vector containing the indices of the first nonzero element
        on each row of A, as well as the expected number
        on the "(n+1)-th row" of A.
    jA : ndarray
        Vector containing the columns of the nonzero entries of A.
    """
    n, m = A.shape
    nnz = np.count_nonzero(A)  # Number of nonzero elements in A
    
    sA = np.zeros((nnz,), dtype = complex)
    jA = np.zeros((nnz,), dtype = int)
    iA = np.zeros((n+1,), dtype = int)
    
    k = 0  # Counter
    for i in range(n):
        iA[i] = k  # Index of the first nonzero element of row i
        nzind = np.nonzero(A[i])  # Indices of nonzero elements on row i
        nzval = A[i][A[i] != 0]  # Values of nonzero elements on row i
        l = len(nzval)  # Number of nonzero elements on row i
        r = np.linspace(k, k+l-1, l, dtype=int)  # Corresponding indices in sA
        sA[r] = nzval[:]
        jA[r] = nzind[:]
        k += l
    iA[n] = k  # Add final index
    
    return sA, iA, jA


def LUcsr(sA, iA, jA, remove_zeros=True):
    """
    Apply LU decomposition to a sparse matrix.
    
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
        * True  : removes all zeros left on the skyline after fill-in (default).
        * False : the zeros are left in the sparse vectors, which is easier to manipulate in some cases.
        
    Returns
    -------
    sLU : ndarray
        Vector containing the nonzero entries
        of the in-place LU decomposition of `A`.
    iA : ndarray
        Vector containing the indices of the first nonzero element
        on each row of the in-place LU decomposition of `A`,
        as well as the expected number
        on the "(`n+1`)-th row" of `A`.
    jLU : ndarray
        Vector containing the columns of the nonzero entries
        of the in-place LU factorization of `A`.
    """
    
    n = len(iA) - 1
    right = 1
    left = 1
    
    # Find the skyline on the left and on the right by checking each column
    for i in range(n):
        curr_right = jA[iA[i+1] - 1] - i + 1
        if curr_right > right:
            right = curr_right
            
        curr_left = i - jA[iA[i]] + 1
        if curr_left > left:
            left = curr_left
        
    bandwidth = left + right - 1
    
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
    
    nleft = n - left
    nright = n - right
    size = int(n*n - (nleft*(nleft + 1) + (nright*(nright + 1)))/2)
    sLU = np.zeros(size, dtype = complex)
    jLU = np.zeros(size)
    curr_index = 0
    
    for i in range(n):
        iright = i + right
        for j in range(iA[i], iA[i+1]):
            ind = get_index(i, jA[j])
            sLU[ind] = sA[j]
        iA[i] = curr_index
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
    iA[-1] = size
    
    # LU decomposition
    for k in range(n):
        index_kk = get_index(k, k)
        akk = sLU[index_kk]
        for j in range(k+1, k + left):
            if j < n:
                index_ljk = get_index(j, k)
                ljk = sLU[index_ljk] / akk
                sLU[index_ljk] = ljk
                index_loop_jk = index_ljk + 1
                index_loop_kk = index_kk + 1
                for m in range(k+1, k + right):
                    if m < n:
                        a = sLU[index_loop_jk] - ljk*sLU[index_loop_kk]
                        sLU[index_loop_jk] = a
                        index_loop_jk += 1
                        index_loop_kk += 1
    
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
                    iA[line] = count
                    line += 1
            iA[-1] = count+1
            sLU = np.delete(sLU, zeros)
            jLU = np.delete(jLU, zeros)
        
    return sLU.T, iA, jLU.T


def LUcsrsolve(sA, iA, jA, b):
    """
    Solves the linear system ``Ax = b`` where `A` is given in CSR format.
    
    The matrix is decomposed using a skyline solver and
    then solved by successive forward and back substitution.
    
    Parameters
    ----------
    sA : ndarray
        Vector containing the nonzero entries of A.
    iA : ndarray
        Vector containing the indices of the first nonzero element
        on each row of A, as well as the expected number
        on the "(n+1)-th row" of A.
    jA : ndarray
        Vector containing the columns of the nonzero entries of A.
    b : ndarray
        Vector of the linear system.
    Returns
    -------
    x : ndarray
        Solution to the matrix equation.
    """
    n = len(iA) - 1
    sA, iA, jA = LUcsr(sA, iA, jA, False)  # Sparse decomposition, while keeping zero elements on the skyline
    
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
    
    x = np.zeros(n, dtype = complex)
    y = np.zeros(n, dtype = complex)

    # Forward substitution to solve Ly = b for y
    y[0] = b[0]
    for i in range(1, n):
        s = 0
        for k in range(i):
            s += get_element(i, k) * y[k]
        y[i] = b[i] - s
        
    # Back substitution to solve Ux = y for x
    x[-1] = y[-1] / get_element(n-1, n-1)
    for i in range(n-2, -1, -1):
        s = 0
        for k in range(i+1, n):
            s += get_element(i, k) * x[k]
        x[i] = (y[i] - s) / get_element(i, i)
    return x


def RCMK(iA, jA):
    """
    Find a permutation vector with the Reverse Cuthill-McKee algorithm.
    
    The permutation vector is good when dealing with long, thin problems,
    but has very little theoretical guarantees
    and one should thus be careful when using it.
    
    Parameters
    ----------
    iA : ndarray
        Vector containing the indices of the first nonzero element
        on each row of A, as well as the expected number
        on the "(n+1)-th row" of A.
    jA : ndarray
        Vector containing the columns of the nonzero entries of A.
    
    Returns
    -------
    r : ndarray
        Permutation vector using the Reverse Cuthill-McKee algorithm.
    """
    n = len(iA) - 1
    Qu = deque()
    R = deque()
    deg = np.zeros(n)
    for i in range(n):
        deg[i] = iA[i+1] - iA[i]
        
    def lowest_index():
        """
        Find the lowest degree node in a graph that is not yet in the queue.
        
        This function is needed for when certain nodes have degree one
        (and thus do not add any other nodes to the queue when popped).
        
        Parameters
        ----------
        None
            
        Returns
        -------
        index : int
            Index of the lowest degree node that is not yet in the queue.
        """
        index = -1
        lowest = n + 1
        for i in range(n):
            if i not in Qu and i not in R and lowest > deg[i]:
                lowest = deg[i]
                index = i
        return index
    
    Qu.append(lowest_index())
    adj = []
    length = 1;
    # If either not all nodes have been permuted
    # or if some nodes haven't been popped yet
    while length < n or len(Qu) > 0:
        # No new nodes need to be added, flush the queue
        if (length == n):
            while Qu:
                R.append(Qu.popleft())
            r = np.zeros(n, dtype = int)
            for i in range(n):
                r[i] = R.pop()
            return r
        # Not all nodes have been popped yet, however the queue is empty
        # Need to find the next node to pop
        if len(Qu) == 0:
            E = lowest_index()
            length += 1
        else :
            E = Qu.popleft()
        R.append(E)
        # Iterate over direct neighbours, add them by order of lowest degree
        # if they aren't yet in the queue
        for j in range(iA[E],iA[E+1]):
            if jA[j] not in R and jA[j] not in Qu:
                adj += [(jA[j], deg[jA[j]])]
        sort = sorted(adj, key=lambda tup: tup[1], reverse=False)
        for x in sort:
            Qu.append(x[0])
        length += len(sort)
        adj = []
    # Turn queue into an RCMK-compliant permutation vector
    # by popping in reverse order
    r = np.zeros(n, dtype = int)
    for i in range(n):
        r[i] = R.pop()
    return r


def ccore(clscale):
    """
    Finite element model of a magnet.
    
    This model generates the matrices for the test cases.
    
    Parameters
    ----------
    clscale : int
        Coarseness of the mesh.
        
    Returns
    -------
    A : ndarray
        Matrix for the matrix equation ``Ax = b``.
    b : ndarray
        Vector for the matrix equation.
    """
    
    mur = 100.     # Relative magnetic permeability of region CORE 1:air 1000:steel
    gap = 0.001     # air gap lenght in meter
    
    
    DEBUG = False
    PY3 = sys.version_info.major == 3
    def printf(*args):
        if not DEBUG: return
        if PY3:
            exec("print( *args )")
        else:
            for item in args: exec("print item,")
            exec("print")
    
    def errorf(*args):
        if PY3:
            exec("print( *args )")
        else:
            for item in args: exec("print item,")
            exec("print")
        exit(1)
    
        
    # This scripts assembles and solves a simple static Laplacian problem
    # using exclusively the python api of Gmsh.
    
    # Geometrical parameters
    L=1
    R=0.3 # R=5 far raw mesh, R=1 normal mesh
    CoreX=0.3
    CoreY=0.3
    A=0.4  # core length in x direction 
    B=0.4  # core length in y direction 
    D=gap  # air gap length
    E=0.1  # core width
    F=0.01
    G=0.05
    
    # Physical regions
    DIRICHLET0 = 11 # Physical Line tag of a=0 boundary condition
    AIR = 1
    CORE = 2        # Physical Surface tag of magnetic core
    COILP = 3       # Physical Surface tag of positive current coil
    COILN = 4       # Physical Surface tag of negative current coil
    
    # Model parameters
    mu0 = 4.e-7*np.pi
    J = 1.e6         # Current density (A/m^2)
    Integration = 'Gauss2'
    
    # Analytical validation 
    CoilSection = G*(B-2*E-2*F)
    RelGap = D/(mu0*E)
    RelCore = (2*(A+B-2*E)-D)/(mu0*mur*E)
    
    
    
    def create_geometry():
        model.add("ccore")
        lc1=L/10.*R
    
        factory.addPoint(0,0,0,lc1, 1)
        factory.addPoint(L,0,0,lc1, 2)
        factory.addPoint(L,L,0,lc1, 3)
        factory.addPoint(0,L,0,lc1, 4)
        factory.addLine(1,2, 1)
        factory.addLine(2,3, 2)
        factory.addLine(3,4, 3)
        factory.addLine(4,1, 4)
        factory.addCurveLoop([1, 2, 3, 4], 1)
    
        # magnetic C-core
        lc2=A/10.*R
        lc3=D/2.*R
        factory.addPoint(CoreX,CoreY,0,lc2, 5)
        factory.addPoint(CoreX+A,CoreY,0,lc2, 6)
        factory.addPoint(CoreX+A,CoreY+(B-D)/2.,0,lc3, 7)
        factory.addPoint(CoreX+A-E,CoreY+(B-D)/2.,0,lc3, 8)
        factory.addPoint(CoreX+A-E,CoreY+E,0,lc2, 9)
        factory.addPoint(CoreX+E,CoreY+E,0,lc2, 10)
        factory.addPoint(CoreX+E,CoreY+B-E,0,lc2, 11)
        factory.addPoint(CoreX+A-E,CoreY+B-E,0,lc2, 12)
        factory.addPoint(CoreX+A-E,CoreY+(B+D)/2.,0,lc3, 13)
        factory.addPoint(CoreX+A,CoreY+(B+D)/2.,0,lc3, 14)
        factory.addPoint(CoreX+A,CoreY+B,0,lc2, 15)
        factory.addPoint(CoreX,CoreY+B,0,lc2, 16)
    
        factory.addLine(5,6, 5)
        factory.addLine(6,7,  6)
        factory.addLine(7,8,  7)
        factory.addLine(8,9,  8)
        factory.addLine(9,10, 9)
        factory.addLine(10,11, 10)
        factory.addLine(11,12, 11)
        factory.addLine(12,13, 12)
        factory.addLine(13,14, 13)
        factory.addLine(14,15, 14)
        factory.addLine(15,16, 15)
        factory.addLine(16,5, 16)
    
        factory.addCurveLoop([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 2)
    
        # inductors
        lc4=lc2 # F/2.*R
    
        factory.addPoint(CoreX+E+F,CoreY+E+F,0,lc4, 17)
        factory.addPoint(CoreX+E+F+G,CoreY+E+F,0,lc4, 18)
        factory.addPoint(CoreX+E+F+G,CoreY+B-E-F,0,lc4, 19)
        factory.addPoint(CoreX+E+F,CoreY+B-E-F,0,lc4, 20)
        factory.addLine(17,18, 17)
        factory.addLine(18,19, 18)
        factory.addLine(19,20, 19)
        factory.addLine(20,17, 20)
    
        factory.addCurveLoop([17, 18, 19, 20], 3)
    
        factory.addPoint(CoreX-F-G,CoreY+E+F,0,lc4, 21)
        factory.addPoint(CoreX-F,CoreY+E+F,0,lc4, 22)
        factory.addPoint(CoreX-F,CoreY+B-E-F,0,lc4, 23)
        factory.addPoint(CoreX-F-G,CoreY+B-E-F,0,lc4, 24)
    
        factory.addLine(21,22, 21)
        factory.addLine(22,23, 22)
        factory.addLine(23,24, 23)
        factory.addLine(24,21, 24)
    
        factory.addCurveLoop([21, 22, 23, 24], 4)
    
        factory.addPlaneSurface([1,2,3,4], 1)
        factory.addPlaneSurface([2], 2)
        factory.addPlaneSurface([3], 3)
        factory.addPlaneSurface([4], 4)
    
        factory.synchronize()
    
        model.addPhysicalGroup(2, [1], 1)
        model.addPhysicalGroup(2, [2], 2)
        model.addPhysicalGroup(2, [3], 3)
        model.addPhysicalGroup(2, [4], 4)
        model.addPhysicalGroup(1, [1,2,3,4], 11)
    
        model.setPhysicalName(2, 1, 'AIR')
        model.setPhysicalName(2, 2, 'CORE')
        model.setPhysicalName(2, 3, 'COILP')
        model.setPhysicalName(2, 4, 'COILN')
        model.setPhysicalName(1, 11, 'DIR')
        return
    
    def solve():
        mshNodes = np.array(model.mesh.getNodes()[0])
        numMeshNodes = len(mshNodes)
        printf('numMeshNodes =', numMeshNodes)
        maxNodeTag = np.amax(mshNodes)
        printf('maxNodeTag =', maxNodeTag)
        
        
        # initializations of global assembly arrays iteratively filled-in during assembly
        matrowflat = np.array([], dtype=np.int32)
        matcolflat = np.array([], dtype=np.int32)
        matflat = np.array([], dtype=np.int32)
        rhsrowflat = np.array([], dtype=np.int32)
        rhsflat = np.array([], dtype=np.int32)
    
        # typNodes[tag] = {0,1,2} 0: does not exist, internal node, boundary node
        # Existing node tags are defined here. Boundary node tag are identified later.
        typNodes = np.zeros(maxNodeTag+1, dtype=np.int32) # 1:exists 2:boundary
        for tagNode in mshNodes:
            typNodes[tagNode] = 1
    
        # The read-in mesh is iterated over, looping successively (nested loops) over:
        # Physical groups/Geometrical entities/Element types/Elements
        vGroups = model.getPhysicalGroups()
        for iGroup in vGroups:
            dimGroup = iGroup[0] # 1D, 2D or 3D
            tagGroup = iGroup[1] # the word 'tag' is systematically used instead of 'number'
            vEntities = model.getEntitiesForPhysicalGroup(dimGroup, tagGroup)
            for tagEntity in vEntities:
                dimEntity = dimGroup # FIXME dimEntity should be optional when tagEntity given.
                vElementTypes = model.mesh.getElementTypes(dimEntity,tagEntity)
                for elementType in vElementTypes:
                    vTags, vNodes = model.mesh.getElementsByType(elementType, tagEntity)
                    numElements = len(vTags)
                    numGroupNodes = len(vNodes)
                    enode = np.array(vNodes).reshape((numElements,-1))
                    numElementNodes = enode.shape[1]
                    printf('\nIn group', tagGroup, ', numElements = e =', numElements)
                    printf('numGroupNodes =', numGroupNodes,', numElementNodes = n =', numElementNodes)
                    printf('%enodes (e,n) =', enode.shape)
    
                    # Assembly of stiffness matrix for all 2 dimensional elements (triangles or quadrangles)
                    if dimEntity==2 :
                        uvwo, numcomp, sf = model.mesh.getBasisFunctions(elementType, Integration, 'Lagrange')
                        #printf('%uvwo =', len(uvwo), '%numcomp =', numcomp, '%sf =', len(sf))
                        weights = np.array(uvwo).reshape((-1,4))[:,3] # only keep the Gauss weights
                        numGaussPoints = weights.shape[0]
                        printf('numGaussPoints = g =', numGaussPoints, ', %weights (g) =', weights.shape)
                        sf = np.array(sf).reshape((numGaussPoints,-1))
                        printf('%sf (g,n) =', sf.shape)
                        if sf.shape[1] != numElementNodes:
                            errorf('Something went wrong')
                        _, numcomp, dsfdu = model.mesh.getBasisFunctions(elementType, Integration, 'GradLagrange')
                        #printf('%uvwo =', len(uvwo), '%numcomp =', numcomp, '%dsfdu =', len(dsfdu))
                        dsfdu = np.array(dsfdu).reshape((numGaussPoints,numElementNodes,3))[:,:,:-1] #remove useless dsfdw
                        printf('%dsfdu (g,n,u) =', dsfdu.shape)
                        
                        qjac, qdet, qpoint = model.mesh.getJacobians(elementType, Integration, tagEntity)
                        printf('Gauss integr:',len(qjac),len(qdet),len(qpoint),'= (9, 1, 3) x',numGaussPoints,'x',numElements)
                        qdet = np.array(qdet).reshape((numElements,numGaussPoints))
                        printf('%qdet (e,g) =', qdet.shape)
                        #remove components of dxdu useless in dimEntity dimensions (here 2D)
                        dxdu = np.array(qjac).reshape((numElements,numGaussPoints,3,3))[:,:,:-1,:-1]
                        # jacobien store by row, so dxdu[i][j] = dxdu_ij = dxi/duj 
                        printf('%dxdu (e,g,x,u)=', dxdu.shape)
                            
                        if tagGroup == CORE:
                            nu = 1./(mur*mu0)
                        else:
                            nu = 1./mu0
                                         
                        dudx = np.linalg.inv(dxdu)
                        # dudx[j][k] = dudx_jk = duj/dxk
                        printf('%dudx (e,g,u,x) =', dudx.shape)
                        #print np.einsum("egxu,eguy->egxy",dxdu,dudx)[0][0];
                        #print np.einsum("egxu,egvx->eguv",dxdu,dudx)[0][0];
                        dsfdx  = np.einsum("egxu,gnu->egnx",dudx,dsfdu); # sum over u = dot product
                        printf('%dsfdx (e,g,n,x) =', dsfdx.shape)
                        localmat = nu * np.einsum("egik,egjk,eg,g->eij", dsfdx, dsfdx, qdet, weights) # Gauss integration
                        printf('%localmat (e,n,n) =', localmat.shape)
                        
                        # The next two lines are rather obscure. See explanations at the bottom of the file. 
                        matcol = np.repeat(enode[:,:,None],numElementNodes,axis=2)
                        matrow = np.repeat(enode[:,None,:],numElementNodes,axis=1)
                        
                        matcolflat = np.append(matcolflat, matcol.flatten())
                        matrowflat = np.append(matrowflat, matrow.flatten())
                        matflat = np.append(matflat, localmat.flatten())
    
                        if tagGroup == COILP or tagGroup == COILN:
                            if tagGroup == COILP:
                                load = J
                            elif tagGroup == COILN:
                                load = -J
                            localrhs = load * np.einsum("gn,eg,g->en", sf, qdet, weights)
                            printf('Check rhs:', np.sum(localrhs), "=", load*CoilSection)
                            rhsrowflat = np.append(rhsrowflat, enode.flatten())
                            rhsflat = np.append(rhsflat, localrhs.flatten())
    
                    # identify boundary node
                    if tagGroup == DIRICHLET0:
                        for tagNode in vNodes:
                            typNodes[tagNode] = 2
    
        printf('\nDimension of arrays built by the assembly process')
        printf('%colflat = ', matcolflat.shape)
        printf('%rowflat = ', matrowflat.shape)
        printf('%localmatflat = ', matflat.shape)
        printf('%rhsrowflat = ', rhsrowflat.shape)
        printf('%rhsflat = ', rhsflat.shape)
    
        # Associate to all mesh nodes a line number in the system matrix
        # reserving top lines for internal nodes and bottom lines for fixed nodes (boundary nodes).
        node2unknown = np.zeros(maxNodeTag+1, dtype=np.int32)
        index = 0
        for tagNode,typ in enumerate(typNodes):
            if  typ == 1: # not fixed
                index += 1
                node2unknown[tagNode] = index
        numUnknowns = index
        printf('numUnknowns =', numUnknowns)
        for tagNode,typ in enumerate(typNodes):
            if  typ == 2: # fixed
                index += 1
                node2unknown[tagNode] = index
    
        if index != numMeshNodes:
            errorf('Something went wrong')
    
        unknown2node = np.zeros(numMeshNodes+1, dtype=np.int32)
        for node, unkn in enumerate(node2unknown):
            unknown2node[unkn] = node
    
        printf('\nDimension of nodes vs unknowns arrays')
        printf('%mshNodes=',mshNodes.shape)
        printf('%typNodes=',typNodes.shape)
        printf('%node2unknown=',node2unknown.shape)
        printf('%unknown2node=',unknown2node.shape)
    
        # Generate system matrix A=globalmat and right hand side b=globalrhs
    
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        # 'node2unknown-1' are because python numbers rows and columns from 0
        globalmat = scipy.sparse.coo_matrix(
            (matflat, (node2unknown[matcolflat]-1,node2unknown[matrowflat]-1) ),
            shape=(numMeshNodes, numMeshNodes)).todense()
        
        globalrhs = np.zeros(numMeshNodes)
        for index,node in enumerate(rhsrowflat):
            globalrhs[node2unknown[node]-1] += rhsflat[index]
    
        printf('%globalmat =', globalmat.shape, ' %globalrhs =', globalrhs.shape)
    
        A = globalmat[:numUnknowns,:numUnknowns]
        b = globalrhs[:numUnknowns]
        success, sol = mysolve(np.array(A, dtype=np.float64), b)
        if not success:
            errorf('Solver not implemented yet')
        sol = np.append(sol,np.zeros(numMeshNodes-numUnknowns))
        printf('%sol =', sol.shape)
        
        # Export solution
        sview = gmsh.view.add("solution")
        gmsh.view.addModelData(sview,0,"","NodeData",unknown2node[1:],sol[:,None])
        #gmsh.view.write(sview,"a.pos")
        printf('Flux (analytical) =', J*CoilSection/(RelCore+RelGap))
        printf('Flux (computed) =', np.max(sol)-np.min(sol))
        return A, b
    
        
    model = gmsh.model
    factory = model.geo
    gmsh.initialize(sys.argv)
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", clscale)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("View[0].IntervalsType", 3)
    gmsh.option.setNumber("View[0].NbIso", 20)
    
    create_geometry()
    if(0):
        model.mesh.setRecombine(2,COILP)
        model.mesh.setRecombine(2,COILN)
        model.mesh.setRecombine(2,AIR)
        model.mesh.setRecombine(2,CORE)
    model.mesh.generate(2)
    
    mat, vec = solve()
    return np.array(mat, dtype=np.float64), vec


def expFull(precision = 'report'):
    """
    Plot suite for homework 3 (dense part)
    
    Various plots are created:
        
        * density of `A` as a function of system size.
        * density of in-place LU factorization as a function of density of `A`.
        * execution time of LUsolve as a function of system size.
        
    Parameters
    ----------
    precision : {'report', 'test'}, optional
        Precision of the plots.
    
    Returns
    -------
    None
    """
    print("==========================================================")
    print("                     DENSE MATRICES")
    print("==========================================================")
    if precision == 'report':
        data = range(5, 50)
    elif precision == 'test':
        data = range(20, 30)
    else:
        print("Please give a valid precision specifier")
    
    # Preallocate result vectors
    a_dens = np.zeros(len(data), dtype = np.float64)
    lu_dens = np.zeros(len(data), dtype = np.float64)
    size = np.zeros(len(data), dtype = int)
    exec_time = np.zeros(len(data), dtype = np.float64)

    index = 0
    for i in data:
        A, b = ccore(i)
        size[index] = len(A)
        a_dens[index] = np.count_nonzero(A)/len(A)**2
        
        t0 = timeit.default_timer()
        LUsolve(A, b) # Factorization happens in-place, A is now filled in
        t1 = timeit.default_timer()
        exec_time[index] = t1 - t0
        
        lu_dens[index] = np.count_nonzero(A)/len(A)**2
        
        index += 1
        
    # Density of LU as a function of density before factorization
    plt.plot(a_dens, lu_dens, 'go')
    plt.xlabel(r"Density of the original matrix")
    plt.ylabel(r"Density of the LU decomposition")
    plt.title(r"Influence of original density"
              "\n"
              r"on density after decomposition")
    plt.legend(['Data points'])
    plt.show()
    
    # Density of LU as a function of system size
    plt.plot(size, lu_dens, 'go')
    plt.xlabel(r"Size of the system")
    plt.ylabel(r"Density of the LU decomposition")
    plt.title(r"Influence of system size"
              "\n"
              r"on density after decomposition")
    plt.legend(['Data points'])
    plt.show()
    
    sizelog = np.log10(size)
    
    # Linear regression
    fit_LUsolve = np.polyfit(sizelog, np.log10(exec_time), 1)
    fit_fn_LUsolve = np.poly1d(fit_LUsolve)
    
    ran = np.log10(range(min(size), max(size)))
    
    # Logarithmic plot of LUsolve execution time
    plt.plot(sizelog, np.log10(exec_time), 'go', ran, fit_fn_LUsolve(ran), '--g')
    plt.xlabel(r"Size of the system, $\log_{10} \,n$")
    plt.ylabel(r"Execution time of LUsolve $[\log_{10} \,\mathrm{s}]$")
    plt.title(r"Execution time of LUsolve on a logarithmic scale"
              "\n"
              r"as a function of $\log_{10} \,n$, the size of $A$")
    plt.legend(['Data points', 'Linear fit'])
    plt.show()

    
def expSparse(precision = 'report'):
    """
    Plot suite for homework 3 (sparse part)
    
    Various plots are created:
        
        * Execution time of CSRformat as a function of system size
        * Influence of Reverse Cuthill-McKee ordering on the density of the in-place LU decomposition 
        * Execution time of RCMK
        * Execution time of LUcsrsolve for the original matrix and for the reordered matrix
        * Spy plots of the coefficient matrix before and after fill-in, both with and without reordering
        
    Parameters
    ----------
    precision : {'report', 'test'}, optional
        Precision of the plots.
    
    Returns
    -------
    None
    """
    print("==========================================================")
    print("                    SPARSE MATRICES")
    print("==========================================================")
    if precision == 'report':
        data = range(5, 50)
    elif precision == 'test':
        data = range(20, 30)
    else:
        print("Please give a valid precision specifier")
    
    # Preallocate result vectors
    lu_dens = np.zeros(len(data), dtype = np.float64)
    lu_rcmk_dens = np.zeros(len(data), dtype = np.float64)
    size = np.zeros(len(data), dtype = int)
    exec_time_format = np.zeros(len(data), dtype = np.float64)
    exec_time_csr_solve = np.zeros(len(data), dtype = np.float64)
    exec_time_rcmk_solve = np.zeros(len(data), dtype = np.float64)
    exec_time_rcmk = np.zeros(len(data), dtype = np.float64)
    
    spy_index = 20
    
    index = 0
    for i in data:
        A, b = ccore(i)
        size[index] = len(A)
        
        t0_format = timeit.default_timer()
        sA, iA, jA = CSRformat(A)
        if i == spy_index:
            plt.spy(scipy.sparse.csr_matrix(A), precision=1, markersize=2)
            plt.show()
        t1_format = timeit.default_timer()
        exec_time_format[index] = t1_format - t0_format
        
        t0_rcmk = timeit.default_timer()
        r = RCMK(iA, jA)
        t1_rcmk = timeit.default_timer()
        exec_time_rcmk[index] = t1_rcmk - t0_rcmk
        
        a_rcmk = (A[:, r])[r, :]
        if i == spy_index:
            plt.spy(scipy.sparse.csr_matrix(a_rcmk), precision=1, markersize=2)
            plt.show()
        b_rcmk = b[r]
        sA_rcmk, iA_rcmk, jA_rcmk = CSRformat(a_rcmk)
        
        t0_csr_solve = timeit.default_timer()
        LUcsrsolve(sA, iA, jA, b)
        t1_csr_solve = timeit.default_timer()
        exec_time_csr_solve[index] = t1_csr_solve - t0_csr_solve
        
        t0_rcmk_solve = timeit.default_timer()
        LUcsrsolve(sA_rcmk, iA_rcmk, jA_rcmk, b_rcmk)
        t1_rcmk_solve = timeit.default_timer()
        exec_time_rcmk_solve[index] = t1_rcmk_solve - t0_rcmk_solve
        
        LUfactorize(a_rcmk)
        LUfactorize(A)
        lu_rcmk_dens[index] = np.count_nonzero(a_rcmk)/size[index]**2
        lu_dens[index] = np.count_nonzero(A)/size[index]**2
        
        index += 1
    
    sizelog = np.log10(size)
    
    # Linear regression
    formatlog = np.log10(exec_time_format)
    
    fit_CSRformat = np.polyfit(sizelog, formatlog, 1)
    fit_fn_CSRformat = np.poly1d(fit_CSRformat)
    
    ran = np.log10(range(min(size), max(size)))
    
    # Logarithmic plot of CSRformat execution time
    plt.plot(sizelog, formatlog, 'bo', ran, fit_fn_CSRformat(ran), '--b')
    plt.xlabel(r"Size of the system, $\log_{10} \,n$")
    plt.ylabel(r"Execution time of CSRformat $[\log_{10} \,\mathrm{s}]$")
    plt.title(r"Execution time of CSRformat on a logarithmic scale"
              "\n"
              r"as a function of $\log_{10} \,n$, the size of $A$")
    plt.legend(['Data points', 'Linear fit'])
    plt.show()
    
    # Density of in-place LU factorization after reordering
    plt.loglog(size, lu_dens, 'bo', size, lu_rcmk_dens, 'ro')
    plt.xlabel(r"Size of the system, $n$")
    plt.ylabel(r"Density")
    plt.title(r"Density of in-place LU decomposition"
              "\n"
              r"as a function of $n$, the size of $A$")
    plt.legend(['Original matrix', 'Reordered matrix'])
    plt.show()
    
    # Linear regression
    rcmklog = np.log10(exec_time_rcmk)
    
    fit_RCMK = np.polyfit(sizelog, rcmklog, 1)
    fit_fn_RCMK = np.poly1d(fit_RCMK)
    
    # Logarithmic plot of RCMK execution time
    plt.plot(sizelog, rcmklog, 'bo', ran, fit_fn_RCMK(ran), '--b')
    plt.xlabel(r"Size of the system, $\log_{10} \, n$")
    plt.ylabel(r"Execution time of RCMK $[\log_{10} \, \mathrm{s}]$")
    plt.title(r"Execution time of RCMK on a logarithmic scale"
              "\n"
              r"as a function of $\log_{10} \, n$, the size of $A$")
    plt.legend(['Data points', 'Linear fit'])
    plt.show()
    
    # Linear regression
    csrsolvelog = np.log10(exec_time_csr_solve)
    rcmksolvelog = np.log10(exec_time_rcmk_solve)
    
    fit_csr_solve = np.polyfit(sizelog, csrsolvelog, 1)
    fit_fn_csr_solve = np.poly1d(fit_csr_solve)
    fit_rcmk_solve = np.polyfit(sizelog, rcmksolvelog, 1)
    fit_fn_rcmk_solve = np.poly1d(fit_rcmk_solve)
    
    # Logarithmic plot of LUcsr execution time
    plt.plot(sizelog, csrsolvelog, 'bo', sizelog, rcmksolvelog, 'ro', ran, fit_fn_csr_solve(ran), '--b', ran, fit_fn_rcmk_solve(ran), '--r')
    plt.xlabel(r"Size of the system, $\log_{10} \, n$")
    plt.ylabel(r"Execution time of LUcsr $[\log_{10} \, \mathrm{s}]$")
    plt.title(r"Execution time of LUcsr on a logarithmic scale"
              "\n"
              r"as a function of $\log_{10} \, n$, the size of $A$")
    plt.legend(['Original matrix', 'Reordered matrix', 'Linear fit (original)', 'Linear fit (reordered)'])
    plt.show()


if __name__ == "__main__":
    expFull(precision='report')
    expSparse(precision='report')