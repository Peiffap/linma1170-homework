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


SolverType = 'LU'


def mysolve(A, b):
    """
    Solve the matrix equation ``Ax = b`` for `x`.
    
    According to which solver is selected by the global variable "SolverType",
    this solver uses either:
        - scipy.sparse.linalg's built-in sparse solver;
        - a solver based on QR factorization;
        - a solver based on LU factorization;
        - a solver based on the GMRES algorithm.
    
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
    for k in range(0, lA-1):
            A[k+1:, k] = A[k+1:, k]/A[k, k]
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
    x = np.zeros(lA)
    y = np.zeros(lA)
    y[0] = b[0]
    
    # Forward substitution to solve Ly = b for y
    for i in range(1, lA):
        y[i] = b[i] - np.dot(A[i,0:i], y[0:i])
        
    # Back substitution to solve Ux = y for x
    for j in range(lA):
        k = lA - (j+1)
        x[k] = (y[k] - np.dot(x[:], A.T[:, k])) / A[k, k]
        
    return x.T


# Sparse matrices

    
def CSRformat(A):
    """
    Compresses a square matrix `A`.
    
    Compression follows the CSR (compressed storage by rows) format,
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
        gauche2 = left-2
        ng = gauche2*(gauche2+1)/2
        gauche2i = gauche2 - i
        if gauche2i > 0:
            ng -= (gauche2i)*(gauche2i+1)/2
            
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
    
    def get_index(i, j):
        """
        Returns the index in sA of element `(i, j)` in `A`.
        
        Parameters
        ----------
            i : int
                Index of the row of the element.
            j : int
                Index of the column of the element.
                
        Returns
        -------
            ind : int
                Index of the element in sA.
        """
        iright = i + right
        if j >= iright or j <= i - left:
            return 0
        
        index = i*bandwidth + (j-i)
        gauche2 = left-2
        ng = gauche2*(gauche2+1)/2
        gauche2i = gauche2 - i
        if gauche2i > 0:
            ng -= (gauche2i)*(gauche2i+1)/2
            
        nd = 0
        if iright > n:
            delta = iright - n
            nd = delta*(delta-1)/2
            
        return int(index - ng - nd)
    
    x = np.zeros(n)
    y = np.zeros(n)

    # Forward substitution to solve Ly = b for y
    y[0] = b[0]
    for i in range(1, n):
        s = 0
        for k in range(i):
            s += sA[get_index(i,k)] * y[k]
        y[i] = b[i] - s
    
    # Back substitution to solve Ux = y for x
    x[n-1] = y[n-1] / sA[get_index(n-1, n-1)]
    for i in range(n-1, -1, -1):
        s = 0
        for k in range(i+1, n):
            s += sA[get_index(i, k)] * x[k]
            
        x[i] = (y[i] - s)/sA[get_index(i, i)]
    
    return x.T


def RCMK(iA, jA):
    n = len(iA)-1
    lowest = float("inf")
    deg = np.zeros((n,1))
    index = -1
    for i in range(n):
        deg[i]= iA[i+1]-iA[i]
        if lowest > deg[i]:
            lowest=deg[i]
            index = i
    Qu = deque([index])
    R = deque()
    adj = []
    while Qu:
        E = Qu.popleft()
        R.append(E)
        for j in range(iA[E],iA[E+1]):
            if jA[j] not in R and jA[j] not in Qu:
                adj +=[(jA[j],deg[jA[j]])]
        sort = sorted(adj,key=lambda tup: tup[1],reverse=False)
        for x in sort:
            Qu.append(x[0])
        adj=[]
    r = np.zeros(n,dtype=int)
    for i in range(n):
        r[i]=R.pop()
    return r

def ccore(clscale):
    # mesh refinement 1:fine 10:coarse 50:very coarse
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


if __name__ == "__main__":
    
#        A=np.array(A,dtype=np.float64)
#        sA,iA,jA = CSRformat(A)
#        r=RCMK(iA,jA)
#        M = scipy.sparse.csr_matrix(A)
#        N1=A[r,:]
#        N=scipy.sparse.csr_matrix(N1[:,r])
#        
#        plt.spy(M)
#        plt.show()
#        plt.spy(N)
#        plt.show()
    timeLUsolve=[]
    timeCSRsolve=[]
    timeRCMKsolve=[]
    size=[]
    for i in range(20,30):
        A, b = ccore(i)
        n=len(A)
        size.append(n)
#        plt.spy(scipy.sparse.csr_matrix(A))
#        plt.show()
        sA,iA,jA = CSRformat(A)
        r = RCMK(iA,jA)
        t2=timeit.default_timer()
        CSRsolve(sA,iA,jA,b)
        t3=timeit.default_timer()
        timeCSRsolve.append(t3-t2)
        
        N1=A[r,:]
        N = N1[:, r]
        M=scipy.sparse.csr_matrix(N)
#        plt.spy(M)
#        plt.show()
        sN,iN,jN = CSRformat(N)
        t4=timeit.default_timer()
        CSRsolve(sN, iN, jN, b[r])
        t5=timeit.default_timer()
        timeRCMKsolve.append(t5-t4)
        
        t0=timeit.default_timer()
        LUsolve(A,b)
        t1=timeit.default_timer()
        timeLUsolve.append(t1-t0)

        
        #print(np.allclose(x, scipy.sparse.linalg.spsolve(A, b)[r], rtol = 1e-5, atol = 1e-5))
        print("========================================================")
        print('Matrix size = %d\n' %n)
        print('Time for LUsolve = %.5f s\n' %(t1-t0))
        print('Time for LUcsrsolve = %.5f s\n' %(t3-t2))
        print('Time for LUcsrsolve (with RCMK) = %.5f s' %(t5-t4))
        print("========================================================\n\n")
        
    plt.plot(size,timeLUsolve, 'ro')
    plt.plot(size,timeCSRsolve,'bo')
    plt.plot(size,timeRCMKsolve,'go')
    plt.show()
#    plt.xlabel('Taille de la matrice A')
#    plt.ylabel("Temps d'exécution")
#    plt.title("Temps d'exécution en fonction de la taille de la matrice A" )
#    plt.show()
    
#    plt.plot(np.log(size),np.log(timeCSR), 'ro')
#    plt.plot(np.log(size),np.log(timeCSR_RCMK), 'bo')
#    plt.plot(np.log(size),np.log(timeDense), 'go')
#    plt.xlabel('Taille de la matrice A')
#    plt.ylabel("Temps d'exécution")
#    plt.title("Temps d'exécution en fonction de la taille de la matrice A" )
#    plt.show()

