# Mathematical imports
import numpy as np
from collections import deque  # Needed for deque used in RCMK

# ccore imports
import sys
import gmsh
import scipy.sparse #### NEEDED FOR CCORE ####

# Imports for plots and timing
import matplotlib.pyplot as plt
import timeit
from mpltools import annotation


# Import ILU
from ILU0 import ILU0


def csrCG(sA, iA, jA, b, rtol, prec, iterations=False):
    """
    Solve a system given in CSR format using the conjugate gradient method
    
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
        Vector.
    rtol : float
        Relative tolerance.
    prec : boolean
        Specifies whether preconditioning needs to be used.
    iterations : boolean, optional
        * True  : returns the number of iterations.
        * False : does not return the nubmer of iterations. (default)
        
    Returns
    -------
    x : ndarray
        Solution to the system ``Ax = b``.
    res : ndarray
        Norms of the successive residuals.
    itercount : int, optional
        Number of iterations needed for convergence.
    """
    n = len(b)
    rtol /= 10  # To make sure we don't exceed the tolerance
    x = np.zeros(n, dtype = np.float64)
    res = np.copy(b)
    residue_array = []
    itercount = 0
    if not prec:
        p = np.copy(res)
        while np.linalg.norm(res)/np.linalg.norm(b) > rtol:
            residue_array.append(np.linalg.norm(res))
            Ap = csr_prod(sA, iA, jA, p)
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
            residue_array.append(np.linalg.norm(res))
            dot_res = np.dot(res, res_tilde)
            Ap = csr_prod(sA, iA, jA, p)
            alpha = dot_res / np.dot(p, Ap)
            x += alpha*p
            res -= alpha*Ap
            res_tilde = ILUcsrsolve(sM, iM, jM, res)
            beta = np.dot(res, res_tilde) / dot_res
            p = res_tilde + beta*p
            itercount += 1
        
    if iterations:  # If we want to return the number of iterations as well
        return x, residue_array, itercount
    else:
        return x, residue_array

        
def csr_prod(sA, iA, jA, p):
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


def reconstruct(sA, iA, jA):
    """
    Turns a CSR matrix into a dense matrix.
    
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
    
    Returns
    -------
    A : ndarray
        Matrix.
    """
    n = len(iA) - 1
    A = np.zeros((n, n))
    for i in range(n):
        A[i, jA[iA[i]:iA[i+1]]] = sA[iA[i]:iA[i+1]]
    return A


def ccore(clscale, mur, gap):
    """
    Finite element model of a magnet.
    
    This model generates the matrices for the test cases.
    
    Parameters
    ----------
    clscale : int
        Coarseness of the mesh.
    mur : int
        Relative permeability of the magnetic core.
    gap : float
        Length of air gap in the magnet.
        
    Returns
    -------
    A : ndarray
        Matrix for the matrix equation ``Ax = b``.
    b : ndarray
        Vector for the matrix equation.
    """
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
        success, sol = True, np.linalg.solve(np.array(A, dtype=np.float64), b)
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


def spectrum_analysis(prec='report'):
    """
    Plot graphs for the analysis of the spectrum
    before and after preconditioning.
    
    Parameters
    ----------
    prec : {'test', 'report'}, optional
        * 'test'   : lower quality plots, but faster execution time.
        * 'report' : high quality plots are generated, default.
        
    Returns
    -------
    None
    """
    if prec == 'report':
        data_mur = range(1, 1000, 30)
        data_clscale = range(10, 45)
        data_gap = range(10, 50)  # Still need to divide
    elif prec == 'test':
        data_mur = range(1, 400, 10)
        data_clscale = range(10, 30)
        data_gap = range(10, 50, 10)  # Still need to divide
    else:
        print('Please specify a valid precision parameter')
        return
    
    
    default_gap = 0.001
    default_mur = 100.
    default_clscale = 10
    
    condA_mur = np.zeros(len(data_mur))
    condMA_mur = np.zeros(len(data_mur))
    
    size_gap = np.zeros(len(data_gap))
    condA_gap = np.zeros(len(data_gap))
    condMA_gap = np.zeros(len(data_gap))
    
    size_clscale = np.zeros(len(data_clscale))
    condA_clscale = np.zeros(len(data_clscale))
    condMA_clscale = np.zeros(len(data_clscale))
    
    eigvals_init = []
    eigvals_prec = []
    
    index = 0
    
    for i in data_clscale:
        A, b = ccore(i, default_mur, default_gap)
        n = len(A)
        sA, iA, jA = CSRformat(A)
        size_clscale[index] = n
        
        sM, iM, jM = ILU0(sA, iA, jA)
        M = reconstruct(sM, iM, jM)
        condA_clscale[index] = np.linalg.cond(A)
        MA = np.linalg.inv((np.eye(n) + np.tril(M, -1)) @ np.triu(M)) @ A
        condMA_clscale[index] = np.linalg.cond(MA)
        
        index += 1
        
    plt.semilogy(size_clscale, condA_clscale, '-bo', size_clscale, condMA_clscale, '-ro')
    plt.xlabel(r"Size of the system, $n$ (influence of $\mathrm{clscale}$)")
    plt.ylabel(r"Condition number $\kappa$ of $A$ and $M^{-1}A$")
    plt.title(r"Effect of preconditioning"
              "\n"
              r"on the condition number of the matrices")
    plt.legend([r'Original matrix $A$', r'Preconditioned matrix $M^{-1}A$'])
    plt.show()
    
    index = 0
    
    for i in data_mur:
        A, b = ccore(default_clscale, i, default_gap)
        n = len(A)
        sA, iA, jA = CSRformat(A)
        
        sM, iM, jM = ILU0(sA, iA, jA)
        M = reconstruct(sM, iM, jM)
        condA_mur[index] = np.linalg.cond(A)
        MA = np.linalg.inv((np.eye(n) + np.tril(M, -1)) @ np.triu(M)) @ A
        condMA_mur[index] = np.linalg.cond(MA)
        
        eigvals_init.append(np.linalg.eigvals(A))
        eigvals_prec.append(np.linalg.eigvals(MA))
        
        index += 1
        
    for i, j in enumerate(eigvals_init):
        plt.semilogy([data_mur[i] for j in range(len(j))], j, 'o', ms=4)
    plt.xlabel(r"Relative permeability, $\mu_{\mathrm{r}}$")
    plt.ylabel(r"Spectrum of $A$")
    plt.title(r"Effect of preconditioning"
              "\n"
              r"on the spectrum of the matrix")
    plt.show()
    
    for i, j in enumerate(eigvals_prec):
        plt.semilogy([data_mur[i] for j in range(len(j))], j, 'o', ms=4)
    plt.xlabel(r"Relative permeability, $\mu_{\mathrm{r}}$")
    plt.ylabel(r"Spectrum of $M^{-1}A$")
    plt.title(r"Effect of preconditioning"
              "\n"
              r"on the spectrum of the matrix after preconditioning")
    plt.show()
        
    plt.plot(data_mur, condA_mur, '-bo', data_mur, condMA_mur, '-ro')
    plt.xlabel(r"Relative permeability, $\mu_{\mathrm{r}}$")
    plt.ylabel(r"Condition number $\kappa$ of $A$ and $M^{-1}A$")
    plt.title(r"Effect of preconditioning"
              "\n"
              r"on the condition number of the matrices")
    plt.legend([r'Original matrix $A$', r'Preconditioned matrix $M^{-1}A$'])
    plt.show()
    
    index = 0
    
    for i in data_gap:
        A, b = ccore(default_clscale, default_mur, i/10000)
        n = len(A)
        sA, iA, jA = CSRformat(A)
        size_gap[index] = n
        
        sM, iM, jM = ILU0(sA, iA, jA)
        M = reconstruct(sM, iM, jM)
        condA_gap[index] = np.linalg.cond(A)
        MA = np.linalg.inv((np.eye(n) + np.tril(M, -1)) @ np.triu(M)) @ A
        condMA_gap[index] = np.linalg.cond(MA)
        
        index += 1
        
    plt.plot(size_gap, condA_gap, '-bo', size_gap, condMA_gap, '-ro')
    plt.xlabel(r"Size of the system, $n$ (influence of $\mathrm{gap}$)")
    plt.ylabel(r"Condition number $\kappa$ of $A$ and $M^{-1}A$")
    plt.title(r"Effect of preconditioning"
              "\n"
              r"on the condition number of the matrices")
    plt.legend([r'Original matrix $A$', r'Preconditioned matrix $M^{-1}A$'])
    plt.show()
    
    
def convergence_analysis(prec='report'):
    """
    Plot graphs for the analysis of CG convergence.
    
    Parameters
    ----------
    prec : {'test', 'report'}, optional
        * 'test'   : lower quality plots, but faster execution time.
        * 'report' : high quality plots are generated, default.
        
    Returns
    -------
    None
    """
    if prec == 'report':
        data_mur = range(30, 1000, 10)
        data_gap = range(10, 50)
    elif prec == 'test':
        data_mur = range(1, 1000, 100)
        data_gap = range(10, 50, 10)
    else:
        print('Please specify a valid precision parameter')
        return
    
    clscale = 17
    gap = 0.008
    mur = 100.
    n_iter = np.zeros(len(data_mur))
    n_iter_prec = np.zeros(len(data_mur))
    condA = np.zeros(len(data_mur))
    condMA = np.zeros(len(data_mur))
    
    index = 0
    
    for i in data_mur:
        A, b = ccore(clscale, i, gap)
        sA, iA, jA = CSRformat(A)
        x, res, n_iter[index] = csrCG(sA, iA, jA, b, 1e-6, False, True)
        x, res_prec, n_iter_prec[index] = csrCG(sA, iA, jA, b, 1e-6, True, True)
        
        sM, iM, jM = ILU0(sA, iA, jA)
        M = reconstruct(sM, iM, jM)
        condA[index] = np.linalg.cond(A)
        MA = np.linalg.inv((np.eye(len(A)) + np.tril(M, -1)) @ np.triu(M)) @ A
        condMA[index] = np.linalg.cond(MA)
        
        index += 1
        
    plt.loglog(condA, n_iter, '-bo', condMA, n_iter_prec, '-ro')
    annotation.slope_marker((20000, 100), (1, 2),
                        text_kwargs={'color': 'cornflowerblue'},
                        poly_kwargs={'facecolor': (0.73, 0.8, 1)})
    annotation.slope_marker((1000, 30), (1, 2),
                        text_kwargs={'color': 'red'},
                        poly_kwargs={'facecolor': (1, 0.8, 0.73)})
    plt.xlabel(r"Condition number of the matrix, $\kappa$ (influence of $\mu_{\mathrm{r}}$)")
    plt.ylabel(r"Number of iterations")
    plt.title(r"Effect of $\kappa$"
              "\n"
              r"on the number of iterations needed for convergence")
    plt.legend([r'Original matrix $A$', r'Preconditioned matrix $M^{-1}A$'])
    plt.show()
    
#    A, b = ccore(10, 100., 0.001)
#    sA, iA, jA = CSRformat(A)
#    x, res = csrCG(sA, iA, jA, b, 1e-6, False)
#    x_prec, res_prec = csrCG(sA, iA, jA, b, 1e-6, True)
#    plt.semilogy(range(len(res)), res, '-b.', range(len(res_prec)), res_prec, '-r.')
#    plt.xlabel(r"Iteration")
#    plt.ylabel(r"Norm of the residual")
#    plt.title(r"Evolution of the value of the residual"
#              "\n"
#              r"after each iteration of the conjugate gradient method")
#    plt.legend([r'Original matrix $A$', r'Preconditioned matrix $M^{-1}A$'])
#    plt.show()
    
#    size = np.zeros(len(data_gap))
#    n_iter = np.zeros(len(data_gap))
#    n_iter_prec = np.zeros(len(data_gap))
#    
#    index = 0
#        
#    for i in data_gap:
#        A, b = ccore(clscale, mur, i/10000)
#        size[index] = len(A)
#        sA, iA, jA = CSRformat(A)
#        x, res, n_iter[index] = csrCG(sA, iA, jA, b, 1e-6, False, True)
#        x, res_prec, n_iter_prec[index] = csrCG(sA, iA, jA, b, 1e-6, True, True)
#        
#        index += 1
#        
#    plt.loglog(size, n_iter, '-bo', size, n_iter_prec, '-ro', size, size, '-go')
#    plt.xlabel(r"Size of the system, $n$ (influence of gap size)")
#    plt.ylabel(r"Number of iterations")
#    plt.title(r"Effect of system size"
#              "\n"
#              r"on the number of iterations needed for convergence")
#    plt.legend([r'Original matrix $A$', r'Preconditioned matrix $M^{-1}A$', 'Theoretical bound'])
#    plt.show()
        
    
    
if __name__ == '__main__':
    #spectrum_analysis(prec='report')
    convergence_analysis(prec='report')