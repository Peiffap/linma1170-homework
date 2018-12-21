# The function mysolve(A, b) is invoked by ccore.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ccore.py -clscale float
# where the line argument 'clscale' allows the global coarsening of the mesh
# -clscale 1  fine mesh
# -clscale 10 coarse mesh


SolverType = 'scipy'

import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import timeit
import sys
import gmsh

import numpy as np

def mysolve(A, b):
    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b)
    #elif SolverType == 'QR':
        # write here your code for the QR solver
    elif SolverType == 'LU':
        return True, LUsolve(A, b)
        # write here your code for the LU solver
    #elif SolverType == 'GMRES':
        # write here your code for the LU solver
    else:
        return False, 0
    
def CSRformat(A):
    n,m =A.shape
    sA= []
    iA= [0]
    jA= []
    count=0
    for i in range(n):
        for j in range(m):
            if A[i][j]!=0:
                sA += [A[i][j]]
                jA += [j]
                count +=1
        iA += [count]
        
    return (np.array(sA,dtype=np.float64),np.array(iA,dtype=int),np.array(jA,dtype=int))

def csrCG(sA,iA,jA,b,rtol,prec,it=False):
    n=len(b)
    x=0
    k=0
    if prec == False:
        r=np.copy(b)
        p= np.copy(r)
        residu=[]
        while np.linalg.norm(r)/np.linalg.norm(b) > rtol:
            residu.append(np.linalg.norm(r))
            old= np.copy(r)
            Ap= product_CSR(sA,iA,jA,p)
            alpha=np.dot(r,r)/np.dot(p,Ap)
            x+=alpha*p
            r-=alpha*Ap
            betha= np.dot(r,r)/np.dot(old,old)
            p = r + betha*p
            k+=1
    else:
        sM,iM,jM= ILU0(sA,iA,jA,remove=False)
        r=np.copy(b)
        rtild=CSRsolve(sM,iM,jM,r)
        p=np.copy(rtild)
        residu=[]
        while np.linalg.norm(r)/np.linalg.norm(b) > rtol:
            residu.append(np.linalg.norm(r))
            old= np.copy(r)
            oldtild=np.copy(rtild)
            Ap= product_CSR(sA,iA,jA,p)
            alpha=np.dot(old,oldtild)/np.dot(p,Ap)
            x+=alpha*p
            r= old - alpha*Ap
            rtild=CSRsolve(sM,iM,jM,r)
            betha= np.dot(r,rtild)/np.dot(old,oldtild)
            p = rtild + betha*p 
            k+=1
            
        #print(k)
    if it:
        return x,residu,k
    else:
        return x,r

        
def product_CSR(sA,iA,jA,p):
    n=len(iA)-1
    product= np.zeros(n)
    for i in range(n):
        for j in range(iA[i],iA[i+1]):
            product[i] += sA[j]*p[jA[j]]
    return product



                   
def ILUtest(B):
    A = np.copy(B)
    n=len(A)
    for i in range(1,n):
        for k in range(i):
            if A[i,k]!=0:
                A[i,k]/=A[k,k]
                for j in range(k+1,n):
                    if A[i,j]!=0:
                        A[i,j]-=A[i,k]*A[k,j]
    return A


    
def GAUCHE_DROITE(sA, iA, jA):
    n=len(iA)-1
    right=1
    left=1
    for i in range(n):
        curr_right=jA[iA[i+1]-1]-i+1
        if curr_right>right:
            right=curr_right
            
        curr_left=i-jA[iA[i]]+1
        if curr_left>left:
            left=curr_left
    return left,right

def ILU0(sA,iA,jA,remove=True):
    
    left,right=GAUCHE_DROITE(sA,iA,jA)
    bandwidth=left+right-1
    n=len(iA)-1
    def index(i,j):
        r=i+right
        l=i-left
        if j >= r or j <= l:
            return 0
        band=i*bandwidth
        index=band-i+j
        nl = ((left-2)*(left-1))//2
        if left-2-i > 0:
            nl -= (left-2-i)*(left-1-i)//2
        nr = 0
        if r > n:
            delta1=r-n
            delta = delta1 - 1
            nr = delta*(delta1)//2
        elem=index-nl-nr
        return elem


    nleft = n - left
    nright = n - right
    size=int(n*n-(nleft*(nleft+1)+(nright*(nright+1)))/2)
    sLU=np.zeros(size)
    jLU=np.zeros(size,dtype=int)
    iLU = np.copy(iA)
    curr_index=0
    for i in range(n):
        for j in range(iLU[i],iLU[i+1]):
            ind=index(i,jA[j]) #index dans sA de l'élément à la position i,jA[j] dans la vraie matrice
            sLU[ind]=sA[j]
        iLU[i]=curr_index
        if i >= left and i + right <= n:
            jLU[curr_index:curr_index+bandwidth]= range(i-left+1,i+right)
            curr_index+=bandwidth
        elif i<= left and i+right<=n:
            part=right+i
            jLU[curr_index:curr_index+part]= range(0,part)
            curr_index+=part
        elif i>=left and i+right>=n:
            part=left+n-i-1
            jLU[curr_index:curr_index+part]= range(n-part,n)
            curr_index+=part
        else:
            jLU[curr_index:curr_index+n]=range(n)
            curr_index+=n
    iLU[-1]=size
    def element(i,j):
        r=i+right
        l=i-left
        if j >= r or j <= l:
            return 0
        band=i*bandwidth
        index=band-i+j
        nl = ((left-2)*(left-1))//2
        if left-2-i > 0:
            nl -= (left-2-i)*(left-1-i)//2
        nr = 0
        if r > n:
            delta1=r-n
            delta = delta1 - 1
            nr = delta*(delta1)//2
        elem=index-nl-nr
        return sLU[elem]

    for i in range(1,n):
        for l in range(iLU[i], int(index(i,i))):
            if sLU[l] !=0:
                k = jLU[l]
                sLU[l]=sLU[l]/element(k,k)
                for jj in range(l+1, iLU[i+1]):
                    if sLU[jj]!=0:
                        j = jLU[jj]
                        sLU[jj] -= sLU[l]*element(k,j)
    if remove == True:                                
        zeros = np.where(abs(sLU) < 3e-16)[0]
        if len(zeros) > 0:                  
            count=0
            line=1
            for i in range(size-1):
                if abs(sLU[i])>3e-16:
                    count+=1
                if jLU[i]>jLU[i+1]:
                    iLU[line]=count
                    line+=1
            iLU[-1]=count+1
            sLU = np.delete(sLU, zeros)
            jLU = np.delete(jLU, zeros)
    return sLU.T,iLU,jLU.T


def CSRsolve(sA,iA,jA,b):
    n=len(iA)-1
    right = iA[1] - iA[0]
    left = iA[-1] - iA[-2]
    bandwidth = left+right-1
    
    def element(i,j):
        r=i+right
        l=i-left
        if j >= r or j <= l:
            return 0
        band=i*bandwidth
        index=band-i+j
        nl = ((left-2)*(left-1))//2
        nr = 0
        if left-2-i > 0:
            nl -= (left-2-i)*(left-1-i)//2
        if r > n:
            delta1=r-n
            delta = delta1 - 1
            nr = delta*(delta1)//2
        elem=index-nl-nr
        return sA[elem]
    
    def ForwardSubCSR(b):
        y = np.zeros(n,dtype=np.float64)
        y[0]=b[0]
        for i in range(1,n):
            prod=0
            for k in range(i):
                prod+=element(i,k)*y[k]
                y[i] = b[i] - prod
        return y

    def BackwardSubCSR(y):
        x = np.zeros(n,dtype=np.float64)
        x[n-1] = y[n-1]/element(n-1, n-1)
        for i in range(n-2,-1,-1):
            diag = element(i,i)
            prod=0
            for k in range(i+1, n):
                prod+=element(i,k)*x[k]
                x[i] = (y[i] -prod)/diag
        return x
    
    y = ForwardSubCSR(b)
    x = BackwardSubCSR(y)

    return x.T



def ccore(clscale,mur,gap):
    #clscale=10# mesh refinement 1:fine 10:coarse 50:very coarse
    #mur=1000# Relative magnetic permeability of region CORE 1:air 1000:steel
    #gap = 0.001     # air gap lenght in meter
    
    
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

def RE(sA,iA,jA):
    # Fonction qui prend en argument une matrice en format CSR, et qui la
    # renvoie en format plein.
    n = len(iA)-1
    A = np.zeros((n,n))
    for i in range(n):
        A[i][jA[iA[i]:iA[i+1]]] = sA[iA[i]:iA[i+1]]
    return A

def Exp_cond():
    condM=[]
    condA=[]
    size=[]
    Res=[]
    ResP=[]
    K1=[]
    K2=[]
    clscale=10
    mur=1000
#    gap=0.001
    rtol=1e-6
    
    for i in range(5,50,1):
        i=i/10000
        A, b = ccore(clscale,mur,i)
        n=len(A)
        size.append(i)
        
        B=np.copy(A)
        sA,iA,jA=CSRformat(B)
        
        sM,iM,jM= ILU0(sA,iA,jA)
        M =RE(sM,iM,jM)
        
        Ltest = np.eye(len(M)) + np.tril(M,-1)
        Utest = np.triu(M)
        inv=np.linalg.inv(Ltest @ Utest)
        MA=inv @ B
        condM.append(np.linalg.cond(MA))
        condA.append(np.linalg.cond(B))
#        Ab =np.linalg.eigvals(B)
#        Bc =np.linalg.eigvals(MA)
#        maxi=max(Bc)/min(Bc)
#        mini=max(Ab)/min(Ab)


    #plots des différents conditionnements 
    ##CLSCALE 10-30
#    plt.plot(size,condA, 'bo', label='Non-préconditionné')
#    plt.plot(size,condM,'ro', label='Préconditionné')
#    plt.legend()
#    plt.ylabel('Nombre de conditionnement')
#    plt.xlabel('Taille de la matrice')
#    plt.title("Nombre de conditionnement \n en fonction de la taille de la matrice" )
#    plt.show()
    ##MUR 1,1000,10
#    plt.plot(size,condA, 'bo', label='Non-préconditionné')
#    plt.plot(size,condM,'ro', label='Préconditionné')
#    plt.legend()
#    plt.ylabel('Nombre de conditionnement')
#    plt.xlabel('Perméabilité relative')
#    plt.title("Nombre de conditionnement \n en fonction de la perméabilité relative" )
#    plt.show()
    ##GAP
    plt.plot(size,condA, 'bo', label='Non-préconditionné')
    plt.plot(size,condM,'ro', label='Préconditionné')
    plt.legend()
    plt.ylabel('Nombre de conditionnement')
    plt.xlabel("largeur de l'entrefer [m]")
    plt.title("Nombre de conditionnement \n en fonction de la largeur de l'entrefer" )
    plt.show()
 #clscale constant et mur constant   
#    plt.semilogy(range(k1),Res[0],'bo')
#    plt.semilogy(range(k2),ResP[0],'ro')
#    plt.show()
 
#    plt.plot(np.log(range(k1)),np.log(Res[0]),'bo')
#    plt.plot(np.log(range(k2)),np.log(ResP[0]),'ro')
#    plt.show()
    
#    plt.plot(range(k1),Res[0],'b')
#    plt.plot(range(k2),ResP[0],'r')
#    plt.show()
 
#    plt.plot(np.log2(condA),np.log2(K1),'bo')
#    plt.plot(np.log10(condM),np.log10(K2),'ro')
#    plt.show()
#    
#    plt.bar(range(size[0]),sorted(Ab),log=True)
#    plt.bar(range(size[0]),sorted(Bc),log=True)
#    plt.show()
if __name__ == "__main__":
    Exp_cond()
#    A,b= ccore(20)
#    sA,iA,jA= CSRformat(A)
#    x,r = csrCG(sA,iA,jA,b,1e-6,True)
#    print(x)
#    print(r)
#    A=np.array([[5,1,2],[1,3,2],[2,2,4]], dtype=np.float64)
#    b=np.array([1,2,3], dtype=np.float64)

#    LU = ILUtest(A)
#    sAA,iAA,jAA=CSRformat(LU)
##    print(sAA,iAA,jAA)
#    #print(sA,iA,jA)
#    sLU,iLU,jLU =ILU0(sA,iA,jA,remove=False)    
#    print(sLU,iLU,jLU)
##    print(np.allclose(sAA,sLU,atol=1e-5,rtol=1e-5))
#    x = CSRsolve(sLU,iLU,jLU,b)
#    print(x)
#    print(np.allclose(x,np.scipy.linalg.solve(A,b),atol=1e-5,rtol=1e-5))

    

