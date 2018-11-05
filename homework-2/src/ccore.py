import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import gmsh
import sys

from mysolve import *

# LINMA1170 parameters 
clscale = 10   # mesh refinement 1:fine 10:coarse 50:very coarse
mur = 100.     # Relative magnetic permeability of region CORE 1:air 1000:steel
gap = 0.001     # air gap length in meters


DEBUG = False
import sys
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

    
# This script assembles and solves a simple static Laplacian problem
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

def solve(clscale):
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
    success, sol = mysolve(A, b, clscale)
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
    return



if __name__ == "__main__":
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
    
    solve(clscale)
    #gmsh.fltk.run()
    
# Solve linear system Ax=b




## Explanation for the serialization of elementary matrices.
##
## node = numpy.array([[1,2,3]]) # one element with nodes 1,2 and 3
## col = numpy.repeat(enode[:,:,None],3,axis=2)
## row = numpy.repeat(enode[:,None,:],3,axis=1)
## >>> col
## array([[[1, 1, 1],
##         [2, 2, 2],
##         [3, 3, 3]]])
## >>> row
## array([[[1, 2, 3],
##         [1, 2, 3],
##         [1, 2, 3]]])
