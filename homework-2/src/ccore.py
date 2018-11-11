# Homework 2 : SVD and condition number
# @author Gilles Peiffer, 24321600
# gilles.peiffer@student.uclouvain.be
# Nov. 12, 2018

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.image as image
import gmsh
import sys

from mysolve import *

"""
Wrapper around the ccore routine so we can call it multiple times in a loop
to generate nice looking graphs.

Some of the functions were altered to make receiving relevant values easier,
but this should not influence the overall idea of ccore.
"""
def ccore_wrapper(clscale, mur, gap, J):
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
    
        
    # This script assembles and solves a simple static Laplacian problem
    # using exclusively the Python api of Gmsh.
    
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
#    J = 1.e6         # Current density (A/m^2)
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
        success, sol, i = mysolve(A, b)
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
        return A, i

        
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
    
    A, i = solve()
    #gmsh.fltk.run()
    return A, i


def data_analysis(precision='report'):
    """
    This function creates all the plots and graphs used in the report
    
    The default precision argument is for report-quality graphs;
    if the precision parameter is set to 'test',
    it executes much quicker but doesn't give nice graphs.
    """
    # Default values
    clscale_def = 16 # Mesh coarseness
    gap_def = 0.001 # Reasonable value of the gap
    mur_def = 100. # Relative magnetic permeability of the magnetic core
    J_def = 1000000 # Current density
    
    ## Influence of clscale
    # Influence of clscale on the condition number of A
    clscale_condition = []
    clscale_par = 25
    if precision == 'test':
        clscale = range(15, 30)
    else:
        clscale = range(10, 51)
    
    for i in clscale:
         A, j = ccore_wrapper(i, mur_def, gap_def, J_def)
         U, s, Vh = np.linalg.svd(A)
         clscale_condition.append(np.linalg.cond(A))
         if i == clscale_def:
             clscale_spectrum1 = s
         elif i == clscale_par:
             clscale_spectrum2 = s
    
    # Plot scattered data points and perform a linear regression on them
    # in order to identify correlation
    
    # Calculate Pearson correlation coefficient and p-value
    if precision == 'test':
        clscale_condition_corr = pearsonr(clscale, clscale_condition)
        # Linear fit
        fit_clscale_condition = np.polyfit(clscale, clscale_condition, 1)
        fit_clscale_condition_fn = np.poly1d(fit_clscale_condition)
    else:
        vec1 = []
        vec1_cond = []
        vec2 = []
        vec2_cond = []
        for i in range(len(clscale)):
            if clscale[i] < 34:
                vec1.append(clscale[i])
                vec1_cond.append(clscale_condition[i])
            else:
                vec2.append(clscale[i])
                vec2_cond.append(clscale_condition[i])
                
        clscale_condition_corr1 = pearsonr(vec1, vec1_cond)
        clscale_condition_corr2 = pearsonr(vec2, vec2_cond)
        clscale_condition_corr = (clscale_condition_corr1[0] + clscale_condition_corr2[0])/2.
        fit_clscale_condition1 = np.polyfit(vec1, vec1_cond, 1)
        fit_clscale_condition2 = np.polyfit(vec2, vec2_cond, 1)
        fit_clscale_condition_fn1 = np.poly1d(fit_clscale_condition1)
        fit_clscale_condition_fn2 = np.poly1d(fit_clscale_condition2)
    
    # Plot data
    if precision == 'test':
        plt.plot(clscale, clscale_condition,'ro', clscale, fit_clscale_condition_fn(clscale), '--r')
    else:
        plt.plot(clscale, clscale_condition,'ro', vec1, fit_clscale_condition_fn1(vec1), '--r', vec2, fit_clscale_condition_fn2(vec2), '--r')
    plt.xlabel("Mesh coarseness [clscale]")
    plt.ylabel(r"$\kappa(A)$")
    plt.title(r"Influence of clscale on the condition number of $A$"
              "\n"
              "Pearson correlation coefficient : %0.3f" %clscale_condition_corr)
    plt.legend(["Data points", "Linear regression"])
    plt.show()

    clblue = plt.bar(range(len(clscale_spectrum1)), clscale_spectrum1, log=True, label=r"$\mathrm{clscale} = %i$" %clscale_def)
    clorange = plt.bar(range(len(clscale_spectrum2)), clscale_spectrum2, log=True, label=r"$\mathrm{clscale} = %i$" %clscale_par)
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value")
    plt.title(r"Singular value spectrum of $A$"
              "\n"
              r"$\mu_{\mathrm{r}} = %i$, $\mathrm{gap} = %.3f \mathrm{m}$, $J = %.0e \mathrm{A}/\mathrm{m}^2$" %(mur_def, gap_def, J_def))
    plt.legend([clblue, clorange], [r"$\mathrm{clscale} = %i$" %(clscale_def), r"$\mathrm{clscale} = %i$" %(clscale_par)])
    plt.show()
    
    ## Influence of relative permeability
    # Influence of the relative permeability on the condition number of A
    perm_condition = []
    perm_par = 820
    if precision == 'test':
        perm = range(10, 1000, 90)
    else:
        perm = range(10, 1000, 10)
    
    for i in perm:
         A, j = ccore_wrapper(clscale_def, i, gap_def, J_def)
         U, s, Vh = np.linalg.svd(A)
         perm_condition.append(np.linalg.cond(A))
         if i == mur_def:
             perm_spectrum1 = s
         elif i == perm_par:
             perm_spectrum2 = s
    
    # Plot scattered data points and perform a logarithmic fit on them
    # in order to identify correlation
    
    # Logarithmic fit
    def perm_fit(x, a, b, c):
        return a+b*np.log10(x + c)
    
    fit_perm_condition, fit_perm_cov = scipy.optimize.curve_fit(perm_fit, perm, perm_condition)
    print(fit_perm_condition)
    
    # Calculate Pearson correlation coefficient and p-value
    perm_condition_corr = pearsonr(perm, fit_perm_condition[0] + fit_perm_condition[1] * np.log10(perm_condition + fit_perm_condition[2]))

    # Plot data
    plt.plot(perm, perm_condition,'bo', range(10, 1000), perm_fit(range(10, 1000), *fit_perm_condition), '--b')
    plt.xlabel("Relative permeability of magnetic core")
    plt.ylabel(r"$\kappa(A)$")
    plt.title(r"Influence of permeability on the condition number of $A$"
              "\n"
              "Pearson correlation coefficient : %0.3f" %perm_condition_corr[0])
    plt.legend(["Data points", "Logarithmic fit"])
    plt.show()

    permblue = plt.bar(range(len(perm_spectrum1)), perm_spectrum1, log=True)
    permorange = plt.bar(range(len(perm_spectrum2)), perm_spectrum2, log=True)
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value")
    plt.title(r"Singular value spectrum of $A$"
              "\n"
              r"$\mathrm{clscale} = %i$, $\mathrm{gap} = %.3f \mathrm{m}$, $J = %.0e \mathrm{A}/\mathrm{m}^2$" %(clscale_def, gap_def, J_def))
    plt.legend([permblue, permorange], [r"$\mu_{\mathrm{r}} = %i$" %(mur_def), r"$\mu_{\mathrm{r}} = %i$" %(perm_par)])
    plt.show()
    
    ## Influence of gap
    # Influence of the gap on the condition number of A
    gap_condition = []
    gap_par = 0.011
    if precision == 'test':
        gap = np.divide(range(1, 101, 10), 1000.)
    else:
        gap = np.divide(range(1, 101), 1000.)
    
    for i in gap:
         A, j = ccore_wrapper(clscale_def, mur_def, i, J_def)
         U, s, Vh = np.linalg.svd(A)
         gap_condition.append(np.linalg.cond(A))
         if i == gap_def:
             gap_spectrum1 = s
         elif i == gap_par:
             gap_spectrum2 = s
    
    # Plot scattered data points and perform an inverse linear regression on them
    # in order to identify correlation
    
    # Inverse linear fit
    def gap_fit(x, a, b):
        return a+b/x
    
    fit_gap_condition, fit_gap_cov = scipy.optimize.curve_fit(gap_fit, gap, gap_condition)
    
    # Calculate Pearson correlation coefficient and p-value
    gap_condition_corr = pearsonr(gap, fit_gap_condition[0] + np.divide(fit_gap_condition[1], gap_condition))

    # Plot data
    plt.plot(gap, gap_condition,'go', np.divide(range(1000, 100000, 1), 1000000.), gap_fit(np.divide(range(1000, 100000, 1), 1000000.), *fit_gap_condition), '--g')
    plt.xlabel(r"Gap size [$\mathrm{m}$]")
    plt.ylabel(r"$\kappa(A)$")
    plt.title(r"Influence of gap size on the condition number of $A$"
              "\n"
              "Pearson correlation coefficient : %0.3f" %gap_condition_corr[0])
    plt.legend(["Data points", "Inversely linear fit"])
    plt.show()

    gapblue = plt.bar(range(len(gap_spectrum1)), gap_spectrum1, log=True)
    gaporange = plt.bar(range(len(gap_spectrum2)), gap_spectrum2, log=True)
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value")
    plt.title(r"Singular value spectrum of $A$"
              "\n"
              r"$\mathrm{clscale} = %i$, $\mu_\mathrm{r} = %i$, $J = %.0e \mathrm{A}/\mathrm{m}^2$" %(clscale_def, mur_def, J_def))
    plt.legend([gapblue, gaporange], [r"$\mathrm{gap} = %.3f \mathrm{m}$" %(gap_def), r"$\mathrm{gap} = %.3f \mathrm{m}$" %(gap_par)])
    plt.show()
    
    
    ## Influence of current density
    # Influence of the current density on the condition number of A
    J_condition = []
    J_par = 100000
    if precision == 'test':
        J = range(100000, 1000001, 100000)
    else:
        J = range(100000, 10000000, 900000)
    
    for i in J:
         A, j = ccore_wrapper(clscale_def, mur_def, gap_def, i)
         U, s, Vh = np.linalg.svd(A)
         J_condition.append(np.linalg.cond(A))
         if i == J_def:
             J_spectrum1 = s
         elif i == J_par:
             J_spectrum2 = s
    
    # Plot scattered data points and perform a linear regression on them
    # in order to identify correlation
    
    # Linear fit
    fit_J_condition = np.polyfit(J, J_condition, 1)
    fit_J_condition_fn = np.poly1d(fit_J_condition)

    # Plot data
    plt.plot(J, J_condition,'mo', J, fit_J_condition_fn(J), '--m')
    plt.xlabel(r"Current density [$\mathrm{A}/\mathrm{m}^2$]")
    plt.ylabel(r"$\kappa(A)$")
    plt.title(r"Influence of current density on the condition number of $A$"
              "\n"
              r"Pearson correlation coefficient undefined ($\kappa(A)$ is constant)")
    plt.legend(["Data points", "Linear fit"])
    plt.show()

    Jblue = plt.bar(range(len(J_spectrum1)), J_spectrum1, log=True)
    Jorange = plt.bar(range(len(J_spectrum2)), J_spectrum2, log=True)
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value")
    plt.title(r"Singular value spectrum of $A$"
              "\n"
              r"$\mathrm{clscale} = %i$, $\mu_\mathrm{r} = %i$, $\mathrm{gap} = %.3f \mathrm{m}$" %(clscale_def, mur_def, gap_def))
    plt.legend([Jblue, Jorange], [r"$J = %.0e \mathrm{A}/\mathrm{m}^2$" %(J_def), r"$J = %.0e \mathrm{A}/\mathrm{m}^2$" %(J_par)])
    plt.show()
    
if __name__ == "__main__":
    data_analysis('report')