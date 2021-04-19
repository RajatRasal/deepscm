import numpy as np
from scipy import sparse
from .utils.vector import veclen, normalized
from .mesh import get_face_normals_and_areas
from .graph import get_edge_incidences, map_face_edges, reverse_orientation, orient_triangles, orient_edge_incidences
from .number.quaternion import conjugate as qu_conjugate
from .number.quaternion import matrix_representation as qu_matrix


def orient_outwards(verts, tris, FN_FA = None, P = None, E = None, debug = True):
    """
    Reorient tris so that normals point outwards.
    
    Compute the flux through the surface of the vector field obtained by normalizing vectors going
    from the center of gravity of the mesh to face barycenters. If positive, leave the orientation
    unchanged, otherwise revert.
    
    As usual, can use optional input FN_FA instead of recomputing it, or can pass on the computations as output
    """
    
    #
    o = np.mean(verts,axis=0)
    B = np.einsum('ijk->ik',verts[tris])/3 # verts[tris] is a 3d tensor (m,3-vertices,3-coordinates)    
    field = normalized(B-o)
    
    #
    if FN_FA:
        FN, FA = FN_FA        
    else:
        FN, FA = get_face_normals_and_areas(verts, tris, P, E)
    
    #
    flux = np.sum(field*FN)
    if debug:
        print(flux)
    if flux < 0:
        if debug:
            print("oriented inwards")
        tris = reverse_orientation(tris)
        
        # Reorient FN, P, E if needed
        if FN_FA is not None:
            FN *= -1           
        if P:
            P[1], P[2] = P[2], P[1]
        if E:
            E[0], E[1], E[2] = -E[0], -E[2], -E[1] # -E[2], -E[1], -E[0]
    
    #
    if FN_FA == []:
        FN_FA.extend([FN,FA])                   
    
    #            
    return tris
    

def compute_facenet(verts, tris, orient = True, P = None, E = None, debug = True, abscos_half_angles = None):
    """
    Compute the face edge-constraint net from the classical mesh. It is specified by the following outputs:
        [*] hyperedges HE: (o,4) float array
        [*] face normals FN: (m,4) float array
        [*] incident faces per edge IF: (o,2) int array
        [*] incident vertices per edge IV: (o,2) int array
        [*] face areas FA: (m) float array
        
    Optional outputs:
    P: face vertices [p1, p2, p3]
    E: face edge vectors [e1, e2, e3]
    
    P and E cannot be accepted as input here to avoid the headache if orient = True
    """  
    
    # Edge correctness checks, edge incidences, orientation / face normals and areas, make everything consistent
    IV, IF, NV = get_edge_incidences(tris)
    FE = map_face_edges(IF, IV, tris)    
    if orient:
        FN_FA = []
        tris = orient_triangles(IV, IF, tris, FE = [FE])        
        tris = orient_outwards(verts, tris, FN_FA = FN_FA, P = P, E = E, debug = debug)
        FN, FA = FN_FA
    else:
        FN, FA = get_face_normals_and_areas(verts, tris, P, E)            
    IV = orient_edge_incidences(IV, IF, FE, tris)
    
    #
    if debug:
        test1 = tris[IF[:,0]][:,:2] == IV
        test1 = test1[:,0] * test1[:,1]
        test2 = tris[IF[:,0]][:,1:3] == IV
        test2 = test2[:,0] * test2[:,1]
        test3 = tris[IF[:,0]][:,[2, 0]] == IV
        test3 = test3[:,0] * test3[:,1]       
        assert(np.all((test1+test2+test3)==1))
        
        test1 = tris[IF[:,1]][:,:2] == IV[:, [1, 0]]
        test1 = test1[:,0] * test1[:,1]
        test2 = tris[IF[:,1]][:,1:3] == IV[:, [1, 0]]
        test2 = test2[:,0] * test2[:,1]
        test3 = tris[IF[:,1]][:,[2, 0]] == IV[:, [1, 0]]
        test3 = test3[:,0] * test3[:,1]       
        assert(np.all((test1+test2+test3)==1))
        
        print('Edge combinatorics -- passed the test.')
    
    # To compute hyperedges we need the bending angles between faces, or equivalently the angle by which to rotate
    # ni for it to fall in the plane Pj
    # Well, we are working from a classical net (standard mesh) so at this stage face normals are actually
    # the vectors normal to the faces ;)
    ni = FN[IF[:,0],:]
    nj = FN[IF[:,1],:]
    
    #
    ni_nj = np.sum(ni*nj,axis=-1)
    ni_cross_nj = np.cross(ni, nj)
    eij = verts[IV[:,1],:]-verts[IV[:,0],:]
    
    sgn = 2*(np.sum(ni_cross_nj*eij,1)>=0)-1 # +1 if ni,nj,eij is direct; -1 otherwise 
    abscos_half_ij = 1+ni_nj # not done yet
    tan_half_ij = sgn * veclen(ni_cross_nj) / abscos_half_ij # tan x/2 = sin x / (1+cos x) 
    
    # Optional
    if abscos_half_angles==[]:
        numerics = abscos_half_ij<0
        abscos_half_ij[numerics] = 0
        abscos_half_ij[~numerics] = abscos_half_ij[~numerics]/2
        abscos_half_ij = np.sqrt(abscos_half_ij)
        abscos_half_angles.append(abscos_half_ij)
    
    # Wrap up the quaternionic hyperedges
    HE = np.c_[ veclen(eij)*tan_half_ij, eij ]
    
    if debug:
        m_ni = np.zeros((len(IF),4))
        m_ni[:,1:] = -ni
        
        test = np.einsum('ijk,ik->ij', qu_matrix(HE, conjugation=False, right_action=True), m_ni)
        test = np.einsum('ijk,ik->ij', qu_matrix(HE, conjugation=True, right_action=False), test)
        test = test[:,1:]
        test = test/np.sum(np.square(HE),axis=-1)[:,np.newaxis]
        adiff = np.abs(nj-test)
        print('Hyperedges -- max test discrepancy: ' + str(np.max(adiff)))
        assert( np.allclose(adiff, 0, atol=1e-6) )
        print('Hyperedges -- passed the test.')
    
    return HE, IV, IF, FN, FA, FE, len(verts) 


def compute_dirac_operator(HE, IF, FA, n, mean_curvature_target=None):
    """
    Intrinsic Dirac operator.
    
    mean_curvature_potential: (m,) array of target mean curvature (density) over each face
    """
    
    # Misc
    o = len(HE)
    m = o*2//3 
    
    # HE is correctly oriented for IF[:,0] and its conjugate is for IF[:,1]
    # In other words, HE[l,:] should be put at index (IF[l,0],IF[l,1]). 
    
    # Create row, col, and data in vector form for now
    row = np.r_[IF[:,0],IF[:,1]]
    col = np.r_[IF[:,1],IF[:,0]]   
    data = np.r_[HE,qu_conjugate(HE)]
    
    if mean_curvature_target is not None:
        row = np.r_[row,np.arange(m)]
        col = np.r_[col,np.arange(m)]
        potential = -mean_curvature_target*FA
        qu_potential = np.zeros((m,4),dtype=data.dtype) 
        qu_potential[:,0] = potential
        data = np.r_[data,qu_potential]
    
    data = qu_matrix(data)
        
    # Fancy pants (ugly pants)
    rows = row.reshape((len(row),1))*4 + np.arange(4).repeat(4)
    cols = col.reshape((len(col),1))*4 + np.broadcast_to(np.arange(4),(4,4)).reshape(-1)
    
    # Stuff
    Aii = FA.repeat(4)
    
    # Return
    D = sparse.csr_matrix((data.reshape(-1), (rows.reshape(-1),cols.reshape(-1))), shape=(m*4, m*4))
    A = sparse.diags(Aii, 0, dtype=D.dtype)
    #iA = sparse.diags(1/Aii, 0, dtype=D.dtype)
        
    return D, A