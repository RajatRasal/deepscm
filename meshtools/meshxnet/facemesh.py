# File for stuff that doesn't fit strictly either in mesh or net :o)
import numpy as np
from scipy import sparse

from .graph import get_face_edge_signs, curl_adjoint_operator
from .mesh import get_cell_normals
from .utils.vector import veclen, sq_norm
from .number.quaternion import conjugate as qu_conjugate
from .number.quaternion import matrix_representation as qu_matrix

    
def compute_mesh_laplacian(facenet, tris, FE_sign=None, E=None, debug=False, negative_eigenvalues=True):
    """
    Copy from mesh_utils :/ plus some stuff
    """
    
    # Misc
    m = len(tris) 
    HE, IV, IF, FN, FA, FE, n = facenet
    o = len(HE)
            
    # Shortcut
    i1 = tris[:,0]
    i2 = tris[:,1]
    i3 = tris[:,2]
    
    # to reorient if needed   
    if FE_sign:
        switch = FE_sign[0]
    else:
        switch = get_face_edge_signs(FE, IF)        
        if FE_sign == []:
            FE_sign.append(switch)       
    
    edges = np.einsum('ijk,ij->ijk',HE[:,1:][FE],switch) # 1st dim: face, 2nd: edge, 3rd: coordinates
    e1 = edges[:,0,:]
    e2 = edges[:,1,:]
    e3 = edges[:,2,:]
    
    if E==[]:
        E.extend([e1,e2,e3])
    
    #
    triangle_area = FA
    area_over_3 = triangle_area/3
    area_over_12 = triangle_area/12
    
    # Compute the area of each vertex 1-ring, from the face contributions.
    # Return a face to vertex 1-ring interpolator, based on the contribution of each face to the 1-ring
    face_number = np.arange(m)    
    FtR_kl = np.r_[triangle_area, triangle_area, triangle_area]
    K = np.r_[i1, i2, i3]
    L = np.r_[face_number, face_number, face_number]        
    FtR = sparse.csr_matrix((FtR_kl, (K, L)), shape=(n, m))
    
    # Our vertex areas
    A1r = FtR.dot(np.ones(m))
        
    # Normalise rows to 1   
    normalisation1r = 1/A1r
    FtR.data *= normalisation1r.repeat(np.diff(FtR.indptr))
    
    # Squared edge lengths
    e1_e1 = np.sum(e1*e1,1)
    e2_e2 = np.sum(e2*e2,1)
    e3_e3 = np.sum(e3*e3,1)
    
    # Edge inner products
    e1_e2 = (e1*e2).sum(axis=1)
    e2_e3 = (e2*e3).sum(axis=1)
    e3_e1 = (e1*e3).sum(axis=1)   
    
    # Compute cotan of all combinations
    cotan_weight_12 = -e1_e2 / veclen(np.cross(e1, e2))
    cotan_weight_23 = -e2_e3 / veclen(np.cross(e2, e3))
    cotan_weight_31 = -e3_e1 / veclen(np.cross(e3, e1))  
    
    # Compute the mass matrix M
    m_off = area_over_12
    m_diag = m_off*2
    
    M_ij = np.r_[m_diag, m_diag, m_diag, m_off, m_off, m_off, m_off, m_off, m_off]
    I = np.r_[i1, i2, i3, i1, i1, i2, i2, i3, i3]
    J = np.r_[i1, i2, i3, i2, i3, i3, i1, i1, i2] 
    
    M = sparse.csc_matrix((M_ij, (I, J)), shape=(n, n))
    
    # Debug only
    if debug:
        area1r_3 = A1r/3        
        assert np.allclose(M.dot(np.ones(n)), area1r_3), 'M did not pass the unit test.'
    
    # Compute the xi_i's, where xi_i = cot(j)cot(k) = (cot(j)+cot(k))/(tan(j)+tan(k)) = 1 - 2*sin(2i)/sum_l(sin(2l))
    # and bar_i(circumcenter) = (1-xi_i)/2 
    xi_1 = cotan_weight_31*cotan_weight_12
    xi_2 = cotan_weight_12*cotan_weight_23
    xi_3 = cotan_weight_23*cotan_weight_31
    
    # Compute the nu_ip's, where nu_ip = e_ip^2 / e_ij^t e_ik; i != p, i and p do not have symmetric roles here
    # Note that, nu_ip = (e_ip^2*tan(p))/(2*A_T)
    # If T is obtuse at i, bar_j(b) = 0.5*nu_ki and bar_j(a) = 1-0.5*nu_ji, bar_k(a) = 0.5*nu_ji and bar_k(b) = 1-0.5*nu_ki
    # with a the intersect of the height at m=(i+j)/2 with the segment jk, and b the intersect the height at n=(i+k)/2 with jk
    # In general, the orthogonal projection o' of o=(j+k)/2 on ij is such that jm/jo' = nu_ji (= ja/jo in the obtuse at i 
    # case). The link between both equations is an application of Thales theorem.
    # Moreover, the area of joo' is 1/4 of A_T/nu_ji (or A_jmo/nu_ji). In the obtuse case (still valid in the acute case but 
    # not as useful), A_jma = A_jmo*nu_ji as well. We use those a lot because every int_{A_M inter T} B_l ends up being
    # written as A_{some inner triangle}*constant_factor or a sum of such terms.
    inu_12 = -e2_e3/e3_e3
    inu_21 = -e3_e1/e3_e3
    inu_13 = -e2_e3/e2_e2
    inu_31 = -e1_e2/e2_e2
    inu_23 = -e3_e1/e1_e1
    inu_32 = -e1_e2/e1_e1
    
    # Now we're going to compute all coefficients of the cell integral matrix B, treating obtuse and acute triangles 
    # differently
    # While at it, we record the contribution of each face to their vertices' cell in terms of area
    B_ij = [] 
    I = []
    J = []
    
    a1 = np.zeros(m)
    a2 = np.zeros(m)
    a3 = np.zeros(m)
        
    # a. Are triangles obtuse at one of their vertices?
    # We'll use boolean indexing afterwards e.g., array[~obtuse_triangle]
    obtuse_at_1 = e2_e3 > 0
    obtuse_at_2 = e3_e1 > 0
    obtuse_at_3 = e1_e2 > 0 
    acute_triangle = ~(obtuse_at_1 + obtuse_at_2 + obtuse_at_3 > 0)
    
    # b. Acute triangles
    area_a = triangle_area[acute_triangle]
    
    if len(area_a):
        i1_a = i1[acute_triangle]
        i2_a = i2[acute_triangle]
        i3_a = i3[acute_triangle]

        xi1_a = xi_1[acute_triangle]
        xi2_a = xi_2[acute_triangle]
        xi3_a = xi_3[acute_triangle]

        inu12_a = inu_12[acute_triangle]
        inu21_a = inu_21[acute_triangle]
        inu13_a = inu_13[acute_triangle]
        inu31_a = inu_31[acute_triangle]
        inu23_a = inu_23[acute_triangle]
        inu32_a = inu_32[acute_triangle]
        
        # Barycentric coordinates of the circumcenter
        bO_1 = (1-xi1_a)/2
        bO_2 = (1-xi2_a)/2
        bO_3 = (1-xi3_a)/2
        
        if debug:
            # Check that the barycentric coordinates sum to 1
            assert np.allclose(bO_1+bO_2+bO_3, np.ones(len(bO_1))), \
                "Barycentric coordinates of the circumcenter do not sum to 1."
        
        # Useful quantities
        bOp1_1 = 1+bO_1
        bOp1_2 = 1+bO_2
        bOp1_3 = 1+bO_3
        
        q11 = bOp1_1*xi1_a
        q21 = bOp1_1*xi2_a
        q31 = bOp1_1*xi3_a
        q12 = bOp1_2*xi1_a
        q22 = bOp1_2*xi2_a
        q32 = bOp1_2*xi3_a
        q13 = bOp1_3*xi1_a
        q23 = bOp1_3*xi2_a
        q33 = bOp1_3*xi3_a
        
        area_over_12_a = area_over_12[acute_triangle]

        # Cell areas   
        a1[acute_triangle] = (1+xi1_a)*area_a/4
        a2[acute_triangle] = (1+xi2_a)*area_a/4
        a3[acute_triangle] = (1+xi3_a)*area_a/4
        
        # Debug only
        if debug:
            meyer_a1_a = e3_e3[acute_triangle]*cotan_weight_12[acute_triangle] + \
                                    e2_e2[acute_triangle]*cotan_weight_31[acute_triangle]
            meyer_a2_a = e1_e1[acute_triangle]*cotan_weight_23[acute_triangle] + \
                                    e3_e3[acute_triangle]*cotan_weight_12[acute_triangle]
            meyer_a3_a = e2_e2[acute_triangle]*cotan_weight_31[acute_triangle] + \
                                    e1_e1[acute_triangle]*cotan_weight_23[acute_triangle] 
            assert np.allclose(a1[acute_triangle], meyer_a1_a/8), "Voronoi areas (acute case) did not pass the unit test"
            assert np.allclose(a2[acute_triangle], meyer_a2_a/8), "Voronoi areas (acute case) did not pass the unit test"
            assert np.allclose(a3[acute_triangle], meyer_a3_a/8), "Voronoi areas (acute case) did not pass the unit test"

        # Main contributions: vertex i over cell i
        
        B_ij.extend([(q11+2)*area_over_12_a, (q22+2)*area_over_12_a, (q33+2)*area_over_12_a])  
        I.extend([i1_a, i2_a, i3_a])
        J.extend([i1_a, i2_a, i3_a])

        # Secondary contributions: vertices j and k over cell i
        # Note that 1/nu_pq + 1/nu_qp = 1, and sum_i bO_i = 1, so that b_1i+b_2i+b_3i = triangle_area/3 as expected.

        #    i = 1st vertex, j = 2nd, k = 3rd
        b_ik = ( 0.5 + q13 - bO_3*inu21_a ) * area_over_12_a
        b_ij = ( 0.5 + q12 - bO_2*inu31_a ) * area_over_12_a
        B_ij.extend([b_ik, b_ij])
        I.extend([i1_a, i1_a])
        J.extend([i3_a, i2_a])

        #    i = 2nd vertex, j = 1st, k = 3rd
        b_ik = ( 0.5 + q23 - bO_3*inu12_a ) * area_over_12_a
        b_ij = ( 0.5 + q21 - bO_1*inu32_a ) * area_over_12_a
        B_ij.extend([b_ik, b_ij])
        I.extend([i2_a, i2_a])
        J.extend([i3_a, i1_a])

        #    i = 3rd vertex, j = 1st, k = 2nd
        b_ik = ( 0.5 + q32 - bO_2*inu13_a ) * area_over_12_a
        b_ij = ( 0.5 + q31 - bO_1*inu23_a ) * area_over_12_a
        B_ij.extend([b_ik, b_ij])
        I.extend([i3_a, i3_a])
        J.extend([i2_a, i1_a])
    
    # c. Obtuse triangles at first vertex
    area_o1 = triangle_area[obtuse_at_1] 
    
    if len(area_o1):
        i1_o1 = i1[obtuse_at_1]
        i2_o1 = i2[obtuse_at_1]
        i3_o1 = i3[obtuse_at_1]

        nu21_o1 = 1/inu_21[obtuse_at_1]
        nu31_o1 = 1/inu_31[obtuse_at_1]
        
        area_over_3_o1 = area_over_3[obtuse_at_1]
        
        # Barycentric coordinates, note that bar_k(a) = 1 - bar_j(a) since a lies on jk. Same for b.
        bBj = 0.5*nu31_o1
        bAk = 0.5*nu21_o1
        bBk = 1 - bBj
        bAj = 1 - bAk

        # Areas        
        a2[obtuse_at_1] = nu21_o1*area_o1/4
        a3[obtuse_at_1] = nu31_o1*area_o1/4
        a1[obtuse_at_1] = area_o1 - a2[obtuse_at_1] - a3[obtuse_at_1]
        
        # Vertex i=1 over cells i=1, j=2, k=3
        b_ji = a2[obtuse_at_1]/6
        b_ki = a3[obtuse_at_1]/6
        b_ii = area_over_3_o1 - b_ji - b_ki
        
        # Vertex j=2 over cells i, j, k
        b_kj = bBj*a3[obtuse_at_1]/3
        b_jj = (1.5 + bAj) * a2[obtuse_at_1]/3
        b_ij = area_over_3_o1 - b_jj - b_kj
        
        # Vertex k=3 over cells i, j, k
        b_jk = bAk*a2[obtuse_at_1]/3
        b_kk = (1.5 + bBk) * a3[obtuse_at_1]/3
        b_ik = area_over_3_o1 - b_jk - b_kk
        
        # Add to B
        B_ij.extend([b_ii, b_ij, b_ik, b_ji, b_jj, b_jk, b_ki, b_kj, b_kk])
        I.extend([i1_o1, i1_o1, i1_o1, i2_o1, i2_o1, i2_o1, i3_o1, i3_o1, i3_o1])
        J.extend([i1_o1, i2_o1, i3_o1, i1_o1, i2_o1, i3_o1, i1_o1, i2_o1, i3_o1])
    
    # d. Obtuse triangles at second vertex
    area_o2 = triangle_area[obtuse_at_2]
   
    if len(area_o2):
        i1_o2 = i1[obtuse_at_2]
        i2_o2 = i2[obtuse_at_2]
        i3_o2 = i3[obtuse_at_2]

        nu12_o2 = 1/inu_12[obtuse_at_2]
        nu32_o2 = 1/inu_32[obtuse_at_2]
        
        area_over_3_o2 = area_over_3[obtuse_at_2]
        
        # Barycentric coordinates, note that bar_k(a) = 1 - bar_j(a) since a lies on jk. Same for b.
        bBj = 0.5*nu32_o2
        bAk = 0.5*nu12_o2
        bBk = 1 - bBj
        bAj = 1 - bAk

        # Areas        
        a1[obtuse_at_2] = nu12_o2*area_o2/4
        a3[obtuse_at_2] = nu32_o2*area_o2/4
        a2[obtuse_at_2] = area_o2 - a1[obtuse_at_2] - a3[obtuse_at_2]
        
        # Vertex i=2 over cells i=2, j=1, k=3
        b_ji = a1[obtuse_at_2]/6
        b_ki = a3[obtuse_at_2]/6
        b_ii = area_over_3_o2 - b_ji - b_ki
        
        # Vertex j=1 over cells i, j, k
        b_kj = bBj*a3[obtuse_at_2]/3
        b_jj = (1.5 + bAj) * a1[obtuse_at_2]/3
        b_ij = area_over_3_o2 - b_jj - b_kj
        
        # Vertex k=3 over cells i, j, k
        b_jk = bAk*a1[obtuse_at_2]/3
        b_kk = (1.5 + bBk) * a3[obtuse_at_2]/3
        b_ik = area_over_3_o2 - b_jk - b_kk
        
        # Add to B
        B_ij.extend([b_ii, b_ij, b_ik, b_ji, b_jj, b_jk, b_ki, b_kj, b_kk])
        I.extend([i2_o2, i2_o2, i2_o2, i1_o2, i1_o2, i1_o2, i3_o2, i3_o2, i3_o2])
        J.extend([i2_o2, i1_o2, i3_o2, i2_o2, i1_o2, i3_o2, i2_o2, i1_o2, i3_o2])

    # e. Obtuse triangles at third vertex
    area_o3 = triangle_area[obtuse_at_3]
    
    if len(area_o3):
        i1_o3 = i1[obtuse_at_3]
        i2_o3 = i2[obtuse_at_3]
        i3_o3 = i3[obtuse_at_3]

        nu13_o3 = 1/inu_13[obtuse_at_3]
        nu23_o3 = 1/inu_23[obtuse_at_3]
        
        area_over_3_o3 = area_over_3[obtuse_at_3]
        
        # Barycentric coordinates, note that bar_k(a) = 1 - bar_j(a) since a lies on jk. Same for b.
        bBj = 0.5*nu13_o3
        bAk = 0.5*nu23_o3
        bBk = 1 - bBj
        bAj = 1 - bAk

        # Areas        
        a2[obtuse_at_3] = nu23_o3*area_o3/4
        a1[obtuse_at_3] = nu13_o3*area_o3/4
        a3[obtuse_at_3] = area_o3 - a2[obtuse_at_3] - a1[obtuse_at_3]
        
        # Vertex i=3 over cells i=3, j=2, k=1
        b_ji = a2[obtuse_at_3]/6
        b_ki = a1[obtuse_at_3]/6
        b_ii = area_over_3_o3 - b_ji - b_ki
        
        # Vertex j=2 over cells i, j, k
        b_kj = bBj*a1[obtuse_at_3]/3
        b_jj = (1.5 + bAj) * a2[obtuse_at_3]/3
        b_ij = area_over_3_o3 - b_jj - b_kj
        
        # Vertex k=1 over cells i, j, k
        b_jk = bAk*a2[obtuse_at_3]/3
        b_kk = (1.5 + bBk) * a1[obtuse_at_3]/3
        b_ik = area_over_3_o3 - b_jk - b_kk
        
        # Add to B
        B_ij.extend([b_ii, b_ij, b_ik, b_ji, b_jj, b_jk, b_ki, b_kj, b_kk])
        I.extend([i3_o3, i3_o3, i3_o3, i2_o3, i2_o3, i2_o3, i1_o3, i1_o3, i1_o3])
        J.extend([i3_o3, i2_o3, i1_o3, i3_o3, i2_o3, i1_o3, i3_o3, i2_o3, i1_o3])
    
    # Build the cell integration matrix B
    B_ij = np.r_[tuple(B_ij)]
    I = np.r_[tuple(I)]
    J = np.r_[tuple(J)] 
    
    B = sparse.csr_matrix((B_ij, (I, J)), shape=(n, n), dtype=np.float64)
    
    # Debug only
    if debug:    
        assert np.allclose(B.T.dot(np.ones(n)), area1r_3), 'B did not pass the unit test.'
     
    # The cotan weight for Dc/Lc is 0.5*cotan
    cotan_weight_12 *= 0.5
    cotan_weight_23 *= 0.5
    cotan_weight_31 *= 0.5    
    
    # Fill sparse weights and indices, including the other symmetric half
    L_ij = np.r_[cotan_weight_12, cotan_weight_12, cotan_weight_23, cotan_weight_23, cotan_weight_31, cotan_weight_31]
    I = np.r_[i1, i2, i2, i3, i3, i1]
    J_L = np.r_[i2, i1, i3, i2, i1, i3] 
    
    mswitch = -switch
    D_ij = np.r_[cotan_weight_12*switch[:,2], cotan_weight_12*mswitch[:,2], \
                  cotan_weight_23*switch[:,0], cotan_weight_23*mswitch[:,0], \
                  cotan_weight_31*switch[:,1], cotan_weight_31*mswitch[:,1]] 
    J_D = np.r_[FE[:,2], FE[:,2], FE[:,0], FE[:,0], FE[:,1], FE[:,1]]
    
    EIP_ij = np.r_[cotan_weight_12, cotan_weight_23, cotan_weight_31]
    I_EIP = np.r_[FE[:,2], FE[:,0], FE[:,1]]     
    
    # Form L and D, non area-normalised and without diagonal entries; as well as the edge (pseudo-)metric
    Lc = sparse.csr_matrix((L_ij, (I, J_L)), shape=(n, n))
    Dc = sparse.csr_matrix((D_ij, (I, J_D)), shape=(n, o))
    
    EIP = sparse.coo_matrix((EIP_ij, (I_EIP, I_EIP)), shape=(o,o)).todia()
    
    
    # Compute diagonal entries and properly adjust the sign depending on whether the positive or negative laplacian is desired
    if negative_eigenvalues:
        Lc = Lc - sparse.diags(Lc * np.ones(n), offsets=0, shape=(n, n))
    else:
        Lc = sparse.diags(Lc * np.ones(n), offsets=0, shape=(n, n)) - Lc
    
    Lc = Lc.tocsr()
    
    # Compute the area of each vertex cell, i.e. that of the Voronoi region (truncated to the 1-ring if obtuse),
    # from the face contributions.
    # Return a face to vertex cell interpolator, based on the contribution of each face to the cell
    face_number = np.arange(m)    
    FtC_kl = np.r_[a1, a2, a3]
    K = np.r_[i1, i2, i3]
    L = np.r_[face_number, face_number, face_number]        
    FtC = sparse.csr_matrix((FtC_kl, (K, L)), shape=(n, m))
    
    # Our vertex areas
    A = FtC.dot(np.ones(m))
        
    # Normalise rows to 1 before passing to FtC    
    normalisation = 1/A
    FtC.data *= normalisation.repeat(np.diff(FtC.indptr))
        
    # Return
    return Lc, B, M, FtC, FtR, A, Dc, EIP


def compute_mixed_laplacian(facenet, tris, FE_sign=None, E=None, debug=False, negative_eigenvalues=True, r=1/3):
    """
    Let's take care of the obtuse angles by switching to barycentric dual mesh when the ratio of barycentric half dual edge
    to circumcentric half dual edge goes above 3.
    
    I'm getting lazy but this time for barycentric cells formulae readily available from 
        Exact integration formulas for the fve method, Vandewalle
    int_{barycentric_subcell_of_k_in_T} L_i dA = 11/54*A if i=k, 7/108*A if i!=k.
    Over the 3 subcells in a triangle, this correctly sums to A/3.
    """
    
    # Misc
    m = len(tris) 
    HE, IV, IF, FN, FA, FE, n = facenet
    o = len(HE)
            
    # Shortcut
    i1 = tris[:,0]
    i2 = tris[:,1]
    i3 = tris[:,2]
    
    # to reorient if needed   
    if FE_sign:
        switch = FE_sign[0]
    else:
        switch = get_face_edge_signs(FE, IF)        
        if FE_sign == []:
            FE_sign.append(switch)       
    
    edges = np.einsum('ijk,ij->ijk',HE[:,1:][FE],switch) # 1st dim: face, 2nd: edge, 3rd: coordinates
    e1 = edges[:,0,:]
    e2 = edges[:,1,:]
    e3 = edges[:,2,:]
    
    if E==[]:
        E.extend([e1,e2,e3])
    
    #
    triangle_area = FA
    area_over_3 = triangle_area/3
    area_over_12 = triangle_area/12
    
    # Compute the area of each vertex 1-ring, from the face contributions.
    # Return a face to vertex 1-ring interpolator, based on the contribution of each face to the 1-ring
    face_number = np.arange(m)    
    FtR_kl = np.r_[triangle_area, triangle_area, triangle_area]
    K = np.r_[i1, i2, i3]
    L = np.r_[face_number, face_number, face_number]        
    FtR = sparse.csr_matrix((FtR_kl, (K, L)), shape=(n, m))
    
    # Our vertex ring areas
    A1r = FtR.dot(np.ones(m))
        
    # Normalise rows to 1   
    normalisation1r = 1/A1r
    FtR.data *= normalisation1r.repeat(np.diff(FtR.indptr))
    
    # Squared edge lengths
    e1_e1 = np.sum(e1*e1,1)
    e2_e2 = np.sum(e2*e2,1)
    e3_e3 = np.sum(e3*e3,1)
    
    # Edge inner products
    e1_e2 = (e1*e2).sum(axis=1)
    e2_e3 = (e2*e3).sum(axis=1)
    e3_e1 = (e1*e3).sum(axis=1)   
    
    # Compute cotan of all combinations
    cotan_weight_12 = -e1_e2 / veclen(np.cross(e1, e2))
    cotan_weight_23 = -e2_e3 / veclen(np.cross(e2, e3))
    cotan_weight_31 = -e3_e1 / veclen(np.cross(e3, e1))  
    
    # Compute the mass matrix M
    m_off = area_over_12
    m_diag = m_off*2
    
    M_ij = np.r_[m_diag, m_diag, m_diag, m_off, m_off, m_off, m_off, m_off, m_off]
    I = np.r_[i1, i2, i3, i1, i1, i2, i2, i3, i3]
    J = np.r_[i1, i2, i3, i2, i3, i3, i1, i1, i2] 
    
    M = sparse.csc_matrix((M_ij, (I, J)), shape=(n, n))
    
    # Debug only
    if debug:
        area1r_3 = A1r/3        
        assert np.allclose(M.dot(np.ones(n)), area1r_3), 'M did not pass the unit test.'
    
    # Compute the xi_i's, where xi_i = cot(j)cot(k) = (cot(j)+cot(k))/(tan(j)+tan(k)) = 1 - 2*sin(2i)/sum_l(sin(2l))
    # and bar_i(circumcenter) = (1-xi_i)/2 
    xi_1 = cotan_weight_31*cotan_weight_12
    xi_2 = cotan_weight_12*cotan_weight_23
    xi_3 = cotan_weight_23*cotan_weight_31
    
    # Compute the nu_ip's, where nu_ip = e_ip^2 / e_ij^t e_ik; i != p, i and p do not have symmetric roles here
    # Note that, nu_ip = (e_ip^2*tan(p))/(2*A_T)
    # If T is obtuse at i, bar_j(b) = 0.5*nu_ki and bar_j(a) = 1-0.5*nu_ji, bar_k(a) = 0.5*nu_ji and bar_k(b) = 1-0.5*nu_ki
    # with a the intersect of the height at m=(i+j)/2 with the segment jk, and b the intersect the height at n=(i+k)/2 with jk
    # In general, the orthogonal projection o' of o=(j+k)/2 on ij is such that jm/jo' = nu_ji (= ja/jo in the obtuse at i 
    # case). The link between both equations is an application of Thales theorem.
    # Moreover, the area of joo' is 1/4 of A_T/nu_ji (or A_jmo/nu_ji). In the obtuse case (still valid in the acute case but 
    # not as useful), A_jma = A_jmo*nu_ji as well. We use those a lot because every int_{A_M inter T} B_l ends up being
    # written as A_{some inner triangle}*constant_factor or a sum of such terms.
    inu_12 = -e2_e3/e3_e3
    inu_21 = -e3_e1/e3_e3
    inu_13 = -e2_e3/e2_e2
    inu_31 = -e1_e2/e2_e2
    inu_23 = -e3_e1/e1_e1
    inu_32 = -e1_e2/e1_e1
    
    # Now we're going to compute all coefficients of the cell integral matrix B, treating obtuse and acute triangles 
    # differently, and in fact being a bit more picky than just acute.
    # While at it, we record the contribution of each face to their vertices' cell in terms of area
    B_ij = [] 
    I = []
    J = []
    
    a1 = np.zeros(m)
    a2 = np.zeros(m)
    a3 = np.zeros(m)
    
    # The cotan weight is 0.5*cotan. This is the half dual edge for a circumcentric (Voronoi) dual.
    cotan_weight_12 *= 0.5
    cotan_weight_23 *= 0.5
    cotan_weight_31 *= 0.5
    
    # The barycentric weight is 1/3 of the median, by the edge. The median length can be obtained from edge lengths
    twice_sum_sq_e = (e1_e1 + e2_e2 + e3_e3)*2
    barycentric_weight_3 = np.sqrt(twice_sum_sq_e/e3_e3-3)*(1/6)
    barycentric_weight_1 = np.sqrt(twice_sum_sq_e/e1_e1-3)*(1/6)
    barycentric_weight_2 = np.sqrt(twice_sum_sq_e/e2_e2-3)*(1/6)
        
    # a. Voronoi or barycentric?
    small_3 = (cotan_weight_12/barycentric_weight_3) < r
    small_1 = (cotan_weight_23/barycentric_weight_1) < r
    small_2 = (cotan_weight_31/barycentric_weight_2) < r
    
    barycentric_dual = (small_1+small_2+small_3) > 0
    voronoi_dual = ~barycentric_dual
    if debug:
        print('Non-Voronoi percentage: ' + str(np.sum(barycentric_dual)/m*100) + '%')
    
    # b. Voronoi
    area_v = triangle_area[voronoi_dual]
    
    if len(area_v):
        i1_v = i1[voronoi_dual]
        i2_v = i2[voronoi_dual]
        i3_v = i3[voronoi_dual]

        xi1_v = xi_1[voronoi_dual]
        xi2_v = xi_2[voronoi_dual]
        xi3_v = xi_3[voronoi_dual]

        inu12_v = inu_12[voronoi_dual]
        inu21_v = inu_21[voronoi_dual]
        inu13_v = inu_13[voronoi_dual]
        inu31_v = inu_31[voronoi_dual]
        inu23_v = inu_23[voronoi_dual]
        inu32_v = inu_32[voronoi_dual]
        
        # Barycentric coordinates of the circumcenter
        bO_1 = (1-xi1_v)/2
        bO_2 = (1-xi2_v)/2
        bO_3 = (1-xi3_v)/2
        
        if debug:
            # Check that the barycentric coordinates sum to 1
            assert np.allclose(bO_1+bO_2+bO_3, np.ones(len(bO_1))), \
                "Barycentric coordinates of the circumcenter do not sum to 1."
        
        # Useful quantities
        bOp1_1 = 1+bO_1
        bOp1_2 = 1+bO_2
        bOp1_3 = 1+bO_3
        
        q11 = bOp1_1*xi1_v
        q21 = bOp1_1*xi2_v
        q31 = bOp1_1*xi3_v
        q12 = bOp1_2*xi1_v
        q22 = bOp1_2*xi2_v
        q32 = bOp1_2*xi3_v
        q13 = bOp1_3*xi1_v
        q23 = bOp1_3*xi2_v
        q33 = bOp1_3*xi3_v
        
        area_over_12_v = area_over_12[voronoi_dual]

        # Cell areas   
        a1[voronoi_dual] = (1+xi1_v)*area_v/4
        a2[voronoi_dual] = (1+xi2_v)*area_v/4
        a3[voronoi_dual] = (1+xi3_v)*area_v/4
        
        # Debug only
        if debug:
            meyer_a1_v = e3_e3[voronoi_dual]*cotan_weight_12[voronoi_dual] + \
                                    e2_e2[voronoi_dual]*cotan_weight_31[voronoi_dual]
            meyer_a2_v = e1_e1[voronoi_dual]*cotan_weight_23[voronoi_dual] + \
                                    e3_e3[voronoi_dual]*cotan_weight_12[voronoi_dual]
            meyer_a3_v = e2_e2[voronoi_dual]*cotan_weight_31[voronoi_dual] + \
                                    e1_e1[voronoi_dual]*cotan_weight_23[voronoi_dual] 
            assert np.allclose(a1[acute_triangle], meyer_a1_v/8), "Voronoi areas did not pass the unit test"
            assert np.allclose(a2[acute_triangle], meyer_a2_v/8), "Voronoi areas did not pass the unit test"
            assert np.allclose(a3[acute_triangle], meyer_a3_v/8), "Voronoi areas did not pass the unit test"

        # Main contributions: vertex i over cell i
        
        B_ij.extend([(q11+2)*area_over_12_v, (q22+2)*area_over_12_v, (q33+2)*area_over_12_v])  
        I.extend([i1_v, i2_v, i3_v])
        J.extend([i1_v, i2_v, i3_v])

        # Secondary contributions: vertices j and k over cell i
        # Note that 1/nu_pq + 1/nu_qp = 1, and sum_i bO_i = 1, so that b_1i+b_2i+b_3i = triangle_area/3 as expected.

        #    i = 1st vertex, j = 2nd, k = 3rd
        b_ik = ( 0.5 + q13 - bO_3*inu21_v ) * area_over_12_v
        b_ij = ( 0.5 + q12 - bO_2*inu31_v ) * area_over_12_v
        B_ij.extend([b_ik, b_ij])
        I.extend([i1_v, i1_v])
        J.extend([i3_v, i2_v])

        #    i = 2nd vertex, j = 1st, k = 3rd
        b_ik = ( 0.5 + q23 - bO_3*inu12_v ) * area_over_12_v
        b_ij = ( 0.5 + q21 - bO_1*inu32_v ) * area_over_12_v
        B_ij.extend([b_ik, b_ij])
        I.extend([i2_v, i2_v])
        J.extend([i3_v, i1_v])

        #    i = 3rd vertex, j = 1st, k = 2nd
        b_ik = ( 0.5 + q32 - bO_2*inu13_v ) * area_over_12_v
        b_ij = ( 0.5 + q31 - bO_1*inu23_v ) * area_over_12_v
        B_ij.extend([b_ik, b_ij])
        I.extend([i3_v, i3_v])
        J.extend([i2_v, i1_v])
    
    # c. Barycentric dual
    area_b = triangle_area[barycentric_dual] 
    
    if len(area_b):
        i1_b = i1[barycentric_dual]
        i2_b = i2[barycentric_dual]
        i3_b = i3[barycentric_dual]

        area_over_3_b = area_over_3[barycentric_dual]

        # Areas        
        a2[barycentric_dual] = area_over_3_b
        a3[barycentric_dual] = area_over_3_b
        a1[barycentric_dual] = area_over_3_b
        
        # i or not i
        b_i = 11/54*area_b
        b_noti = 7/108*area_b
        
        # Add to B
        B_ij.extend([b_i, b_noti, b_noti, b_noti, b_i, b_noti, b_noti, b_noti, b_i])
        I.extend([i1_b, i1_b, i1_b, i2_b, i2_b, i2_b, i3_b, i3_b, i3_b])
        J.extend([i1_b, i2_b, i3_b, i1_b, i2_b, i3_b, i1_b, i2_b, i3_b])        
    
    # Build the cell integration matrix B
    B_ij = np.r_[tuple(B_ij)]
    I = np.r_[tuple(I)]
    J = np.r_[tuple(J)] 
    
    B = sparse.csr_matrix((B_ij, (I, J)), shape=(n, n), dtype=np.float64)
    
    # Debug only
    if debug:    
        assert np.allclose(B.T.dot(np.ones(n)), area1r_3), 'B did not pass the unit test.' 
        
        
    # Lc/Dc weights, primal to dual edge ratio
    edge_hodge_star_1 = cotan_weight_23
    edge_hodge_star_2 = cotan_weight_31
    edge_hodge_star_3 = cotan_weight_12
    
    edge_hodge_star_1[barycentric_dual] = barycentric_weight_1[barycentric_dual]
    edge_hodge_star_2[barycentric_dual] = barycentric_weight_2[barycentric_dual]
    edge_hodge_star_3[barycentric_dual] = barycentric_weight_3[barycentric_dual]
    
    # Fill sparse weights and indices, including the other symmetric half
    L_ij = np.r_[edge_hodge_star_3, edge_hodge_star_3, \
                 edge_hodge_star_1, edge_hodge_star_1, \
                 edge_hodge_star_2, edge_hodge_star_2]
    I = np.r_[i1, i2, i2, i3, i3, i1]
    J_L = np.r_[i2, i1, i3, i2, i1, i3] 
    
    mswitch = -switch
    D_ij = np.r_[edge_hodge_star_3*switch[:,2], edge_hodge_star_3*mswitch[:,2], \
                  edge_hodge_star_1*switch[:,0], edge_hodge_star_1*mswitch[:,0], \
                  edge_hodge_star_2*switch[:,1], edge_hodge_star_2*mswitch[:,1]] 
    J_D = np.r_[FE[:,2], FE[:,2], FE[:,0], FE[:,0], FE[:,1], FE[:,1]]
    
    EHS_ij = np.r_[edge_hodge_star_3, edge_hodge_star_1, edge_hodge_star_2]
    I_EHS = np.r_[FE[:,2], FE[:,0], FE[:,1]]     
    
    # Form L and D, non area-normalised and without diagonal entries; as well as the edge metric
    Lc = sparse.csr_matrix((L_ij, (I, J_L)), shape=(n, n))
    Dc = sparse.csr_matrix((D_ij, (I, J_D)), shape=(n, o))
    
    EHS = sparse.coo_matrix((EHS_ij, (I_EHS, I_EHS)), shape=(o,o)).todia()   
    
    # Compute diagonal entries and properly adjust the sign depending on whether the positive or negative laplacian is desired
    if negative_eigenvalues:
        Lc = Lc - sparse.diags(Lc * np.ones(n), offsets=0, shape=(n, n))
    else:
        Lc = sparse.diags(Lc * np.ones(n), offsets=0, shape=(n, n)) - Lc
    
    Lc = Lc.tocsr()
    
    # Compute the area of each vertex cell, i.e. that of the Voronoi region (truncated to the 1-ring if obtuse),
    # from the face contributions.
    # Return a face to vertex cell interpolator, based on the contribution of each face to the cell
    face_number = np.arange(m)    
    FtC_kl = np.r_[a1, a2, a3]
    K = np.r_[i1, i2, i3]
    L = np.r_[face_number, face_number, face_number]        
    FtC = sparse.csr_matrix((FtC_kl, (K, L)), shape=(n, m))
    
    # Our vertex areas
    A = FtC.dot(np.ones(m))
        
    # Normalise rows to 1 before passing to FtC    
    normalisation = 1/A
    FtC.data *= normalisation.repeat(np.diff(FtC.indptr))
        
    # Return
    return Lc, B, M, FtC, FtR, A, Dc, EHS, voronoi_dual, negative_eigenvalues


def compute_mesh_helmotzian(facenet, tris, FE_sign=None, laplacian_holder=None):
    """
    laplacian: as output by compute_mesh_laplacian
    Note: the expression is normalized differently than just d_0 delta_1 + delta_2 d_1, with EIP (or EHS) appearing
    in the first term rather than the second. I suspect I did so for two reasons: 
    (1) this is the relevant LHS for the eigensystem to solve for the kernel of the Helmotzian, with the RHS metric being EHS
    (2) this decouples clearly the part that is numerically always friendly from the part that depends on the choice of cells.
    """
    
    #
    if laplacian_holder:
        _, _, _, _, _, VA, Dc, EHS, _, _ = laplacian_holder
    else:
        laplacian = compute_mixed_laplacian(facenet, tris, FE_sign=FE_sign, negative_eigenvalues=True)
        _, _, _, _, _, VA, Dc, EHS, _, _ = laplacian
        
        if laplacian_holder==[]:
            laplacian_holder.extend(laplacian)            
            
    # Misc
    m = len(tris) 
    HE, IV, IF, FN, FA, FE, n = facenet
    o = len(HE)
    
    #        
    C_adjoint = curl_adjoint_operator(IF, m)
    
    # Given what the respective routines for operators return,
    # H1 = C_adjoint*1/FA*C - EIP*G*D, D = 1/VA Dc   
    # Instead we are going to build things in an exact symmetric fashion to reduce potential trouble
    
    # The observation here is that -(EIP G)^T = Dc, so we don't need the gradient operator.
    # And we don't need EIP either for this part.
    # Instead compute (1/sqrt(VA) Dc)^T (1/sqrt(VA) Dc) = Dc^t 1/VA Dc = - EIP G 1/VA Dc = - EIP G D
    
    # Same trick for curl* curl. Our C is already face integrated ("Ci") and C_adjoint edge integrated ("C*i")
    # so we have C^t = C_adjoint,
    # so C_adjoint 1/FA C = (C_adjoint 1/sqrt(FA)) (C_adjoint 1/sqrt(FA))^t
    
    # As a positive side-effect, we eliminate\delay the use of the shabby EIP, in case of obtuse angles.
    # It will only be used to normalize the eigenmodes in some variant or another
    
    #  
    ihFA = 1/np.sqrt(FA)
    h_CtiFAC = C_adjoint*sparse.diags(ihFA, 0)
    CtiFAC = h_CtiFAC*h_CtiFAC.transpose(copy=True)
    
    # We don't modify in place just in case Dc is used somewhere else
    ihVA = 1/np.sqrt(VA)
    h_mEHSGD = Dc.copy()
    h_mEHSGD.data *= ihVA.repeat(np.diff(h_mEHSGD.indptr))
    mEHSGD = h_mEHSGD.transpose(copy=True)*h_mEHSGD
    
    # Form the Helmotzian
    H1 = CtiFAC + mEHSGD
    
    return H1, EHS       


def compute_geometric_face_laplacian(facenet, tris, FE_sign=None, laplacian_holder=None, negative_eigenvalues=True):
    """
    Using Discrete Exterior Calculus, this is d_1 delta_2.
    We get 1/A_i sum_j 1/EHS_ij (h_j-h_i).
    """

    #
    if laplacian_holder:
        EHS = laplacian_holder[7]
    else:
        laplacian = compute_mixed_laplacian(facenet, tris, FE_sign=FE_sign)
        EHS = laplacian[7]
        
        if laplacian_holder==[]:
            laplacian_holder.extend(laplacian)
     
    iEHS = 1/EHS.data
    iEHS = iEHS.squeeze()
    
    # Misc.
    m = len(tris)
    HE, IV, IF, FN, FA, FE, n = facenet
            
    #
    f1 = IF[:,0]
    f2 = IF[:,1]
    
    #
    Lf_ij = np.r_[-iEHS,-iEHS,iEHS,iEHS]
    if negative_eigenvalues:
        Lf_ij = -Lf_ij
    I = np.r_[f1, f2, f1, f2]
    J = np.r_[f2, f1, f1, f2]
    
    qu_I = I.reshape((len(I),1))*4 + np.arange(4)
    qu_J = J.reshape((len(J),1))*4 + np.arange(4)
    qu_Lf_ij = np.repeat(Lf_ij,4)
    
    #
    Lf = sparse.csr_matrix((Lf_ij, (I, J)), shape=(m, m))    
    qu_Lf = sparse.csr_matrix((qu_Lf_ij, (qu_I.reshape(-1), qu_J.reshape(-1))), shape=(m*4, m*4))
    
    return Lf, qu_Lf, negative_eigenvalues     


def compute_vertex_normals(verts, tris, FN, laplacian, atol=1e-1):
    """
    Based on the Laplacian whenever we can, otherwise reverting to face normal interpolation.
    
    """
    
    Lc, B, M, FtC, FtR, VA, _, _, voronoi_dual, negative_eigenvalues = laplacian
    
    VN_cells, _ = get_cell_normals(FN, FtC) # we'll also use them to disambiguate the sign.
    
    eps = 1e-5
    VN = Lc.dot(verts)
    sqn_uVN = np.sqrt(sq_norm(VN))/(2*VA)
    mask = (sqn_uVN > atol**2)*(FtR.dot(voronoi_dual)>1-eps)
    
    VN[~mask,:] = VN_cells[~mask,:]
    VN[mask,:] /= veclen(VN[mask,:]).reshape((-1,1))
    VN[mask,:] *= ( 2*(np.sum(VN[mask,:]*VN_cells[mask,:],axis=-1)>0)-1 )[:,np.newaxis]
    
    return VN


def compute_dual_hyperedges(facenet, verts, tris, FE_sign=None, VN_holder=None, laplacian_holder=None, r=.2):
    
    # Misc.
    m = len(tris) 
    HE, IV, IF, FN, FA, FE, n = facenet
    o = len(HE)
    
    #
    E = []    
    laplacian = compute_mixed_laplacian(facenet, tris, FE_sign=FE_sign, E=E, negative_eigenvalues=True, r=r)
    _, _, _, _, _, VA, Dc, EHS, voronoi_dual, _ = laplacian
    barycentric_dual = ~voronoi_dual    
        
    if laplacian_holder==[]:
        laplacian_holder.extend(laplacian)  
        
    if VN_holder:
        VN = VN_holder[0]
    else:
        VN = compute_vertex_normals(verts, tris, FN, laplacian, atol=1e-1)
        
        if VN_holder==[]:
            VN_holder.append(VN)
    
    #
    dual_verts = np.zeros((m,3))    
    dual_verts[barycentric_dual, :] = np.mean(verts[tris[barycentric_dual,:]] , axis=1)
    
    # verts for voronoi faces    
    v1, v2, v3 = [e.squeeze() for e in np.split(verts[tris[voronoi_dual,:]], 3, axis=1)]
    
    # Face edges for voronoi faces
    e1, e2, e3 = E
    e1 = e1[voronoi_dual,:]
    e2 = e2[voronoi_dual,:]
    e3 = e3[voronoi_dual,:]
    
    # Squared edge lengths
    e1_e1 = np.sum(e1*e1,1)
    e2_e2 = np.sum(e2*e2,1)
    e3_e3 = np.sum(e3*e3,1)
    
    # Edge inner products
    e1_e2 = (e1*e2).sum(axis=1)
    e2_e3 = (e2*e3).sum(axis=1)
    e3_e1 = (e1*e3).sum(axis=1)
    
    # Unnormalized barycentric coordinates for the circumcenter
    w1 = -e1_e1*e2_e3
    w2 = -e2_e2*e3_e1
    w3 = -e3_e3*e1_e2
    
    # Altogether
    dual_verts[voronoi_dual, :] = (v1*w1.reshape((-1,1)) + 
                                   v2*w2.reshape((-1,1)) + 
                                   v3*w3.reshape((-1,1))) / (8*np.square(FA[voronoi_dual])).reshape((-1,1))
    
    # Now we can create dual edges
    dual_edges = dual_verts[IF[:,0]]-dual_verts[IF[:,1]]
    
    # ... and compute the curvature along the dual edge
    ImHE = HE[:,1:]
    u_edges = ImHE/veclen(ImHE).reshape((-1,1))
    Hij_dual = np.sum(u_edges*(VN[IV[:,1]]-VN[IV[:,0]]), axis=-1) * veclen(dual_edges) # twice the edge curvature...
    
    # Wrap up in the dual hyperedges
    HE_star = np.c_[Hij_dual, dual_edges]
    
    return HE_star, dual_verts


def compute_dirac_star_operator(HE_star, VA, facenet, rho=None):
    """
    Intrinsic Dirac operator on dual mesh.
    
    mean_curvature_potential: (n,) array of target mean curvature (half density) at each vertex / dual face
    """
    
    # Misc. 
    HE, IV, IF, FN, FA, FE, n = facenet
    m = len(IV)
    o = len(HE) 
    
    # Create row, col, and data in vector form for now
    row = np.r_[IV[:,0],IV[:,1]]
    col = np.r_[IV[:,1],IV[:,0]]   
    data = np.r_[HE_star,qu_conjugate(HE_star)] / 2
    
    if rho is not None:
        row = np.r_[row,np.arange(n)]
        col = np.r_[col,np.arange(n)]
        potential = -rho*VA
        qu_potential = np.zeros((n,4),dtype=data.dtype) 
        qu_potential[:,0] = potential
        data = np.r_[data,qu_potential]
    
    data = qu_matrix(data)
        
    #
    rows = row.reshape((len(row),1))*4 + np.arange(4).repeat(4)
    cols = col.reshape((len(col),1))*4 + np.broadcast_to(np.arange(4),(4,4)).reshape(-1)
    
    #
    Aii = VA.repeat(4)
    
    # Return
    D = sparse.csr_matrix((data.reshape(-1), (rows.reshape(-1),cols.reshape(-1))), shape=(n*4, n*4))
    A = sparse.diags(Aii, 0, dtype=D.dtype)
        
    return D, A