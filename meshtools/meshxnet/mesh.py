import numpy as np
from scipy import sparse
from .utils.vector import veclen, normalized

def get_face_vertices(verts, tris):
    p1 = verts[tris[:,0],:]
    p2 = verts[tris[:,1],:]
    p3 = verts[tris[:,2],:]
    
    return p1,p2,p3

def get_face_edges(verts, tris, P = None):
    if P:
        p1,p2,p3 = P
    else:
        p1,p2,p3 = get_face_vertices(verts, tris)
        if P == []:
            P.extend([p1,p2,p3])
            
    e1 = p3-p2
    e2 = p1-p3
    e3 = p2-p1
        
    return e1, e2, e3

def get_face_barycenters(verts, tris, P = None):
    if P:
        p1,p2,p3 = P
    else:
        p1,p2,p3 = get_face_vertices(verts, tris)
        if P == []:
            P.extend([p1,p2,p3])
            
    b = (p1+p2+p3)/3
        
    return b

def get_area_vectors(verts, tris, P = None, E = None):
    if E:
        e1,e2,e3 = E
    else:
        e1,e2,e3 = get_face_edges(verts, tris, P)
        if E == []:
            E.extend([e1,e2,e3])
            
    n = ( np.cross(e1, e2) + np.cross(e2, e3) + np.cross(e3, e1) ) / 6
    
    return n

def get_squared_edge_lengths(verts, tris, P = None, E = None):
    """
    If passed, P or E should be an empty list, or a 3-element list of all face vertices/edges.
    In the former case, the list is updated (and returned) as a side-effect of computations.
    """
    
    if E:
        e1,e2,e3 = E
    else:
        e1,e2,e3 = get_face_edges(verts, tris, P)
        if E == []:
            E.extend([e1,e2,e3])
    
    return np.c_[np.sum(e1*e1,1),np.sum(e2*e2,1),np.sum(e3*e3,1)]


def get_face_normals_and_areas(verts, tris, P = None, E = None):           
    area_vector = get_area_vectors(verts, tris, P, E)
    triangle_area = veclen(area_vector)
    
    return area_vector/triangle_area[:, np.newaxis], triangle_area


def get_cell_normals(FN, FtC):
    """
    FtC, the face to vertex cell interpolator, can typically be obtained from
    compute_mesh_laplacian() as the two are closely related.
    """
    
    uVN = FtC.dot(FN) 
        # may actually not have norm 1, this is after all only weighted linear interpolation from face voronoi areas
        
    VN = uVN / veclen(uVN)[:,np.newaxis]
    return VN, uVN


def get_ring_normals(FN, FtR):
    """
    FtR, the face to vertex 1-ring interpolator, can typically be obtained from
    compute_mesh_laplacian() as the two are closely related.
    """
    
    uVN = FtR.dot(FN) 
        # may actually not have norm 1, this is after all only weighted linear interpolation from face voronoi areas
        
    VN = uVN / veclen(uVN)[:,np.newaxis]
    return VN, uVN


def compute_mesh_laplacian(verts, tris, P = None, E = None, FN = None, FtC = None, FtR = None, GC = None, debug = False):
    """
    computes a sparse matrix representing the discretized laplace-beltrami operator of the mesh
    given by n vertex positions ("verts") and a m triangles ("tris") 
    
    verts: (n, 3) array (float)
    tris: (m, 3) array (int) - indices into the verts array

    computes the conformal weights ("cotangent weights") for the mesh, ie:
    w_ij = - .5 * (cot \alpha + cot \beta)

    See:
        Olga Sorkine, "Laplacian Mesh Processing"
        and for theoretical comparison of different discretizations, see 
        Max Wardetzky et al., "Discrete Laplace operators: No free lunch"

    returns matrices Lc, B, M, A, A1r with the meanings defined below. We are working with a mixed finite volume / finite 
    element formulation here. Note that it is not necessary for any combination of these matrices, e.g. B^{-1}Lc, to relate 
    to a pointwise approximation of the Laplace-Beltrami operator. 
    In fact, since grad_x(A_{1-ring})/A_{1-ring} -> Lx when the area tends towards 0, where grad_x is the Euclidean gradient
    with respect to the coordinates of x ("relative variation of the area around the point when the point moves"), we can
    expect neither B^{-1}Lc nor A^{-1}Lc to provide consistent pointwise estimates of the Laplacian at vertices. Recall that
    Lc*x = grad_x(A_{1-ring}) for any triangulation.
    
    A: area of the (truncated) Voronoi cells.
    A1r: area of the 1-ring around vertices.
    Lc: symmetric part of the Laplacian, technically the integral over cells of the Laplacian
    B: to compute the integral over cells of any piecewise linear function from its value at vertices. B_ij = int_{A_i} B_j
    M: the usual mass matrix, M_pq = int_A B_p B_q. Only involved when looking fo eigenmodes of the discrete Laplace Beltrami
    operator, since the proper normalisation condition for eigenfunctions to live in the same space as "functions over meshes" 
    should be Phi_n^T M Phi_n = 1. (M-orthonormal basis)
    
    The heat flow du/dt = Lu would be discretized as B*dU/dt = Lc*U, where U is the value at vertices.
    The Yamabe flow du/dt = Lu + c.n would be discretised as B*dU/dt = Lc*U + c A.N where N is the FtV
    interpolation of face normals (unnormalised).
    
    Different area normalisations can be used, we follow the one in Meyer et al., 
    "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
    rather than the expression A = triangle_area/3 given in the "Geodesics in Heat" paper.
    
    Optional inputs/outputs:
    P: face vertices [p1, p2, p3]
    E: face edges [e1, e2, e3]
    FN: face normals [np.array(m,3)]
    FtV: face to vertex interpolation matrix, scipy csr format
    
    Optional outputs:
    GC: gaussian curvature, np.array(n,)
    """
    
    # Misc
    n = len(verts)
    m = len(tris)
        
    # Get face vertices and edges if not passed, and pass them on if required
    if not P:
        p1, p2, p3 = get_face_vertices(verts, tris)       
        if P == []:
            P.extend([p1,p2,p3])
        else:
            P = [p1,p2,p3]
                    
    if E:
        e1,e2,e3 = E
    else:
        e1,e2,e3 = get_face_edges(verts, tris, P)
        if E == []:
            E.extend([e1,e2,e3])   
            
    # Get face normals and triangle areas, useful later
    face_normals, triangle_area = get_face_normals_and_areas(verts, tris, P, E)
    if FN == []:
        FN.append(face_normals)        
            
    # Shortcut
    i1 = tris[:,0]
    i2 = tris[:,1]
    i3 = tris[:,2]
    
    area_over_3 = triangle_area/3
    area_over_12 = triangle_area/12
    
    # Compute the area of each vertex 1-ring, from the face contributions.
    # Return a face to vertex 1-ring interpolator, based on the contribution of each face to the 1-ring
    face_number = np.arange(m)    
    FtR_kl = np.r_[triangle_area, triangle_area, triangle_area]
    K = np.r_[i1, i2, i3]
    L = np.r_[face_number, face_number, face_number]        
    face_to_ring = sparse.csr_matrix((FtR_kl, (K, L)), shape=(n, m))
    
    # Our vertex areas
    A1r = face_to_ring.dot(np.ones(m))
        
    # Normalise rows to 1 before passing to FtC    
    if FtR == []:
        normalisation1r = 1/A1r
        face_to_ring.data *= normalisation1r.repeat(np.diff(face_to_ring.indptr))
        FtR.append(face_to_ring)
    
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
    
    # Compute the gaussian curvature if the user provides a structure for us to output it
    if (GC == []) or debug:
        zero_vec = np.zeros(m)
        half_pi = np.full(m, np.pi/2)
        
        theta3 = np.where(np.isclose(cotan_weight_12, zero_vec), half_pi, np.arctan(1/cotan_weight_12)) 
            # between -pi/2 and pi/2       
        theta1 = np.where(np.isclose(cotan_weight_23, zero_vec), half_pi, np.arctan(1/cotan_weight_23))
        theta2 = np.where(np.isclose(cotan_weight_31, zero_vec), half_pi, np.arctan(1/cotan_weight_31))
        
        theta1[np.where(theta1<0)] += np.pi # between 0 and pi
        theta2[np.where(theta2<0)] += np.pi
        theta3[np.where(theta3<0)] += np.pi
        
        # compute 2*pi - sum_{faces that have x as vertex} face_angle_at_x as
        # 2*pi - sum_{v=1,2,3} sum_{faces that have x as v-th vertex} face_angle_at_x
        gaussian_curvature = np.full(n, 2*np.pi)
        for (i, theta) in [(i1,theta1), (i2,theta2), (i3,theta3)]:
            contribution = np.bincount(i, theta)
            gaussian_curvature[:len(contribution)] -= contribution 
     
        # we will have to normalise by cell areas below, so far it is the average over the cell
    
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
     
    # The cotan weight for Lc is 0.5*cotan
    cotan_weight_12 *= 0.5
    cotan_weight_23 *= 0.5
    cotan_weight_31 *= 0.5    
    
    # Fill sparse weights and indices, including the other symmetric half
    W_ij = np.r_[cotan_weight_23, cotan_weight_23, cotan_weight_31, cotan_weight_31, cotan_weight_12, cotan_weight_12]
    I = np.r_[i2, i3, i3, i1, i1, i2]
    J = np.r_[i3, i2, i1, i3, i2, i1] 
    
    # Form L, non area-normalised and without diagonal entries
    Lc = sparse.csr_matrix((W_ij, (I, J)), shape=(n, n))
    
    # Compute diagonal entries (this is the negative Laplacian, with positive eigenvalues)
    Lc = Lc - sparse.diags(Lc * np.ones(n), offsets=0, shape=(n, n))
    Lc = Lc.tocsr()
    
    # Compute the area of each vertex cell, i.e. that of the Voronoi region (truncated to the 1-ring if obtuse),
    # from the face contributions.
    # Return a face to vertex cell interpolator, based on the contribution of each face to the cell
    face_number = np.arange(m)    
    FtC_kl = np.r_[a1, a2, a3]
    K = np.r_[i1, i2, i3]
    L = np.r_[face_number, face_number, face_number]        
    face_to_cell = sparse.csr_matrix((FtC_kl, (K, L)), shape=(n, m))
    
    # Our vertex areas
    A = face_to_cell.dot(np.ones(m))
        
    # Normalise rows to 1 before passing to FtC    
    if FtC == []:
        normalisation = 1/A
        face_to_cell.data *= normalisation.repeat(np.diff(face_to_cell.indptr))
        FtC.append(face_to_cell)
        
    if GC == []:
        gaussian_curvature /= A1r
        GC.append(gaussian_curvature)
        
    # Return
    return Lc, B, M, A, A1r


def compute_finite_element_laplacian(verts, tris, P = None, E = None):
    """
    Simple finite element laplacian system to solve eigenvalue problems
    L_c phi = lambda M phi.
    
    For i!=j vertices of T, the triangle brings a contribution -0.25/A_T*<ei|ej> = -0.5*cot(alpha_ij), where
    ei, ej are edges opposite to vertices i,j and alpha_ij is the angle opposite to edge ij (=alpha_k).
    
    For i=j the triangle T contributes by 0.25/A_T*l_i^2=0.5*l_i/h_i, l_i the length of ei, h_i the height for edge ei.
    This rewrites as 0.5*{cot(alpha_j)+cot(alpha_k)}.
    
    In other words Lc is the same cotangent Laplacian as usually. So we don't actually implement this :)
    """
    
    print('Not implemented. Call compute_mesh_laplacian() and use Lc and M.')
    return None


def compute_mesh_laplacian_meyer(verts, tris, P = None, E = None, FN = None, FtC = None, FtR = None, GC = None):
    """
    computes a sparse matrix representing the discretized laplace-beltrami operator of the mesh
    given by n vertex positions ("verts") and a m triangles ("tris") 
    
    verts: (n, 3) array (float)
    tris: (m, 3) array (int) - indices into the verts array

    computes the conformal weights ("cotangent weights") for the mesh, ie:
    w_ij = - .5 * (cot \alpha + cot \beta)

    See:
        Olga Sorkine, "Laplacian Mesh Processing"
        and for theoretical comparison of different discretizations, see 
        Max Wardetzky et al., "Discrete Laplace operators: No free lunch"

    returns matrix L that computes the laplacian coordinates, e.g. L * x = delta
    in the form of L = diag(A)^{-1} Lc
    
    Different area normalisations can be used, we follow the one in Meyer et al., 
    "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds"
    rather than the expression A = triangle_area/3 given in the "Geodesics in Heat" paper.
    
    Optional inputs/outputs:
    P: face vertices [p1, p2, p3]
    E: face edges [e1, e2, e3]
    FN: face normals [np.array(n,3)]
    FtV: face to vertex interpolation matrix, scipy csr format
    
    Optional outputs:
    GC: gaussian curvature, np.array(n,)
    """
    
    # Misc
    n = len(verts)
    m = len(tris)
        
    # Get face vertices and edges if not passed, and pass them on if required
    if not P:
        p1, p2, p3 = get_face_vertices(verts, tris)       
        if P == []:
            P.extend([p1,p2,p3])
        else:
            P = [p1,p2,p3]
                    
    if E:
        e1,e2,e3 = E
    else:
        e1,e2,e3 = get_face_edges(verts, tris, P)
        if E == []:
            E.extend([e1,e2,e3])   
            
    # Get face normals and triangle areas, useful later
    face_normals, triangle_area = get_face_normals_and_areas(verts, tris, P, E)
    if FN == []:
        FN.append(face_normals)        
            
    # Shortcut
    i1 = tris[:,0]
    i2 = tris[:,1]
    i3 = tris[:,2]
    
    # Squared edge lengths
    e1_e1 = np.sum(e1*e1,1)
    e2_e2 = np.sum(e2*e2,1)
    e3_e3 = np.sum(e3*e3,1)
    
    # Edge inner products
    e1_e2 = (e1*e2).sum(axis=1)
    e2_e3 = (e2*e3).sum(axis=1)
    e3_e1 = (e1*e3).sum(axis=1)    
    
    # Compute cotan of all combinations, adjust the sign in advance
    cotan_weight_12 = -e1_e2 / veclen(np.cross(e1, e2))   
    cotan_weight_23 = -e2_e3 / veclen(np.cross(e2, e3))
    cotan_weight_31 = -e3_e1 / veclen(np.cross(e3, e1))
    
    # Compute the gaussian curvature if the user provides a structure for us to output it
    if GC == []:
        theta1 = np.arctan(1/cotan_weight_23) # between -pi/2 and pi/2
        theta2 = np.arctan(1/cotan_weight_31)
        theta3 = np.arctan(1/cotan_weight_12)
        
        theta1[np.where(theta1<0)] += np.pi # between 0 and pi
        theta2[np.where(theta2<0)] += np.pi
        theta3[np.where(theta3<0)] += np.pi
        
        # compute 2*pi - sum_{faces that have x as vertex} face_angle_at_x as
        # 2*pi - sum_{v=1,2,3} sum_{faces that have x as v-th vertex} ace_angle_at_x
        gaussian_curvature = np.full(n, 2*np.pi)
        for (i, theta) in [(i1,theta1), (i2,theta2), (i3,theta3)]:
            contribution = np.bincount(i, theta)
            gaussian_curvature[:len(contribution)] -= contribution 
     
        # we will have to normalise by cell areas below
    
    # The cotan weight is 0.5*cotan
    cotan_weight_12 *= 0.5    
    cotan_weight_23 *= 0.5
    cotan_weight_31 *= 0.5
    
    # Are triangles obtuse at one of their vertices?
    obtuse_at_1 = e2_e3 > 0
    obtuse_at_2 = e3_e1 > 0
    obtuse_at_3 = e1_e2 > 0
    obtuse_triangle = obtuse_at_1 + obtuse_at_2 + obtuse_at_3 > 0
    
    # Fill sparse weights and indices, including the other symmetric half
    W_ij = np.r_[cotan_weight_23, cotan_weight_23, cotan_weight_31, cotan_weight_31, cotan_weight_12, cotan_weight_12]
    I = np.r_[i2, i3, i3, i1, i1, i2]
    J = np.r_[i3, i2, i1, i3, i2, i1] 
    
    # Form L, non area-normalised and without diagonal entries
    Lc = sparse.csr_matrix((W_ij, (I, J)), shape=(n, n))
    
    # Compute diagonal entries
    Lc = Lc - sparse.spdiags(Lc * np.ones(n), 0, n, n)
    Lc = Lc.tocsr()
    
    # Compute the area of each vertex
    # We take the cotan weight whenever the triangle is non obtuse, the triangle area over 2 or 4 if the triangle is
    # obtuse at the vertex of interest or at another of its vertices
    
    # 1. Compute the cotan contributions of edges whenever the triangle is non obtuse; 
    # note that cotan_12 -> contribution for edge 3 
    happy_triangle = 1 - obtuse_triangle
    
    e3_contribution = e3_e3*cotan_weight_12*happy_triangle
    e2_contribution = e2_e2*cotan_weight_31*happy_triangle
    e1_contribution = e1_e1*cotan_weight_23*happy_triangle
    
    # 2. Compute the cotan/voronoi contributions to vertex areas within each triangle
    # e3_contribution -> contribution for the two points on edge 3
    a_1 = e2_contribution + e3_contribution
    a_2 = e3_contribution + e1_contribution
    a_3 = e1_contribution + e2_contribution
    
    # 3. Add the triangle area contributions for obtuse triangles
    a_1 += (obtuse_triangle+obtuse_at_1)*triangle_area
    a_2 += (obtuse_triangle+obtuse_at_2)*triangle_area
    a_3 += (obtuse_triangle+obtuse_at_3)*triangle_area
    
    a_1 *= 0.25
    a_2 *= 0.25
    a_3 *= 0.25
    
    # 4. Get the point area from the contributions of triangles it belongs to
    
    # Originally done as below:
    # A = np.zeros(n)
    # for (i, w) in [(i1,a_1), (i2,a_2), (i3,a_3)]:
    #    a = np.bincount(i, w)
    #    A[:len(a)] += a       
    # Done. But we actually want, as a bonus, if the user asks for it, 
    # to return a face to vertex interpolator, based on those weights above
    # So we are going to join both tasks.
    face_number = np.arange(len(i1))    
    FtC_kl = np.r_[a_1, a_2, a_3]
    K = np.r_[i1, i2, i3]
    L = np.r_[face_number, face_number, face_number]        
    face_to_vertex = sparse.csr_matrix((FtC_kl, (K, L)), shape=(n, m))
    
    # Our vertex areas
    A = face_to_vertex.dot(np.ones(m))
    
    # Normalise rows to 1 before passing to FtC
    if (FtC == []) or GC:
        normalisation = 1/A
    
    if FtC == []:
        face_to_vertex.data *= normalisation.repeat(np.diff(face_to_vertex.indptr))
        FtC.append(face_to_vertex)
        
    if GC == []:
        gaussian_curvature *= normalisation
        GC.append(gaussian_curvature)
        
    # Return
    return Lc, sparse.spdiags(A, 0, n, n), sparse.spdiags(A, 0, n, n), A, A


def get_mean_curvature(verts, laplacian, gauss_map, delta = None):
    return get_mean_curvature_2(verts, laplacian, gauss_map, delta)/2


def get_mean_curvature_2(verts, laplacian, gauss_map, delta = None):
    """
    verts: (n,3) np.array
    laplacian: the (NEGATIVE) mesh laplacian returned by compute_mesh_laplacian()
    gauss_map: (n,3) np.array, the map of vertex normals
    """
    
    # We need differential coordinates, compute them if needed
    if delta:
        delta_coordinates = delta[0]
    else:
        delta_coordinates = get_differential_coordinates(verts, laplacian)
        if delta == []:
            delta.append(delta_coordinates)
            
    # Could be more accurate to take the norm of delta coordinates instead, and to use
    # some other way to retrieve the correct sign.
    # Indeed, gauss_map is typically computed as the direction of FtV.n_F where n_F are face normals,
    # which is the average normal direction (up to normalisation) on the vertex cell.
    # It might differ from a "normalised" Lc*X, which would be a pointwise estimate of the normal since
    # A1r^{-1}Lc is our pointwise approximation of the Laplace-Beltrami operator.
    
    #unsigned_curvature = np.sqrt(np.sum(np.square(delta_coordinates), axis=1))
    delta_gauss_map = np.sum(delta_coordinates*gauss_map, axis = -1)
    #sign = 2*(delta_gauss_map>0) - 1
    
    #return sign*unsigned_curvature
    return delta_gauss_map


def get_differential_coordinates(verts, laplacian):
    """
    Computes the differential coordinates, the vertex normals and mean curvature
    
    laplacian = (Lc, B, M, A, A1r), as per the output of compute_mesh_laplacian
    """
    
    Lc, B, M, A, A1r = laplacian
    delta = Lc.dot(verts)/A1r[:, np.newaxis]
    
    return delta