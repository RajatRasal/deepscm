import numpy as np
from scipy import sparse


def edge_to_faces_dictionary(tris):
    result = {}
    
    for (f,tri) in enumerate(tris):
        e01 = (min(tri[0],tri[1]),max(tri[0],tri[1]))
        e12 = (min(tri[1],tri[2]),max(tri[1],tri[2]))
        e20 = (min(tri[0],tri[2]),max(tri[0],tri[2]))
        if e01 in result:
            result[e01][0].append(f)
            result[e01][1].append(tri[2])
        else:
            result[e01] = [[f],[tri[2]]]
        if e12 in result:
            result[e12][0].append(f)
            result[e12][1].append(tri[0])
        else:
            result[e12] = [[f],[tri[0]]]        
        if e20 in result:
            result[e20][0].append(f)
            result[e20][1].append(tri[1])
        else:
            result[e20] = [[f],[tri[1]]]
            
    bad_edge = [len(tris_e[0])!=2 for (e,tris_e) in result.items()]    
    if sum(bad_edge):
        raise RuntimeError("The mesh geometry is corrupted (non-manifold). There are edges without two faces attached.")
        
    return result


def get_edge_incidences(tris):    
    EtF = edge_to_faces_dictionary(tris)
    
    temp = [(e[0],e[1],tris_e[0][0],tris_e[0][1],tris_e[1][0],tris_e[1][1]) for (e,tris_e) in EtF.items()] 
    result = np.array([e for t in temp for e in t]).reshape((-1,6))
    
    return np.hsplit(result,3) # per edge (i.e. on each row), we get: the two incident vertices, the two adjacent faces,
                            # and the two "neighbouring" vertices (i.e. the remaining one, not connected to the edge,
                            # for each adjacent triangle)

        
def map_face_edges(IF, IV, tris):
    """
    Outputs:
    
    FE: (m, 3) array of face edges, gives the index of face edges (w.r.t. (o,d) edge-based arrays) 
    """
    
    # Misc.
    o = len(IF)
    m = len(tris)
    
    # 1.
    face_edges = np.zeros((m,3), dtype=int)
    position = np.zeros(m, dtype=int)
    for e in range(o):
        f = IF[e,0]
        face_edges[f,position[f]] = e
        position[f] += 1
        
        f = IF[e,1]
        face_edges[f,position[f]] = e
        position[f] += 1
        
    return face_edges

        
def orient_triangles(IV, IF, tris, FE = None):
    """
    Takes as input the edge incidences (face and vertex), as well as a set of (non-oriented) triangles.
    Sort tris in place so that it is (arbitrarily) oriented. 
    
    Assumes that the graph is globally orientable, and assumes a single connected component 
    (the latter constraint would be easy to remove)
    
    Return tris (same container as the input - modified in place).
    
    Optionally return face edges in FE (just pass an empty list as input), 
    or uses face edges passed in FE (pass the face edge array as a 1-element list).
    """
    
    # Simple idea: each edge should be traversed in opposite directions on its two adjacent faces.
    
    # Misc.
    o = len(IV)
    m = len(tris)
    
    # Map face edges
    if FE:
        face_edges = FE[0]
    else:
        face_edges = map_face_edges(IV, IF, tris)
        
        if FE == []:
            FE.append(face_edges)
    
    # 
    oriented_face = np.zeros(m, dtype=bool)
    dangling_edge_stack = [0]
    
    while dangling_edge_stack:
        e = dangling_edge_stack.pop()       
        v = IV[e,0]
        v_ = IV[e,1]
        
        # Which face do we need to orient?
        f = IF[e,0]
        if not oriented_face[f]:
            f_ref = IF[e,1] # We'll try to use the other face to decide the orientation
        else:
            f_ref = f # We'll use this face to decide the orientation
            f = IF[e,1] # We'll orient that face instead
        
        # At this stage, we know for sure f != f_ref
        forward = True # whether to cross the edge in reverse or forward in the face f that we are going to orient
        
        if oriented_face[f_ref]:
            # Is e crossed in the forward or reversed direction in the already oriented face?
            i1, i2, i3 = tris[f_ref,:]
            
            if i1==v:
                if i3 != v_:
                    # crossed in forward direction in f_ref, so to be crossed in reverse next
                    forward = False
                #else: forward = True
            elif i1==v_:
                if i2 != v:
                    # crossed in forward direction in f_ref, so to be crossed in reverse next
                    forward = False
                #else: forward = True
            else:
                # i1 != v, v_
                if i2 != v_:
                    # crossed in forward direction in f_ref, so to be crossed in reverse next
                    forward = False
                #else: forward = True
                
        # Take care of f's orientation
        i1, i2, i3 = tris[f,:]
        if i1==v:
            if i3 != v_: # i2 = v_
                if not forward:
                    tris[f,1] = i3
                    tris[f,2] = v_
                # else nothing to do
            else: # i3 = v_
                if forward:
                    tris[f,1] = v_
                    tris[f,2] = i2
                # else we're good
        elif i1==v_:
                if i2 != v: # i3 = v
                    if not forward:
                        tris[f,1] = v
                        tris[f,2] = i2
                else: # i2 = v
                    if forward:
                        tris[f,2] = v
                        tris[f,1] = i3
        else:
            # i1 != v, v_
            if i2 != v:
                if forward:
                    tris[f,1] = v
                    tris[f,2] = v_
            else:
                if not forward:
                    tris[f,1] = v_
                    tris[f,2] = v
                    
        # All done for face f's orientation.
        oriented_face[f] = True
        
        # We just need to mark its edges as dangling if necessary
        for e_ in face_edges[f]:
            if IF[e_,0] != f:
                f_ = IF[e_,0]
            else:
                f_ = IF[e_,1]
            if not oriented_face[f_]:
                dangling_edge_stack.append(e_)
                
        # And now we can move on to the next edge in the stack of dangling edges
        
    # All done
    return tris


def reverse_orientation(tris):
    """
    Reverses the orientation of triangles throughout the mesh.
    
    !! This breaks the consistency of the FE array with the tris array (see order_face_edges to restore it) !! 
    
    """
    
    # Switch the last two columns so that instead of i1-i2-i3 we now follow i1-i3-i2
    tris[:,[1, 2]] = tris[:, [2, 1]]   
    return tris


def order_face_edges(FE, IV, tris):
    """
    Make the order of face edges FE consistent with the order of vertices in tris (acts in place on FE, returns FE).
    
    If i1, i2, i3 are the 3 vertex indices, then
        e1: i2 - i3
        e2: i3 - i1
        e3: i1 - i2
    (here we are not interested in the sign.) 
    
    Typically, you want to call this after orienting the triangles since changing the ordering in tris will break the
    consistency with any current FE.
    """
    
    # The idea: e3 = i1-i2 is characterized by not using i3; and likewise for the others e2->i2, e1->i1
    
    # Misc.
    m = len(tris)
    i2 = tris[:,1]
    i3 = tris[:,2]
        
    # Compute which edge is which    
    FEV = IV[FE] # 3d tensor: 1st faces, 2nd edges, 3rd vertices (m*3*2)
    face_edge_order = np.zeros((m,3),dtype=int) # for each face and each edge, says 0 if it is e1, 1 if e2, 2 if e3
    
    i2_ = np.broadcast_to(i2.reshape((m,1,1)),(m,3,2))    
    test = np.sum(FEV - i2_ != 0, axis=-1) - 1 # now (m,3) with only one non zero value per row
        # sum over edge vertices, only one of the 3 face edges has value 1, the one which didn't have i2 (others have 0)
    face_edge_order[test.nonzero()] = 1
    
    i3_ = np.broadcast_to(i3.reshape((m,1,1)),(m,3,2))    
    test = np.sum(FEV - i3_ != 0, axis=-1) - 1
    face_edge_order[test.nonzero()] = 2

    # Swap
    FE_ = FE.copy()
    FE[range(m),face_edge_order[:,0]] = FE_[:,0]
    FE[range(m),face_edge_order[:,1]] = FE_[:,1]
    FE[range(m),face_edge_order[:,2]] = FE_[:,2]
    
    #
    return FE


def orient_edge_incidences(IV, IF, FE, tris):
    """
    Order the two adjacent faces consistently given the direction in which eij points and the mesh orientation.
    Specifically, IF[:,0] should always be the face that traverses the edge in the orientation given by IV[:,0]->IV[:,1],
    given its current orientation (as given by the vertex cycling order).
    
    We correct the order of entries in IV if necessary for this to hold, in place. Returns IV.
    
    Assumes that the order of edges in FE is consistent with the vertex ordering in tris. 
    Call order_face_edges() beforehand if necessary.    
    """
    
    # Actually, let's do it here:
    FE = order_face_edges(FE, IV, tris)
    
    #
    o = len(IV)
    F = IF[:,0]
    test = (FE[F,:] - np.arange(o)[:,np.newaxis]) == 0
    face_edge_number = test.nonzero()[1] # 0, 1, or 2 for each edge in 0... o-1
    
    # e1 = i2 - i3, e2 = i3 - i1, e3 = i1 - i2 
    # The face orientation is i1 -> i2 -> i3 -> i1
    # So edges IV have the same orientation as face F=IF[:,0] if IV[:,0] is equal to
    #     i2 when FE[F,:]=0
    #     i3 when FE[F,:]=1
    #     i1 when FE[F,:]=2
        
    reverse = IV[:,0] != tris[:, [1, 2, 0]][F, face_edge_number]
    reverse = reverse.astype(int)
    
    IV_ = IV.copy()
    IV[:,0] = IV_[range(o),reverse]
    IV[:,1] = IV_[range(o),1-reverse]
    
    return IV


def get_face_edge_signs(FE, IF):
    """
    Returns an array (m,3) of integers with value
    [*] -1 if the face edge points in the other direction than the one in the corresponding one
            in the (o,2) edge vertex incidence array; i.e. have to change the sign of Im(HE)
    [*] 1 otherwise
    
    We take oriented face edges to be i2->i3, i3->i1, i1->i2.
    We take oriented edges as recorded in the IV array to be IV[:,0]->IV[:,1]. 
    """
    
    # Straightforward now that we don't have crappy legacy triangle edge definitions
    reverse = IF[FE,0] != np.arange(len(FE)).reshape((-1,1))
    return 1-2*reverse


def compute_face_laplacian(IF, IV, tris, FE):
    # Misc.
    m = len(tris)
            
    #
    f1 = IF[:,0]
    f2 = IF[:,1]
    
    Lf_ij = np.repeat(np.array([-1,-1,1,1])/3,len(IF))
    I = np.r_[f1, f2, f1, f2]
    J = np.r_[f2, f1, f1, f2]
    
    qu_I = I.reshape((len(I),1))*4 + np.arange(4)
    qu_J = J.reshape((len(J),1))*4 + np.arange(4)
    qu_Lf_ij = np.repeat(Lf_ij,4)
    
    #
    Lf = sparse.coo_matrix((Lf_ij, (I, J)), shape=(m, m))
    Lf.sum_duplicates()
    
    qu_Lf = sparse.coo_matrix((qu_Lf_ij, (qu_I.reshape(-1), qu_J.reshape(-1))), shape=(m*4, m*4))
    qu_Lf.sum_duplicates()
    
    return Lf.tocsc(), qu_Lf.tocsc()


def vertex_averaging_operator(tris, n):
    """
    Matrix of vertex to face averaging. 
    
    Consistent with a finite element interpretation: integral over face is A_f/3 * sum_v value(v), so
    the dimensionless density on the face is avg(value(v)). 
    
    For us if phi solves D phi = rho A phi, phi is dimensionless too ([rho] = [mean curvature density]).
    """
    
    # Misc.
    m = len(tris)
    
    # Data
    I = np.arange(m).repeat(3)
    J = tris.reshape(-1)
    Avg = np.full(m*3,1/3)
    
    qu_I = I.reshape((len(I),1))*4 + np.arange(4)
    qu_J = J.reshape((len(J),1))*4 + np.arange(4)
    
    qu_I = qu_I.reshape(-1)
    qu_J = qu_J.reshape(-1)
    qu_Avg = np.full(m*3*4,1/3)
    
    # Build the matrices
    Avg_v = sparse.csr_matrix((Avg,(I,J)), shape=(m,n))
    qu_Avg_v = sparse.csr_matrix((qu_Avg,(qu_I,qu_J)), shape=(m*4,n*4))
    
    return Avg_v, qu_Avg_v


def edge_averaging_operator(FE, m):
    """
    Matrix of edge to face averaging. 
    
    Quantities living on edges are typically integral quantities already. So we just average with equal weights.
    (e.g. the hyperedges, or Gf(e=v->v')=f(v')-f(v) with G gradient and f a function over vertices) 
    """
    
    # Misc.
    o = len(FE)
    
    # Data
    I = np.arange(m).repeat(3)
    J = FE.reshape(-1)
    Avg = np.full(m*3,1/3)
    
    qu_I = I.reshape((len(I),1))*4 + np.arange(4)
    qu_J = J.reshape((len(J),1))*4 + np.arange(4)
    
    qu_I = qu_I.reshape(-1)
    qu_J = qu_J.reshape(-1)
    qu_Avg = np.full(m*3*4,1/3)
    
    # Build the matrices
    Avg_e = sparse.csr_matrix((Avg,(I,J)), shape=(m,o))
    qu_Avg_e = sparse.csr_matrix((qu_Avg,(qu_I,qu_J)), shape=(m*4,o*4))
    
    return Avg_e, qu_Avg_e


def gradient_operator(IV, n):
    o = len(IV)
    
    data = np.c_[-np.ones(o, dtype=float),np.ones(o, dtype=float)]
    I = np.arange(o).repeat(2)
    J = IV.copy()
    
    # qu_I = I.reshape((-1,1))*4 + np.arange(4)
    # qu_J = J.reshape((-1,1))*4 + np.arange(4)
    # qu_data = data.reshape(-1).repeat(4)
    
    G = sparse.csr_matrix((data.reshape(-1), (I,J.reshape(-1))), shape=(o,n)) 
    # qu_G = sparse.csr_matrix((qu_data, (qu_I.reshape(-1), qu_J.reshape(-1))), shape=(o*4,n*4))
    
    return G


def curl_operator(FE, o, FE_sign=None):
    m = len(FE)
    
    if FE_sign:
        switch = FE_sign[0]
    else:
        switch = get_face_edge_signs(FE, IF)        
        if FE_sign == []:
            FE_sign.append(switch)
            
    data = switch.copy().astype(float)
    I = np.arange(m).repeat(3)
    J = FE.copy()
    
    # integrated curl (not normalized by A_F)
    C = sparse.csr_matrix((data.reshape(-1), (I, J.reshape(-1))), shape=(m,o))
    
    return C


def curl_adjoint_operator(IF, m):
    o = len(IF)
    
    data = np.c_[np.ones(o, dtype=float), -np.ones(o, dtype=float)]
    I = np.arange(o).repeat(2)
    J = IF.copy()
    
    # integrated curl adjoint (not normalized by cotan weights)
    C_adjoint = sparse.csr_matrix((data.reshape(-1), (I, J.reshape(-1))), shape=(o,m))
    
    return C_adjoint


def get_adjacent_faces(IF, FE, switch):
    adjacent_id = (switch != -1).reshape(-1) 
        # the "adjacent" triangle has index one in IF[:,] if switch==1 (reverse=0)
    return IF[FE.reshape(-1),adjacent_id.reshape(-1).astype(int)].reshape((-1,3))


def get_dual_cells(IV, IF, FE, n):
    """
    quick & dirty temporary implementation
    """
    
    #
    dual_cells = {i: [] for i in range(n)}    
    
    #
    o = len(IV)
    for e in range(o):
        c = IV[e, 0]
        
        if dual_cells[c]:
            continue
        
        f = IF[e,1]
        polygon = [f]
                
        e_ = e    
        f_ = IF[e_,0]
        while(f_ != f):
            polygon.append(f_)
            face_edges = FE[f_,:]
            for fe in face_edges:
                if (fe!=e_) and (np.sum(IV[fe,:]==c)==1):
                    break
            
            e_ = fe
            if IF[e_,0]==f_:
                f_ = IF[e_,1]
            else:
                f_ = IF[e_,0]
            
        dual_cells[c].extend(polygon)       
        
    return dual_cells