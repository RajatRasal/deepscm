import pyvista as pv
import numpy as np
from ..polydata import _get_face_normals_and_areas
from .polydata import PolyDataToNumpy, numpyToPolyData


def clean_PolyData(polydata):
    """
    Removes duplicate points, duplicate faces stuck on top of each others.   
    If all is good, use orient_vtkPolyData to do the vertex/polygon ordering consistency check.
    
    This is all heuristic, just takes care of bugs introduced in the potentially 
    non-manifold output of PyACVD, for my use case.
    
    Give up if after doing all of that, there are edges with more than two faces.
    
    Returns a new polydata with the clean geometry.
    """
    
    verts, tris = PolyDataToNumpy(polydata)
    
    # Remove duplicate points, reroute triangle vertex indices if needed
    verts, vert_representers = np.unique(verts, return_inverse=True, axis=0)
    tris = vert_representers[tris]
    
    # Remove duplicate triangles
    reordered_tris, fs = np.unique(tris,return_index=True,axis=0)
    tris = tris[np.sort(fs),:] 
        # don't use reordered_tris, which doesn't have this nice spatially coherent way of going through the mesh
        # instead trim/remove the duplicate triangles' rows.
        # purely aesthetics e.g., when looking at a face mapping after remapping a coarse mesh;
        # might have a computational impact also not to be spatially contiguous but not in my routines so far afaik.
        
    # Check for points that are attached to a unique face (e.g. due to collapsed faces at extremities)
    num_faces_attached = np.zeros(len(verts))
    
    nf1 = np.bincount(tris[:,0])
    nf2 = np.bincount(tris[:,1])
    nf3 = np.bincount(tris[:,2])
    num_faces_attached[:len(nf1)] += nf1
    num_faces_attached[:len(nf2)] += nf2
    num_faces_attached[:len(nf3)] += nf3
    
    collapsed_corner = num_faces_attached == 1
    tris_to_keep = np.sum(collapsed_corner[tris],axis=1) == 0
    tris = tris[tris_to_keep,:]
    
    # Remove unused verts
    used_vert_ids, tris = np.unique(tris, return_inverse=True) # unique ids, sorted in ascending order , and the remapped tris
    tris = tris.reshape(-1,3)   
    verts = verts[used_vert_ids,:]
    
    # Check for bad edges, throw an error if there are such edges
    edge_to_triangles = {}
    for (f,tri) in enumerate(tris):
        e01 = (min(tri[0],tri[1]),max(tri[0],tri[1]))
        e12 = (min(tri[1],tri[2]),max(tri[1],tri[2]))
        e20 = (min(tri[0],tri[2]),max(tri[0],tri[2]))
        if e01 in edge_to_triangles:
            edge_to_triangles[e01].append(f)
        else:
            edge_to_triangles[e01] = [f]
        if e12 in edge_to_triangles:
            edge_to_triangles[e12].append(f)
        else:
            edge_to_triangles[e12] = [f]        
        if e20 in edge_to_triangles:
            edge_to_triangles[e20].append(f)
        else:
            edge_to_triangles[e20] = [f]
            
    bad_edge = [len(tris_e)>2 for (e,tris_e) in edge_to_triangles.items()]
    
    if sum(bad_edge):
        # This can actually happen for instance if two faces at an extremity of the object collapse into a single
        # one after remeshing, i.e. the extreme point is attached to a single face. Might fix this later.
        raise RuntimeError("The mesh geometry is corrupted (non-manifold). " +
        "Auto-attempt to recover a correct geometry failed. There are edges with more than two faces attached.")
    
    # Create a new vtk polydata, run it through orientation consistency checks
    clean_polydata = numpyToPolyData(verts, tris)
    return orient_PolyData(clean_polydata)  
    

def orient_PolyData(polydata):
    """
    Guarantees that the ordering of vertices is consistent to have a consistent orientation
    of normals throughout the mesh.
    """
    
    return polydata.compute_normals(
                        split_vertices=False, flip_normals=False,
                        consistent_normals=True,
                        auto_orient_normals=True,
                        non_manifold_traversal=True)


def compute_face_normals_and_areas(polydata):
    FN, FA = _get_face_normals_and_areas(*PolyDataToNumpy(polydata))  
    polydata.cell_arrays['normals'] = FN
    polydata.cell_arrays['Areas'] = FA
    
    return polydata