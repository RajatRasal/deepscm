import numpy as np
from ...examples import numpyToPolyData
from .surface_distance import surface_distance_filter


def align(verts, tris, VA, target_verts=None, target_tris=None, translation_only=False):
    """
    target_verts are expected to be already in their canonical position (for example by running this routine on a 
    template shape with target_verts=None, target_tris=None, and using the aligned template as a target from there onwards)
    """
    
    barycenter = np.average(verts, weights=VA, axis=0)
    centered_verts = verts-barycenter
    
    if translation_only:
        return centered_verts
    
    # Make the principal directions match with x y z, highest magnitude with x and lowest with z.
    # (and of course centroids to 0)       
    svd = np.linalg.svd(centered_verts*np.sqrt(VA).reshape((-1,1)), full_matrices=False)       
    assert(np.all(svd[1]>0))
    
    #
    R = svd[2]    
    if np.linalg.det(R) < 0: # could be a roto-inversion
        R[0] *= -1 # clearly not a unique choice, but if it matters it will be accounted for below
    rotated_verts = centered_verts @ R.T
    
    if target_verts is not None:
        # Might want to flip directions to minimize the error, try exhaustively the ones that preserve orientation
        flips = [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]
        best_flip = None
        best_error = np.inf
        
        source_polydata = numpyToPolyData(rotated_verts, tris)
        target_polydata = numpyToPolyData(target_verts, target_tris)
        print(target_polydata)

        for flip in flips:
            source_polydata.points = rotated_verts*flip
            source_polydata = surface_distance_filter(source_polydata, target_polydata, inplace=True)
            squared_distances = np.square(source_polydata.point_arrays['Distance'])
            error = np.average(squared_distances, weights=VA)
            if error<best_error:
                best_error = error
                best_flip = flip

        rotated_verts = rotated_verts*best_flip
    
    return rotated_verts
    