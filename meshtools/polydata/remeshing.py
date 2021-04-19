import pyvista as pv
import numpy as np
import pyacvd
from ..polydata import utils


class Remesher(object):
    """
    To generate a mesh with finer resolution from a coarser mesh.
    
    It's amazing. The quality of the PyACVD remeshing has actually gone done since it moved to pyvista.
    However, the running time of the code as improved and it appears that the algorithm now outputs exactly the 
    requested number of nodes.
    My understanding is that cluster centers are now probably fixed to a subset of point coodinates from the original mesh,
    rather than optimizing over the location. This means that the mesh has to be subdivided enough times beforehand for the
    algorithm to have a chance to perform adequately. This would explain why the examples now show .subdivide(n).
    
    Not very much to my liking, but I have introduced a heuristic to do the subdivision, so that at least we make sure that
    we have enough nodes for the requested nclus, and then some. Really unfortunately, the 'and then some' is tied to mesh
    quality, so the best is to go overboard if unsure as the code is fast and it makes a big difference.
    
    (Doesn't preserve the original point locations.)
    """
    
    def __init__(self):
        self._num_points_per_unit_area = 1.0
        
    def set_num_points_per_unit_area(self, num_points_per_unit_area):
        self._num_points_per_unit_area = num_points_per_unit_area
        
    def set_num_points_per_unit_area_to_target(self, mesh):
        mesh = utils.compute_face_normals_and_areas(mesh)
        mesh_area = np.sum(mesh.cell_arrays['Areas'])
        self._num_points_per_unit_area = mesh.n_points / mesh_area
    
    def remesh(self, input_polydata, nclus = 0, nsubdivide = 5):
        """
        Remesh to desired resolution.
        """
        
        if nclus == 0:
            # Compute the number of clusters for the algo.
            input_polydata = utils.compute_face_normals_and_areas(input_polydata)
            total_area = np.sum(input_polydata.cell_arrays['Areas'])
            nclus = np.ceil(total_area * self._num_points_per_unit_area).astype('int')
            
        nsubdivide += int(np.ceil(np.log(nclus/input_polydata.n_points)/np.log(2))) - 1
        
        # Create clustering object
        input_polydata.triangulate(inplace=True)
        clus = pyacvd.Clustering(input_polydata)
    
        # Generate clusters
        clus.subdivide(nsubdivide)
        clus.cluster(nclus)
    
        # Generate uniform mesh
        output_polydata = clus.create_mesh()
        
        # Edit. This comment is old, but I don't want to find out again.
        # Turns out that this little culprit right here does not preserve normal consistency (I think :P)
        # So to be on the safe side, and because it already uses Polydata anyway, we'll pass it through
        # an orientation filter
        # Turns out that it may ALSO slap two identical faces on top of each others! Fix that first if
        # possible, then check consistency.           
        return utils.clean_PolyData(output_polydata)