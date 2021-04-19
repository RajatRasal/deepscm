import numpy as np
import pyvista as pv
import itertools
"""
vtk polydata is the internal structured mesh (polydata) representation used by the c/python vtk library.
vtk has good compatibility with the Paraview viewer. Both tools are developed by the same team (Kitware).
vtk implements many so-called "filters" to do standard mesh processing, although we may not need most or any of them in this project. A filter is simply something that takes e.g., a mesh, as input and outputs e.g., another mesh modified in some way.

Some of those filters are implemented in paraview and can be used and their result visualized on the fly.

vtkPolydata stores information about 
1) the mesh combinatorics, i.e. number of nodes (also called vertices) 
    and their connectivity (i.e. edges connecting a subset of nodes);
    vtk is a fairly generic library, it can manipulate a number of higher level abstractions, the most obvious one
    is called a mesh cell (also, a face): in our case cells are triangular faces.
2) the mesh geometry, i.e. the 3D world coordinates of vertices (x,y,z for each vertex)
3) any additional data "living" on the mesh in the form of data arrays; the data can be attached to vertices (PointData) or to
    cells (CellData).

The code below is meant to bring the mesh combinatorics/geometry to a more palatable numpy format. Similarly to what vtk
does internally, we can summarize the mesh information in two arrays:
1) verts, an array of vertex coordinates (x,y,z); this encodes the geometry of the mesh.
2) tris, an array encoding the mesh combinatorics by specifying a list of triangular faces; 
    each row consists of three integer values i,j,k referring to the index of the corresponding vertices in the "verts" array.

From these two arrays, one can infer edges and compute a variety of geometric operators or invariants. This is done in separate
files.

[edit. We now use a wrapper library called pyvista that handles under the hood the conversion between numpy arrays and
vtk data structures. Some of the routines below have been almost obviated!]

"""


def PolyDataToNumpy(polydata):
    """
    Processes a pyvista polydata structure into vertices, triangles with deep copy.
    
    To avoid deep copy and keep a link between numpy arrays and polydata points/faces, use the built-in functionalities.
    For points, the getter and setter:
        polydata.points
    
    For faces, also the getter and setter
        polydata.faces
        
    Note that faces doesn't assume that all faces are triangles. It is a more generic array that looks like
    [n1,v1_{1},...,v1_{n1}, n2,v2_{1},...,v2_{n2}, etc.]
    where ni is the number of vertices on face i, and vi_{k} is the vertex index for the kth vertex on face i.
    
    Our code however assumes we have triangles, and converts faces to a "tris" array of shape
    [[v1_{1},v1_{2},v1_{3}],
     [v2_{1},v2_{2},v2_{3}],
     etc.]
    
    input: a pv.PolyData
    """

    numpy_verts = np.copy(polydata.points).reshape(-1,3)
    
    numpy_tris = np.copy(polydata.faces)
    if (numpy_tris.size) != (polydata.n_faces*4):
        raise ValueError("Function only supports Polydata with triangle faces.")  
       
    numpy_tris = numpy_tris.reshape(-1,4)
    numpy_tris = numpy_tris[:,1:]   
    
    return numpy_verts, numpy_tris


def numpyToPolyData(verts, tris):
    """
    Turn a set of vertices and triangles back into a PolyData structure.
    
    verts: (n,3) float np.array
    tris: (m,3) int np.array of vert indices for each triangle
    """
    
    # Numpy to vtk. The point data is straightforward
    # For polygons, we need to specify the number of vertices per face
    faces = np.c_[np.full((tris.shape[0],), 3, dtype=np.int), tris]       
    return pv.PolyData(verts, faces, deep = True)





def PolyDataArrayDataToNumpy(polydata, array_names = []):
    """
    Retrieves the relevant point data from a PolyData input,
    and returns a K-key dict of (n,) arrays, 
    where n is the number of vertices in the polydata and K the number of arrays to retrieve.
    
    Does a deep copy.
    
    To preserve a link between the numpy data and the underlying point/cell data instead, 
    just use the built-in @property's:
        polydata.cell_arrays
        polydata.point_arrays
    e.g. polydata.point_arrays['array_name'] to get or set. Can also be used to add or remove arrays (directly with np arrays).
    
    array_names: list of strings
    """
    
    point_data = polydata.point_arrays
    cell_data = polydata.cell_arrays
    field_data = polydata.field_arrays
    
    result = {}
    
    if len(array_names) == 0:
        # retrieve everything
        array_names = list(itertools.chain.from_iterable(
            point_data.keys(), 
            cell_data.keys(),
            field_data.keys()))

    for name in array_names:
        result[name] = np.copy(polydata.get_array(name))
        #if name in point_data.keys():
        #    result[name] = point_data[name].copy()
        #elif name in cell_data.keys():
        #    result[name] = cell_data[name].copy()
        #elif name in field_data.keys():
        #    result[name] = field_data[name].copy()
        #else:
        #    raise ValueError("Input PolyData has no array named " + array_name)

    return result


def numpyPointDataToPolyData(polydata, data_dict):
    for name, array in data_dict.items():
        #paraview_compatible_array = np.multiply(array, np.abs(array)>1e-10) 
        polydata.point_arrays[name] = array
    
    return polydata

def numpyTextureDataToPolyData(polydata, data):
    #paraview_compatible_array = np.multiply(data, np.abs(data)>1e-10)
    polydata.t_coords(data)
    
    return polydata

def numpyCellDataToPolyData(polydata, data_dict):
    for name, array in data_dict.items():
        #paraview_compatible_array = np.multiply(array, np.abs(array)>1e-10) 
        polydata.cell_arrays[name] = array
    
    return polydata