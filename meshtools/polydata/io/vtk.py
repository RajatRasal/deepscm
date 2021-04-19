import sys
import os
import re

import pyvista as pv

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
    
The code below is simply meant to load/save vtk polydata from/to the disk.
"""

def generate_dataset_filenames(root_dir, structures = {}, extension = ""):
    """
    Generate a list of filenames from a root folder, by going recursively through subfolders.
    
    If a dictionary of structures is passed, only filenames that match one of the values (regex)
    are retrieved; and the output is a list of (key, filenames)-dictionaries.
    
    Otherwise the output is just a list of filenames.
    
    The filenames are returned as a filename/path relative to the root_dir.
    
    root: path to a root folder
    """  
    
    if not structures:
        result = []
        for root, dirs, files in os.walk(root_dir):
            rel_dir = os.path.relpath(root, root_dir)
            result.extend([os.path.join(rel_dir, file) for file in files])
        return result
    
    else:
        result = []
        
        for root, dirs, files in os.walk(root_dir):
            rel_dir = os.path.relpath(root, root_dir)
            subject_output = {}

            for name in files:
                for key in structures:
                    if re.search(structures[key], name) and (extension in name):
                        subject_output[key] = os.path.join(rel_dir, name)
                        break

            if subject_output:
                result.append(subject_output)
            
        return result
    

def read_PolyData(filename):
    """
    Read a vtk polydata file and computes the normal at each point.    
    filename: e.g., 'Filename.vtk'
    """    
    return pv.PolyData(filename)


def write_PolyData(polydata, filename, binary=False):
    """
    Write the vtk polydata data into the file filename.
    
    data: e.g., pv.PolyData type
    filename: e.g., 'Filename.vtk' or '*.ply' or whatever. Extension inferred from filename.
    binary: False to save in ASCII.
    """
    
    os.makedirs(os.path.dirname(filename), exist_ok = True)    
    polydata.save(filename, binary)
