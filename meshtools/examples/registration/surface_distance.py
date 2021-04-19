import vtk
import pyvista as pv

def surface_distance_filter(source_polydata, target_polydata, signed=True, inplace=False):    
    distance_filter = vtk.vtkDistancePolyDataFilter()
    distance_filter.NegateDistanceOn()
    distance_filter.SetInputData(0, source_polydata)
    distance_filter.SetInputData(1, target_polydata)
    if not signed:
        distance_filter.SignedDistanceOff()
    
    distance_filter.Update()

    mesh = pv.filters._get_output(distance_filter) # the array is named 'Distance'
    if inplace:
        source_polydata.overwrite(mesh)
        return source_polydata
    else:
        return mesh