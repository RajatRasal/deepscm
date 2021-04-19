from pyvista import Sphere
from ...examples import PolyDataToNumpy
from ...examples import Remesher


def sphere(r=1, theta_res=64, phi_res=64, remesh=False, npts=256):
    poly = Sphere(r, theta_resolution=theta_res, phi_resolution=phi_res)
    
    if remesh:
        remesher = Remesher()
        poly = remesher.remesh(poly, nclus = npts)
        
    return PolyDataToNumpy(poly)