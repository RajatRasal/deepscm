# This file is partially adapted from the NVIDIA kaolin library, available at
# https://github.com/NVIDIAGameWorks/kaolin
# specifically kaolin/metrics/mesh.py.

# It was modified to use the trimesh implementation instead on CPU, as well as a few
# necessary adjustments to dispatch to the correct implementation depending on the platform
# and for ease of building (at the time of writing).

import torch
from ._points2surface import _points2triangulation_cuda, _points2triangulation_cpu


def points_to_surface(points: torch.Tensor, verts: torch.Tensor, tris: torch.Tensor, mesh=None):
    r"""Computes the minimum distances from a set of points to a mesh
    Args:
            points (torch.Tensor): set of points (n, 3)
            verts (torch.Tensor): mesh vertices (N, 3)
            tris (torch.LongTensor): mesh faces (m, 3)
            mesh (optional): the trimesh.Trimesh with exactly verts and tris as verts and tris, can save
                the cost of rebuilding the mesh. [only relevant on CPU]
    Returns:
            distances: distance between each point (in points) and surface
    """

    # extract triangle vertex coordinates from verts and tris
    v1 = torch.index_select(verts, 0, tris[:, 0])
    v2 = torch.index_select(verts, 0, tris[:, 1])
    v3 = torch.index_select(verts, 0, tris[:, 2])

    # not backpropable
    with torch.no_grad():
        if points.is_cuda:
            closest_id, dist_type = _points2triangulation_cuda(points.detach(), v1.detach(), v2.detach(), v3.detach())
        else:
            closest_id, dist_type = _points2triangulation_cpu(points.detach(), mesh, verts.detach(), tris.detach())
    
    # distances backpropable onto points and verts
    return _points_to_closest_triangle_sqdist(points, [v1, v2, v3], closest_id, dist_type), closest_id, dist_type


def _points_to_closest_triangle_sqdist(p, verts, closest_triangle_ids, dist_type):
    # recompute surface based on the calculated correct assignments of points and triangles
    # and the type of distance, type 1 to 3 indicates which edge to calculate to,
    # type 4 indicates the distance is from a point on the triangle, not an edge.
    
    #
    v1, v2, v3 = verts
    
    #
    v1 = v1[closest_triangle_ids]
    v2 = v2[closest_triangle_ids]
    v3 = v3[closest_triangle_ids]

    type_1 = (dist_type == 0)
    type_2 = (dist_type == 1)
    type_3 = (dist_type == 2)
    type_4 = (dist_type == 3)

    #
    v21 = v2 - v1
    v32 = v3 - v2
    v13 = v1 - v3

    p1 = p - v1
    p2 = p - v2
    p3 = p - v3

    #
    output = torch.empty_like(p[:,0])  
    if type_1.any():
        output[type_1] = _compute_edge_sqdist(v21[type_1], p1[type_1])
    if type_2.any():
        output[type_2] = _compute_edge_sqdist(v32[type_2], p2[type_2]) 
    if type_3.any():
        output[type_3] = _compute_edge_sqdist(v13[type_3], p3[type_3])     
    if type_4.any():         
        nor = torch.cross(v21[type_4], v13[type_4])
        output[type_4] = _compute_planar_sqdist(nor, p1[type_4])    

    return output


def _compute_edge_sqdist(v, p):
    # batched distance between an edge and a point
    if v.shape[0] == 0:
        return v
    # again not clear that _compute_dot using bmm performs better
    dots = (v*p).sum(-1)
    sqnorms = (v**2).sum(-1)
    dots = torch.clamp(dots/sqnorms, 0.0, 1.0).view(-1, 1)
    dots = v*dots - p
    return (dots**2).sum(-1)

def _compute_planar_sqdist(n, p):
    # it is not clear that bmm or einsum perform better
    return ((n*p).sum(-1))**2 / (n**2).sum(-1)

def _compute_dot(p1, p2):
    # batched dot product
    return torch.bmm(p1.view(p1.shape[0], 1, 3),
                     p2.view(p2.shape[0], 3, 1)).view(-1)