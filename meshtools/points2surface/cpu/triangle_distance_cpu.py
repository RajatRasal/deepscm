# This is a direct adaptation of trimesh's code
# Micro-changes were needed to store the "dist_type" as defined in the kaolin inspired code,
# for consistency between cpu and cuda variants.
from collections import deque

import numpy as np
import trimesh
from trimesh import util
from trimesh.constants import tol
from trimesh.proximity import nearby_faces


def forward_cpu(points, mesh, verts, tris):
    if mesh is not None:
        return closest_point(mesh, points)
    return closest_point(trimesh.Trimesh(verts, tris, process=False), points)

def closest_point(mesh, points):
    """
    Given a mesh and a list of points find the closest point
    on any triangle.
    Parameters
    ----------
    mesh   : trimesh.Trimesh
      Mesh to query
    points : (m, 3) float
      Points in space
    Returns
    ----------
    closest : (m, 3) float
      Closest point on triangles for each point
    distance : (m,)  float
      Distance
    triangle_id : (m,) int
      Index of triangle containing closest point
    """
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    # do a tree- based query for faces near each point
    candidates = nearby_faces(mesh, points)
    # view triangles as an ndarray so we don't have to recompute
    # the MD5 during all of the subsequent advanced indexing
    triangles = mesh.triangles.view(np.ndarray)

    # create the corresponding list of triangles
    # and query points to send to the closest_point function
    query_point = deque()
    query_tri = deque()
    for triangle_ids, point in zip(candidates, points):
        query_point.append(np.tile(point, (len(triangle_ids), 1)))
        query_tri.append(triangles[triangle_ids])

    # stack points into an (n,3) array
    query_point = np.vstack(query_point)
    # stack triangles into an (n,3,3) array
    query_tri = np.vstack(query_tri)

    # do the computation for closest point
    query_close, query_type = closest_point_corresponding(query_tri, query_point)
    query_group = np.cumsum(np.array([len(i) for i in candidates]))[:-1]

    distance_2 = ((query_close - query_point) ** 2).sum(axis=1)

    # find the single closest point for each group of candidates
    result_close = np.zeros((len(points), 3), dtype=np.float64)
    result_tid = np.zeros(len(points), dtype=np.int64)
    result_distance = np.zeros(len(points), dtype=np.float64)
    result_type = np.zeros(len(points), dtype=np.int64)

    # go through results to get minimum distance result
    for i, close_points, distance, dist_type, candidate in zip(
            np.arange(len(points)),
            np.array_split(query_close, query_group),
            np.array_split(distance_2, query_group),
            np.array_split(query_type, query_group),
            candidates):

        # unless some other check is true use the smallest distance
        idx = distance.argmin()

        # if we have multiple candidates check them
        if len(candidate) > 1:
            # (2, ) int, list of 2 closest candidate indices
            idxs = distance.argsort()[:2]
            # make sure the two distances are identical
            check_distance = distance[idxs].ptp() < tol.merge
            # make sure the magnitude of both distances are nonzero
            check_magnitude = (np.abs(distance[idxs]) > tol.merge).all()

            # check if query-points are actually off-surface
            if check_distance and check_magnitude:
                # get face normals for two points
                normals = mesh.face_normals[np.array(candidate)[idxs]]
                # compute normalized surface-point to query-point vectors
                vectors = ((points[i] - close_points[idxs]) /
                           distance[idxs, np.newaxis] ** 0.5)
                # compare enclosed angle for both face normals
                dots = util.diagonal_dot(normals, vectors)
                # take the idx with the most positive angle
                idx = idxs[dots.argmax()]

        # take the single closest value from the group of values
        result_close[i] = close_points[idx]
        result_tid[i] = candidate[idx]
        result_distance[i] = distance[idx]
        result_type[i] = dist_type[idx]

    return result_close, result_distance, result_tid, result_type # distance is actually squared


def closest_point_corresponding(triangles, points):
    """
    Return the closest point on the surface of each triangle for a
    list of corresponding points.
    Implements the method from "Real Time Collision Detection" and
    use the same variable names as "ClosestPtPointTriangle" to avoid
    being any more confusing.
    Parameters
    ----------
    triangles : (n, 3, 3) float
      Triangle vertices in space
    points : (n, 3) float
      Points in space
    Returns
    ----------
    closest : (n, 3) float
      Point on each triangle closest to each point
    """

    # check input triangles and points
    triangles = np.asanyarray(triangles, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError('triangles shape incorrect')
    if not util.is_shape(points, (len(triangles), 3)):
        raise ValueError('need same number of triangles and points!')

    # store the location of the closest point
    result = np.zeros_like(points)
    # which points still need to be handled
    remain = np.ones(len(points), dtype=np.bool)
    dist_type = np.zeros(len(points), dtype=np.int) # will do dist_type = 3 - dist_type at the end

    # if we dot product this against a (n, 3)
    # it is equivalent but faster than array.sum(axis=1)
    ones = [1.0, 1.0, 1.0]

    # get the three points of each triangle
    # use the same notation as RTCD to avoid confusion
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    # check if P is in vertex region outside A
    ab = b - a
    ac = c - a
    ap = points - a
    # this is a faster equivalent of:
    # util.diagonal_dot(ab, ap)
    d1 = np.dot(ab * ap, ones)
    d2 = np.dot(ac * ap, ones)

    # is the point at A
    is_a = np.logical_and(d1 < tol.zero, d2 < tol.zero)
    if is_a.any():
        result[is_a] = a[is_a]
        remain[is_a] = False
        dist_type[is_a] = 3

    # check if P in vertex region outside B
    bp = points - b
    d3 = np.dot(ab * bp, ones)
    d4 = np.dot(ac * bp, ones)

    # do the logic check
    is_b = (d3 > -tol.zero) & (d4 <= d3) & remain
    if is_b.any():
        result[is_b] = b[is_b]
        remain[is_b] = False
        dist_type[is_b] = 3

    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = ((vc < tol.zero) &
             (d1 > -tol.zero) &
             (d3 < tol.zero) & remain)
    if is_ab.any():
        v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).reshape((-1, 1))
        result[is_ab] = a[is_ab] + (v * ab[is_ab])
        remain[is_ab] = False
        dist_type[is_ab] = 3

    # check if P in vertex region outside C
    cp = points - c
    d5 = np.dot(ab * cp, ones)
    d6 = np.dot(ac * cp, ones)
    is_c = (d6 > -tol.zero) & (d5 <= d6) & remain
    if is_c.any():
        result[is_c] = c[is_c]
        remain[is_c] = False
        dist_type[is_c] = 2

    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (vb < tol.zero) & (d2 > -tol.zero) & (d6 < tol.zero) & remain
    if is_ac.any():
        w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).reshape((-1, 1))
        result[is_ac] = a[is_ac] + w * ac[is_ac]
        remain[is_ac] = False
        dist_type[is_ac] = 1

    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = ((va < tol.zero) &
             ((d4 - d3) > - tol.zero) &
             ((d5 - d6) > -tol.zero) & remain)
    if is_bc.any():
        d43 = d4[is_bc] - d3[is_bc]
        w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).reshape((-1, 1))
        result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
        remain[is_bc] = False
        dist_type[is_bc] = 2

    # any remaining points must be inside face region
    if remain.any():
        # point is inside face region
        denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
        v = (vb[remain] * denom).reshape((-1, 1))
        w = (vc[remain] * denom).reshape((-1, 1))
        # compute Q through its barycentric coordinates
        result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)
        
    dist_type = 3 - dist_type
    return result, dist_type