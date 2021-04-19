import numpy as np

def triangle_strip():
    verts = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0],[2,1,0]])
    tris = np.array([[0,1,2],[1,2,3],[2,3,4]])
    return verts, tris