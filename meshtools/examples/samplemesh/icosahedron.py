import numpy as np

def icosahedron():
    golden_ratio = (1+np.sqrt(5))/2
    n = 12
    m = 20
    
    verts = np.array([[0, 1, golden_ratio],    #0
                      [0, -1, golden_ratio],   #1
                      [0, 1, -golden_ratio],   #2
                      [0, -1, -golden_ratio],  #3
                      [1, golden_ratio, 0],    #4
                      [-1, golden_ratio, 0],   #5
                      [1, -golden_ratio, 0],   #6
                      [-1, -golden_ratio, 0],  #7
                      [golden_ratio, 0, 1],    #8
                      [golden_ratio, 0, -1],   #9
                      [-golden_ratio, 0, 1],   #10
                      [-golden_ratio, 0, -1]]) #11
    
    tris = np.array([[0, 1, 10],
                     [1, 0, 8],
                     [1, 10, 7],
                     [1, 6, 7],
                     [1, 6, 8],
                     [6, 8, 9],
                     [8, 9, 4],
                     [9, 4, 2],
                     [4, 2, 5],
                     [2, 5, 11],
                     [5, 11, 10],
                     [11, 10, 7],
                     [0, 8, 4],
                     [6, 7, 3],
                     [7, 3, 11],
                     [3, 11, 2],
                     [9, 2, 3],
                     [0, 4, 5],
                     [0, 5, 10],
                     [3, 6, 9]])
    
    return verts, tris