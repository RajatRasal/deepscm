# This file is partially adapted from the NVIDIA kaolin library, available at
# https://github.com/NVIDIAGameWorks/kaolin
# specifically kaolin/metrics/mesh.py.

# It was modified to use the trimesh implementation instead on CPU, as well as a few
# necessary adjustments to dispatch to the correct implementation depending on the platform
# and for ease of building (at the time of writing). 
import torch
from .cpu import triangle_distance_cpu
if torch.cuda.is_available():
    from . import cuda
    import triangle_distance_cuda


def _points2triangulation_cpu(points, mesh, verts, tris):
    _, _, idx, dist_type = triangle_distance_cpu.forward_cpu(points.numpy(), mesh, verts.numpy(), tris.numpy())  
    return torch.from_numpy(idx).long(), torch.from_numpy(dist_type)
            
def _points2triangulation_cuda(points, v1, v2, v3):
    _, idx, dist_type = _TriangleDistanceCudaFunction.apply(points, v1, v2, v3)
    return idx.long(), dist_type   
    
class _TriangleDistanceCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, verts_1, verts_2, verts_3):
        # weird manipulation moved here
        # not sure why this is done when the batch size of 1(?) is ultimately ignored.
        points = points.view(1, -1, 3)
        verts_1 = verts_1.view(1, -1, 3)
        verts_2 = verts_2.view(1, -1, 3)
        verts_3 = verts_3.view(1, -1, 3)
        
        #
        batchsize, n, _ = points.size()
        points = points.contiguous()
        verts_1 = verts_1.contiguous()
        verts_2 = verts_2.contiguous()
        verts_3 = verts_3.contiguous()

        dist = torch.zeros(batchsize, n)
        idx = torch.zeros(batchsize, n, dtype=torch.int)
        dist_type = torch.zeros(batchsize, n, dtype=torch.int)

        dist = dist.cuda()
        idx = idx.cuda()
        dist_type = dist_type.cuda()
        triangle_distance_cuda.forward_cuda(points, verts_1, verts_2, verts_3, dist, idx, dist_type)
        #ctx.save_for_backward(idx1)

        return dist[0], idx[0], dist_type[0]
        #return dist1[0].detach(), idx1[0].detach().long(), type1[0].long()  
