import math
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import imageio

def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], 
                    interpolation=transforms.InterpolationMode.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)

    return image

def load_depth(path, image_dims=(640, 480)):
    '''
    path: full path to depth file
    image_dims: resize image to this size
    '''
    depth_image = imageio.imread(path)
    depth_image = resize_crop_image(depth_image, image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0
    return depth_image
    
def load_pose(path):
    '''
    path: full path to a pose file
    '''
    return torch.from_numpy(np.genfromtxt(path).astype(np.float32))

def load_intrinsic(path):
    '''
    path: full path to intrinsic file
    '''
    return torch.from_numpy(np.genfromtxt(path).astype(np.float32))

# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    '''
    create intrinsic matrix from focal length and camera centers
    '''
    intrinsic = torch.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''
    intrinsic: existing intrinsic matrix, corresponds to 640x480 image
    intrinsic_image_dim: default 640x480, dims of image in the existing instrinsic matrix
    image_dim: dims of the feature map, ~40x30
    '''
    # no need to change anything
    if intrinsic_image_dim == image_dim:
        return intrinsic
    # keep the "30" dim fixed, find the corresponding width ~40
    # ~ 30 * 640/480 = 40    
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    # multiply focal length x by a factor of (40/640) ~ 0.0625
    intrinsic[0,0] *= float(resize_width)/float(intrinsic_image_dim[0])
    # multiply focal length y by a factor of (30/480) ~ 0.0625 
    intrinsic[1,1] *= float(image_dim[1])/float(intrinsic_image_dim[1])
    # multiply the center of the image by the same factor 
    # account for cropping here -> subtract 1
    intrinsic[0,2] *= float(image_dim[0]-1)/float(intrinsic_image_dim[0]-1)
    intrinsic[1,2] *= float(image_dim[1]-1)/float(intrinsic_image_dim[1]-1)
    return intrinsic

class ProjectionHelper():
    def __init__(self, intrinsic, depth_min, depth_max, image_dims, volume_dims, voxel_size):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims
        self.volume_dims = volume_dims
        self.voxel_size = voxel_size

    def update_intrinsic(self, new_intrinsic):
        self.intrinsic = new_intrinsic

    def depth_to_skeleton(self, ux, uy, depth):
        '''
        ux, uy: image coordinates
        depth: depth to which these image coordinates must be projected 
        '''
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.Tensor([depth*x, depth*y, depth])


    def skeleton_to_depth(self, p):
        '''
        p: point in 3D
        '''
        x = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        y = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        return torch.Tensor([x, y, p[2]])

    def compute_frustum_bounds(self, world_to_grid, camera_to_world):
        '''
        Given the location of the camera and the location of the grid
        find the bounds of the grid that the camera can see
        '''
        # create an empty array with the same device and datatype as cam2world
        # with dims: 
            # 8: 8 points
            # 4: homogenous coordinates (1 at the end)
            # 1: value
        corner_points = camera_to_world.new_empty(8, 4, 1).fill_(1)
	    # nearest frustum corners (depth min)
        # lower left intrinsic will be used only 
        corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min).unsqueeze(1)
        # lower right intrinsic will be used only 
        corner_points[1][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min).unsqueeze(1)
        # upper right intrinsic will be used only 
        corner_points[2][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        # upper left intrinsic will be used only 
        corner_points[3][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        
        # far frustum corners (depth max)
        # lower left corner
        corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max).unsqueeze(1)
        # lower right corner
        corner_points[5][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max).unsqueeze(1)
        # upper right corner
        corner_points[6][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)
        # upper left corner
        corner_points[7][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)

        # go from camera coords to world coords - use cam2world matrix
        p = torch.bmm(camera_to_world.repeat(8, 1, 1), corner_points)
        # get a *range* of grid coords for these corner points
        # p_lower: take floor of world coords, then map to grid coords
        pl = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.floor(p)))
        # p_upper: take ceil of world coords, then map to grid coords
        pu = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.ceil(p)))

        # remove the last 1 from homogenous coordinates
        # in each case (rounded up/down) find the *coordinates* closest to the origin
        bbox_min0, _ = torch.min(pl[:, :3, 0], 0)
        bbox_min1, _ = torch.min(pu[:, :3, 0], 0)
        # then take the minimum of those 2 cases
        # -> get a single (x, y, z) 
        bbox_min = np.minimum(bbox_min0, bbox_min1)
        # repeat for maximum
        bbox_max0, _ = torch.max(pl[:, :3, 0], 0)
        bbox_max1, _ = torch.max(pu[:, :3, 0], 0) 
        bbox_max = np.maximum(bbox_max0, bbox_max1)
        return bbox_min, bbox_max

    def get_coverage(self, depth, camera_to_world, world_to_grid):
        coverage = self.compute_projection(depth, camera_to_world, world_to_grid, 
                                return_coverage=True)
        if coverage is None:
            return 0
        else:
            return coverage

    def lin_ind_to_coords(self, lin_ind, coords):
        '''
        Get XYZ coordinates within the grid
        ie. homogenous coordinate XYZ of each voxel 

        lin_ind: [0, 1, 2, 3 ..] tensor of integers
        coords: empty array to fill coords, has dim (4, len(lin_ind))
        '''
        # Z = N / (X*Y)
        # IMP: use a floored division here to keep only the integer coordinates!
        coords[2] = lin_ind // (self.volume_dims[0]*self.volume_dims[1])
        # similarly fill X and Y
        tmp = lin_ind - (coords[2]*self.volume_dims[0]*self.volume_dims[1]).long()
        coords[1] = tmp // self.volume_dims[0]
        coords[0] = torch.remainder(tmp, self.volume_dims[0])
        # last coord is just 1
        coords[3].fill_(1)

        return coords

    def compute_projection(self, depth, camera_to_world, world_to_grid, return_coverage=False):
        '''
        depth: a single depth image
        cam2world: single transformation matrix
        world2grid: single transformation matrix
        return_coverage: get only the coverage, or indices?
    
        TODO: make runnable on CPU as well
        '''
        # compute projection by voxels -> image
        # camera pose is camera->world, invert it
        world_to_camera = torch.inverse(camera_to_world)
        grid_to_world = torch.inverse(world_to_grid)
        
        # lowest xyz and highest xyz seen by the camera in grid coords
        # ie a bounding box of the frustum created by the camera 
        # between depth_min and depth_max
        voxel_bounds_min, voxel_bounds_max = self.compute_frustum_bounds(world_to_grid, 
                                                                camera_to_world)
        
        # min coords that are negative are pulled up to 0, should be within the grid
        voxel_bounds_min = np.maximum(voxel_bounds_min, 0)# .cuda()
        # max coord should be within grid dimensions, any greater is pulled down 
        # to grid dim
        voxel_bounds_max = np.minimum(voxel_bounds_max, self.volume_dims).float() #.cuda()

        # indices from 0,1,2 .. 31*31*62 = num_voxels
        lin_ind_volume = torch.arange(0, self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2], out=torch.LongTensor()) #.cuda()
        # empty array with size (4, num_voxels)
        coords = camera_to_world.new_empty(4, lin_ind_volume.size(0))
        coords = self.lin_ind_to_coords(lin_ind_volume, coords)

        # the actual voxels that the camera can see
        # based on the lower bound
        # X/Y/Z coord of the voxel > min X/Y/Z coord
        mask_frustum_bounds = torch.ge(coords[0], voxel_bounds_min[0]) \
                            * torch.ge(coords[1], voxel_bounds_min[1]) \
                            * torch.ge(coords[2], voxel_bounds_min[2])
        # based on the upper bound
        # X/Y/Z coord of the voxel < max X/Y/Z coord
        mask_frustum_bounds = mask_frustum_bounds \
                            * torch.lt(coords[0], voxel_bounds_max[0]) \
                            * torch.lt(coords[1], voxel_bounds_max[1]) \
                            * torch.lt(coords[2], voxel_bounds_max[2])
        # no voxels within the frustum bounds of the camera
        if not mask_frustum_bounds.any():
            return None
        
        # pick only these voxels within the frustum bounds
        lin_ind_volume = lin_ind_volume[mask_frustum_bounds]
        # create new coordinates within the visible voxels (same as before)
        # TODO: why?
        coords = coords.resize_(4, lin_ind_volume.size(0))
        coords = self.lin_ind_to_coords(lin_ind_volume, coords)

        # grid coords -> world coords -> camera coords XYZ
        p = torch.mm(world_to_camera, torch.mm(grid_to_world, coords))

        # project XYZ onto image -> XY coords
        p[0] = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        p[1] = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        # convert XY coords to integers = pixel coordinates
        pi = torch.round(p).long()

        # check which image coords lie within image bounds -> valid
        valid_ind_mask = torch.ge(pi[0], 0) \
                        * torch.ge(pi[1], 0) \
                        * torch.lt(pi[0], self.image_dims[0]) \
                        * torch.lt(pi[1], self.image_dims[1])
        if not valid_ind_mask.any():
            return None

        # valid X coords of image
        valid_image_ind_x = pi[0][valid_ind_mask]
        # valid Y coords of image
        valid_image_ind_y = pi[1][valid_ind_mask]
        # linear index into the image = Y + img_width*X
        valid_image_ind_lin = valid_image_ind_y * self.image_dims[0] + valid_image_ind_x

        # flatten the depth image, select the depth values corresponding 
        # to the valid pixels
        depth_vals = torch.index_select(depth.view(-1), 0, valid_image_ind_lin)
        # filter depth pixels based on 3 conditions
        # 1. depth > min_depth 
        # 2. depth < max_depth
        # 3. depth is within voxel_size of voxel Z coordinate
        depth_mask = depth_vals.ge(self.depth_min) \
                    * depth_vals.le(self.depth_max) \
                    * torch.abs(depth_vals - p[2][valid_ind_mask]).le(self.voxel_size)
        # no valid depths
        if not depth_mask.any():
            return None

        # pick the 3D indices which have valid 2D and valid depth
        lin_ind_update = lin_ind_volume[valid_ind_mask]
        lin_ind_update = lin_ind_update[depth_mask]

        # just need the coverage, not the projection
        if return_coverage:
            return len(lin_ind_update)

        # create new tensors to store the indices
        # each volume in the batch has a different number of valid voxels/pixels 
        # but tensor shape needs to be same size for all in batch
        # hence create tensor with the max size
        # store the actual number of indices in the first element
        # rest of the elements are the actual indices!

        lin_indices_3d = lin_ind_update.new_empty(self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2] + 1) 
        lin_indices_2d = lin_ind_update.new_empty(self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2] + 1) 

        # 3d indices: indices of the valid voxels computed earlier
        lin_indices_3d[0] = lin_ind_update.shape[0]
        lin_indices_3d[1:1+lin_indices_3d[0]] = lin_ind_update
        
        # 2d indices: have the same shape
        lin_indices_2d[0] = lin_ind_update.shape[0]
        # values: the corresponding linear indices into the flattened image
        # where depth values are nonzero
        lin_indices_2d[1:1+lin_indices_2d[0]] = \
            torch.index_select(valid_image_ind_lin, 0, torch.nonzero(depth_mask)[:,0])

        return lin_indices_3d, lin_indices_2d
