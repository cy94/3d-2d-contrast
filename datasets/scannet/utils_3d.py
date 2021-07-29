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

def load_depth_multiple(paths, image_dims, out):
    '''
    paths: paths to depth files
    out: out array
    '''
    for ndx, path in enumerate(paths):
        out[ndx] = torch.Tensor(load_depth(path, image_dims))

    return out

def load_depth(path, image_dims=(640, 480)):
    '''
    path: full path to depth file
    image_dims: resize image to this size
    '''
    depth_image = imageio.imread(path)
    depth_image = resize_crop_image(depth_image, image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0
    return depth_image
    

def load_pose_multiple(paths, out):
    '''
    paths: paths to pose files
    out: out array
    '''
    for ndx, path in enumerate(paths):
        out[ndx] = torch.Tensor(load_pose(path))
    return out
    
def load_pose(path):
    '''
    path: full path to a pose file
    '''
    return torch.from_numpy(np.genfromtxt(path).astype(np.float32))

def load_rgbs_multiple(paths, image_dims, out, transform=None):
    '''
    paths: paths to color files
    out: out array
    '''
    for ndx, path in enumerate(paths):
        out[ndx] = torch.Tensor(load_color(path, image_dims, transform=transform))
    return out

def load_color(path, image_dims, transform=None):
    rgb = imageio.imread(path)
    rgb = resize_crop_image(rgb, image_dims)
    if transform is not None:
        rgb = transform(rgb)
    rgb =  np.transpose(rgb, [2, 0, 1]) 
    return rgb

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

        self.device = torch.device('cpu')

        # create coords only once, clone and use next 
        # indices from 0,1,2 .. 31*31*62 = num_voxels
        self._lin_ind_volume = torch.arange(0, self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2], out=torch.LongTensor()).to(self.device)
        # empty array with size (4, num_voxels)
        tmp = torch.empty(4, self._lin_ind_volume.size(0))
        self._coords = self.lin_ind_to_coords(self._lin_ind_volume, tmp)

    def get_lin_ind_volume(self):
        return self._lin_ind_volume.clone().to(self.device)

    def get_subvol_coords(self):
        return self._coords.clone().to(self.device)

    def to(self, device):
        self.device = device
        return self

    def update_intrinsic(self, new_intrinsic):
        self.intrinsic = new_intrinsic.to(self.device)

    def depth_to_skeleton(self, ux, uy, depth):
        '''
        ux, uy: image coordinates 
        depth: depth to which these image coordinates must be projected 
        '''
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.vstack((depth*x, depth*y, depth)).T

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

        # put all X, Y, depths in single tensors, compute everything together
        X, Y, depth = torch.Tensor((
            # nearest frustum corners (depth min)
            # lower left 
            (0, 0, self.depth_min),
            # lower right 
            (self.image_dims[0] - 1, 0, self.depth_min),
            # upper right 
            (self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_min),
            # upper left  
            (0, self.image_dims[1] - 1, self.depth_min),
            # lower left corner
            (0, 0, self.depth_max),
            # lower right corner
            (self.image_dims[0] - 1, 0, self.depth_max),
            # upper right corner
            (self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_max),
            # upper left corner
            (0, self.image_dims[1] - 1, self.depth_max),
        )).to(self.device).T

        # compute all 8 points together
        # unsqueeze (8, 3) -> (8, 3, 1)
        corner_points[:, :3] = self.depth_to_skeleton(X, Y, depth).unsqueeze(2)

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
        bbox_min = torch.minimum(bbox_min0, bbox_min1)
        # repeat for maximum
        bbox_max0, _ = torch.max(pl[:, :3, 0], 0)
        bbox_max1, _ = torch.max(pu[:, :3, 0], 0) 
        bbox_max = torch.maximum(bbox_max0, bbox_max1)
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
        coords[2] = lin_ind.div(self.volume_dims[0]*self.volume_dims[1], rounding_mode='floor')
        # similarly fill X and Y
        tmp = lin_ind - (coords[2]*self.volume_dims[0]*self.volume_dims[1]).long()
        coords[1] = tmp.div(self.volume_dims[0], rounding_mode='floor')
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
        voxel_bounds_min = torch.maximum(voxel_bounds_min, torch.Tensor([0, 0, 0]).to(self.device)).to(self.device)
        # max coord should be within grid dimensions, any greater is pulled down 
        # to grid dim
        voxel_bounds_max = torch.minimum(voxel_bounds_max, torch.Tensor(self.volume_dims).to(self.device)).float().to(self.device)

        # indices from 0,1,2 .. 31*31*62 = num_voxels
        lin_ind_volume = torch.arange(0, self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2], out=torch.LongTensor()).to(self.device)
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
        # and the corresponding coordinates
        coords = coords[:, mask_frustum_bounds]

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
        # where the depth mask was valid
        lin_indices_2d[1:1+lin_indices_2d[0]] = \
            torch.index_select(valid_image_ind_lin, 0, torch.nonzero(depth_mask)[:,0])

        return lin_indices_3d, lin_indices_2d

def project_2d_3d(feat2d, lin_indices_3d, lin_indices_2d, volume_dims):
    '''
    Project 2d features to 3d features
    '''
    # is the 2D feature (W, H)? then C=1, else (C, W, H) -> get C
    num_feat = 1 if len(feat2d.shape) == 2 else feat2d.shape[0]
    # required shape is C, D, H, W, create an empty volume
    output = feat2d.new_zeros(num_feat, volume_dims[2], volume_dims[1], volume_dims[0])
    # number of valid voxels which can be mapped to pixels
    num_ind = lin_indices_3d[0]
    # if there are any voxels to be mapped
    if num_ind > 0:
        # reshape the 2d feature to have 2 dimensions (C, W*H)
        # then pick the required 2d features
        # get the features for the required pixels
        vals = torch.index_select(feat2d.view(num_feat, -1), 1, lin_indices_2d[1:1+num_ind])
        # reshape the output volume to (C, W*H*D), then insert the 2d features
        # at the requires locations
        output.view(num_feat, -1)[:, lin_indices_3d[1:1+num_ind]] = vals
    return output

