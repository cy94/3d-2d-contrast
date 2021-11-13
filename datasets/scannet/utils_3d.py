import math
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import imageio
import cv2

def resize_crop_image(image, new_image_dims):
    '''
    image: H, W
    new_image_dims: Wnew, Hnew
    TODO: this convention is confusing?
    '''
    # W, H
    image_dims = [image.shape[1], image.shape[0]]
    # old W,H = new W,H?
    if image_dims == new_image_dims:
        return image
    # W' = (Hnew/H)*W
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    # H,W -> H',W'
    image = transforms.Resize([new_image_dims[1], resize_width], 
                    interpolation=transforms.InterpolationMode.NEAREST)(Image.fromarray(image))
    # H',W'-> Hnew, Wnew
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    # return Hnew,Wnew image
    return image

def load_depth_multiple(paths, image_dims, out):
    '''
    paths: paths to depth files
    out: out array
    '''
    for ndx, path in enumerate(paths):
        if path.exists():
            out[ndx] = torch.Tensor(load_depth(path, image_dims))

    return out

def load_depth(path, image_dims=(640, 480)):
    '''
    path: full path to depth file
    image_dims: resize image to this size
    '''
    # read 480, 640 array
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
        if path.exists():
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
        if path.exists():
            out[ndx] = torch.Tensor(load_color(path, image_dims, transform=transform))
    return out

def load_color(path, image_dims, transform=None):
    '''
    path: str path to file
    image_dims: Wnew, Hnew
    '''
    # reads a H, W, 3 array
    rgb = imageio.imread(path)
    # resize H, W -> Hnew, Wnew

    # use cv2.resize as done during enet training
    rgb = cv2.resize(rgb, image_dims)
    
    # normalize the image, etc 
    if transform is not None:
        # transform is on a dict with 'x' key
        rgb = transform({'x': rgb})['x']
    # no need to transpose the dims, its already H, W
    # put channel first, get C, H, W
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
        # W, H dims of the image that looks at the subvol
        self.image_dims = image_dims
        # W, H, D dims of the subvol
        self.volume_dims = volume_dims
        # side length of the boxel
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
        Given x,y pixel coords and depth, map to camera coords XYZ

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

        # image_dims is W,H -> 40,30

        # put all pixel coords XY+depth = (X, Y, depths) in a single tensor, 
        # map to pixel coords
        X, Y, depth = torch.Tensor((
            # nearest frustum corners (corresponding to depth min)
            # lower left 
            (0, 0, self.depth_min),
            # lower right 
            (self.image_dims[0] - 1, 0, self.depth_min),
            # upper right 
            (self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_min),
            # upper left  
            (0, self.image_dims[1] - 1, self.depth_min),
            # farthest frustum corners (corresponding to depth max)
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
        # get camera coords
        corner_points[:, :3] = self.depth_to_skeleton(X, Y, depth).unsqueeze(2)

        # go from camera coords to world coords - use cam2world matrix
        p = torch.bmm(camera_to_world.repeat(8, 1, 1), corner_points)
        # get a *range* of grid coords for these corner points
        # p_lower: take floor of world coords, then map to grid coords
        pl = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.floor(p)))
        # p_upper: take ceil of world coords, then map to grid coords
        pu = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.ceil(p)))

        # remove the last "1" from homogenous coordinates, get grid XYZ
        # in each case (rounded up/down) find the grid coords closest to the origin
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

    @staticmethod
    def downsample_coords(coords, in_vol_dims, out_vol_dims):
        '''
        Map coords from the input volume to output volume dimensions
        eg: 32^3 -> 4^3
        this is a many-one mapping

        coords: homogeous coords (4, N)
        in_vol_dims: dims of the vol that coords refers to
        out_vol_dims: dims of the vol that output coords refer to

        return: new (4, N)
        '''
        # downsampling factor in each dimension
        factor = (torch.Tensor(in_vol_dims) // torch.Tensor(out_vol_dims)).to(coords.device)
        # make a copy
        new_coords = coords.clone()
        # divide by the factor
        new_coords[:3, :] = (new_coords[:3, :].T // factor).T

        return new_coords

    @staticmethod
    def coords_to_lin_inds(coords, num_inds, vol_dims):
        '''
        coords: (4, N homogenous coords)
        vol_dims: dims of the volume

        return: (N+1,) array
        '''
        lin_inds = coords.new_empty(1 + num_inds)
        inds = coords[2, :]*vol_dims[0]*vol_dims[1] + coords[1, :]*vol_dims[0] + coords[0, :]
        lin_inds[0] = len(inds)
        lin_inds[1:1+len(inds)] = inds

        return lin_inds

    @staticmethod
    def lin_ind2d_to_coords2d_static(lin_ind, img_dims, coords=None):
        '''
        Get XY coordinates within the image grid

        lin_ind: [0, 1, 2, 3 ..] tensor of integers - only the valid indices
        coords: empty array to fill coords, (2, len(lin_ind))
        img_dims: W, H of the depth image into which the coords are used

        Static method, does the same thing as below
        additionally need to pass in the volume dims
        '''
        if coords is None:
            coords = torch.empty(2, len(lin_ind), dtype=torch.long).to(lin_ind.device)

        # IMP: use a floored division here to keep only the integer coordinates!
        # Y = N / width = number of filled widths
        coords[1] = lin_ind.div(img_dims[0], rounding_mode='floor')
        # position within the row is X -> remove full widths to get X
        coords[0] = lin_ind - (coords[1]*img_dims[0]).long()

        return coords


    @staticmethod
    def lin_ind_to_coords_static(lin_ind, vol_dims, coords=None):
        '''
        Get XYZ coordinates within the grid
        ie. homogenous coordinate XYZ of each voxel 

        lin_ind: [0, 1, 2, 3 ..] tensor of integers - only the valid indices
        coords: empty array to fill coords, has dim (4, len(lin_ind))

        Static method, does the same thing as below
        additionally need to pass in the volume dims
        '''
        if coords is None:
            coords = torch.empty(4, len(lin_ind)).to(lin_ind.device)

        # Z = N / (X*Y)
        # IMP: use a floored division here to keep only the integer coordinates!
        coords[2] = lin_ind.div(vol_dims[0]*vol_dims[1], rounding_mode='floor')
        # similarly fill X and Y
        tmp = lin_ind - (coords[2]*vol_dims[0]*vol_dims[1]).long()
        coords[1] = tmp.div(vol_dims[0], rounding_mode='floor')
        coords[0] = torch.remainder(tmp, vol_dims[0])
        # last coord is just 1
        coords[3].fill_(1)

        return coords
        
    def lin_ind_to_coords(self, lin_ind, coords=None):
        '''
        call the static method
        '''
        return self.lin_ind_to_coords_static(lin_ind, self.volume_dims, coords)


    def compute_projection(self, depth, camera_to_world, world_to_grid, return_coverage=False):
        '''
        depth: a single depth image, H,W tensor
        cam2world: single transformation matrix, pose of the camera
        world2grid: single transformation matrix
                    world coords->grid coords (xyz -> (0-32,0-32,0-32))
        return_coverage: get only the coverage, or indices?

        NOTE: treat the subvolume as XYZ = W,H,D dimensions and return indices 
        into a W,H,D volume 
        treat the depth image as a W,H array later according to pixel coordinates
        '''
        # compute projection by voxels -> image
        # camera pose is camera->world, invert it
        # TODO: invert everything outside and pass it in
        world_to_camera = torch.inverse(camera_to_world)
        grid_to_world = torch.inverse(world_to_grid)
        
        # lowest xyz and highest xyz seen by the camera in grid coords
        # ie a bounding box of the frustum created by the camera 
        # between depth_min and depth_max
        voxel_bounds_min, voxel_bounds_max = self.compute_frustum_bounds(world_to_grid, 
                                                                camera_to_world)
        
        # min coords that are negative are pulled up to 0, should be within the grid
        voxel_bounds_min = torch.maximum(voxel_bounds_min, torch.Tensor([0, 0, 0]).to(self.device)).to(self.device)
        # max coord should be within grid dimensions, anything larger is pulled down 
        # to grid dim
        voxel_bounds_max = torch.minimum(voxel_bounds_max, torch.Tensor(self.volume_dims).to(self.device)).float().to(self.device)

        # indices from 0,1,2 .. 31*31*62 = num_voxels
        lin_ind_volume = torch.arange(0, self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2], out=torch.LongTensor()).to(self.device)
        # coord is the center of the voxel?
        # ((0,0,0), (1,0,0),..(2,0,0),...(N,N,N))
        coords = self.lin_ind_to_coords(lin_ind_volume)

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

        # project camera coords XYZ onto image -> pixel XY coords
        # x = (focal length * X / Z) + x-offset
        p[0] = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        p[1] = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        # convert XY coords to integers = pixel coordinates
        pi = torch.round(p).long()

        # check which image coords lie within image bounds -> valid
        # image dims: (Width (x), Height (y))
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
        # linear index into the image = Y*img_width + X
        valid_image_ind_lin = valid_image_ind_y * self.image_dims[0] + valid_image_ind_x

        # flatten the depth image, select the depth values corresponding 
        # to the valid pixels
        # **IMP**: the depth arg is H, W but pixel coordinates are according to 
        # a W, H array -> tranpose depth before using it
        depth_tmp = depth.reshape(depth.shape[::-1])
        depth_vals = torch.index_select(depth_tmp.view(-1), 0, valid_image_ind_lin)
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
        # TODO: do this more efficiently, store a variable number of indices
        # measure how much GPU usage this saves 
        # (since backprojection is done 1 at a time later, no need to collate)
        lin_indices_3d = lin_ind_update.new_empty(self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2] + 1) 
        lin_indices_2d = lin_ind_update.new_empty(self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2] + 1) 

        # 3d indices: indices of the valid voxels computed earlier
        num_indices = lin_ind_update.shape[0]
        lin_indices_3d[0] = num_indices
        lin_indices_3d[1:1+num_indices] = lin_ind_update
        
        # 2d indices: have the same shape
        lin_indices_2d[0] = num_indices
        # values: the corresponding linear indices into the flattened image
        # where the depth mask was valid
        lin_indices_2d[1:1+num_indices] = \
            torch.index_select(valid_image_ind_lin, 0, torch.nonzero(depth_mask)[:,0])

        return lin_indices_3d, lin_indices_2d

def project_2d_3d(feat2d, lin_indices_3d, lin_indices_2d, volume_dims):
    '''
    Project 2d features to 3d features
    feat2d: C, H, W output of 2D CNN 
    lin_indices_3d: size = product(volume_dims). 1st elem is the number of actual inds
    lin_indices_2d: size = product(volume_dims). 1st elem is the number of actual inds
    volume_dims: (W, H, D) of a subvol in voxels

    return: C,D,H,W volume
    '''
    # dimension of the feature
    feat_dim = feat2d.shape[0]
    # required shape is C,D,H,W, create an empty volume
    output = feat2d.new_zeros(feat_dim, volume_dims[2], volume_dims[1], volume_dims[0])
    # number of valid voxels which can be mapped to pixels
    num_ind = lin_indices_3d[0]
    # if there are any voxels to be mapped
    if num_ind > 0:
        # linear indices into H,W image
        inds2d = lin_indices_2d[1:1+num_ind]
        # then pick the required 2d features from CHW using linear inds
        feats = feat2d.view(feat_dim, -1)[:, inds2d]
        # index into CDHW volume using linear inds
        # insert the 2d features at the required locations
        inds3d = lin_indices_3d[1:1+num_ind]
        # indices into WHD tensor
        output.view(feat_dim, -1)[:, inds3d] = feats

    return output

