import sys
import mayavi.mlab as mlab
import numpy as np
import torch
#mlab.options.offscreen = True

colors = np.array(
    [
        [128, 128, 128, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        [139, 137, 137, 255],   # other_flat           dark red
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

voxel_size = 0.5
pc_range = [-40, -40,  -1, 40, 40, 5.4]
occ_size = [200, 200, 16]
cls_num=18
use_visible_mask=True
visual_path = sys.argv[1]

fov_voxels = np.load(visual_path)

occ = fov_voxels['arr_0']
# occ = fov_voxels['semantics']

occ=np.flip(occ,axis=1)
occ = occ.astype(np.float32)+1
occ = np.where(occ== cls_num, 0, occ)

if use_visible_mask:
    gt_path=sys.argv[2]
    fov_voxels = np.load(gt_path)
    visible_mask = fov_voxels['mask_camera']
    occ[visible_mask==0] = 0

def get_vertices(occ):
    x = torch.linspace(0, occ.shape[0] - 1, occ.shape[0])
    y = torch.linspace(0, occ.shape[1] - 1, occ.shape[1])
    z = torch.linspace(0, occ.shape[2] - 1, occ.shape[2])
    X, Y, Z = torch.meshgrid(x, y, z)
    vv = torch.stack([X, Y, Z], dim=-1)
    
    vertices = vv[occ > 0.5]
    vertices[:, 0] = (vertices[:, 0] + 0.5) * (pc_range[3] - pc_range[0]) / occ_size[0]  + pc_range[0]
    vertices[:, 1] = (vertices[:, 1] + 0.5) * (pc_range[4] - pc_range[1]) / occ_size[1]  + pc_range[1]
    vertices[:, 2] = (vertices[:, 2] + 0.5) * (pc_range[5] - pc_range[2]) / occ_size[2]  + pc_range[2]

    vertices = vertices.cpu().numpy()
    semantics = occ[occ> 0]
    vertices = np.concatenate([vertices, semantics[:, None]], axis=-1)

    return vertices
fov_voxels=get_vertices(occ)

fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
fov_voxels[..., 3]-=1

figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))

plt_plot_fov = mlab.points3d(
    fov_voxels[:, 1],
    fov_voxels[:, 0],
    fov_voxels[:, 2],
    fov_voxels[:, 3],
    colormap="viridis",
    scale_factor=0.95*voxel_size,
    mode="cube",
    opacity=1,
    vmin=0,
    vmax=19,
)

plt_plot_fov.glyph.scale_mode = "scale_by_vector"
plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

mlab.show()
