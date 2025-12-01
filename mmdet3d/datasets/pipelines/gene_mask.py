import numpy as np


voxels = np.zeros((200, 200, 16), dtype=int)
max_length=np.sqrt(200**2+200**2+16**2)
def mark_voxels(ray_starts, ray_middles, lengths, step_size=0.1):

    directions = ray_middles - ray_starts
    norms = np.linalg.norm(directions, axis=1)
    directions = directions / norms[:, None] 

    num_steps = (max_length / step_size).astype(int)
    step_vectors = directions * step_size


    coordinates = ray_starts[:, None, :] + np.arange(num_steps + 1)[None,:,  None] * step_vectors[:, None, :]
    rounded_coordinates = np.round(coordinates).astype(int)


    valid_coordinates_mask = (rounded_coordinates[:, :, 0] >= -100) & \
                             (rounded_coordinates[:, :, 0] < 100) & \
                             (rounded_coordinates[:, :, 1] >= -100) & \
                             (rounded_coordinates[:, :, 1] < 100) & \
                             (rounded_coordinates[:, :, 2] >= -8) & \
                             (rounded_coordinates[:, :, 2] < 8)
    valid_coordinates = rounded_coordinates[valid_coordinates_mask]

    voxels[valid_coordinates[:, 0] + 100, valid_coordinates[:, 1] + 100, valid_coordinates[:, 2] + 8] = 1

ray_starts = np.array([
    [0, 0, 0],
    [10, 10, 5]
])
ray_middles = np.array([
    [20, 20, 10],
    [50, 50, 8]
])
lengths = np.array([30, 20])

mark_voxels(ray_starts, ray_middles, lengths)

print(voxels)
