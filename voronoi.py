import numpy as np
import cv2
from scipy.spatial import Voronoi

# Parameters
np.random.seed()
image_size = 1024
num_points = 300  # Increase for denser map
points = np.random.rand(num_points, 2) * image_size
vor = Voronoi(points)

# Create white canvas
canvas = 255 * np.ones((image_size, image_size, 3), dtype=np.uint8)

# Draw only bounded Voronoi cells
for region_index in vor.point_region:
    region = vor.regions[region_index]
    if not -1 in region and len(region) > 0:
        try:
            polygon = np.array([vor.vertices[i] for i in region], dtype=np.int32)
            if polygon.shape[0] >= 3:
                # Clip vertices to image bounds
                polygon = np.clip(polygon, 0, image_size - 1)
                cv2.polylines(canvas, [polygon], isClosed=True, color=(0, 0, 0), thickness=4)
        except:
            continue

# Save image
cv2.imwrite("Terrain_generation/output/voronoi.png", canvas)
np.save("Terrain_generation/output/voronoi_map.npy", canvas)

