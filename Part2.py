import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed for the 2D point sampling')
args = parser.parse_args()
np.random.seed(args.seed)

# Step 1: Generate random points in 2D space
num_points = 400
points = np.random.rand(num_points, 2)

# Step 2: Generate elevation (z-values) with a mountain-like distribution
# NUM1: Gaussian Mountain
# def mountain_function(x, y):
#     return np.exp(-5 * ((x - 0.5)**2 + (y - 0.5)**2))

# NUM2: Simple Hill (Circular Cone)
# def mountain_function(x, y):
#     return 1 - np.sqrt((x - 0.5)**2 + (y - 0.5)**2)

# NUM3: Multiple Peaks:
# def mountain_function(x, y):
#     return (np.exp(-5 * ((x - 0.25)**2 + (y - 0.25)**2)) +
#             np.exp(-5 * ((x - 0.75)**2 + (y - 0.75)**2)))

# NUM4: Ridges (Sine Waves):
# def mountain_function(x, y):
#     return np.sin(5 * np.pi * x) * np.sin(5 * np.pi * y)

# NUM5: Volcanic Crater:
# def mountain_function(x, y):
#     r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
#     return np.exp(-5 * r) - np.exp(-20 * r)

# NUM6: Plateau with Cliffs:
# def mountain_function(x, y):
#     return np.tanh(10 * (0.5 - np.sqrt((x - 0.5)**2 + (y - 0.5)**2)))

# NUM7: mountain with a valley
def mountain_function(x, y):
    return np.exp(-5 * ((x - 0.5)**2 + (y - 0.5)**2)) - 0.5 * np.exp(-5 * ((x - 0.25)**2 + (y - 0.75)**2))

z_values = mountain_function(points[:, 0], points[:, 1])

# Combine the 2D points and their corresponding z-values
points_3d = np.column_stack((points, z_values))

# Step 3: Perform Delaunay triangulation
tri = Delaunay(points[:, :2])

# Step 4: Define a function to calculate the gradient at each point
def calculate_gradient(tri, z_values):
    gradients = np.zeros((len(tri.simplices), 2))
    for i, simplex in enumerate(tri.simplices):
        pts = tri.points[simplex]
        z = z_values[simplex]
        
        A = np.c_[pts[:, 0], pts[:, 1], np.ones(3)]
        coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        gradients[i] = -coef[:2]
    
    return gradients

gradients = calculate_gradient(tri, z_values)

# Step 5: Simulate the flood path with boundary checks
def simulate_flood_path(start_point, tri, gradients, z_values, boundary):
    path = [start_point]
    current_point = start_point
    
    for _ in range(100):
        simplex_index = tri.find_simplex(current_point)
        if simplex_index == -1:
            break  # Outside the triangulation
        
        gradient = gradients[simplex_index]
        if np.linalg.norm(gradient) < 1e-6:
            break  # Flat area, stop
        
        next_point = current_point + 0.01 * gradient
        
        # Check if the next point is within the boundary
        if (next_point[0] < boundary[0] or next_point[0] > boundary[1] or
            next_point[1] < boundary[2] or next_point[1] > boundary[3]):
            break
        
        path.append(next_point)
        current_point = next_point
    
    return np.array(path)

# Define the boundary of the simulation area
boundary = [0, 1, 0, 1]  # [x_min, x_max, y_min, y_max]

# Step 6: Create a grid overlay
grid_size = 0.1
x_grid = np.arange(0, 1 + grid_size, grid_size)
y_grid = np.arange(0, 1 + grid_size, grid_size)
x_grid, y_grid = np.meshgrid(x_grid, y_grid)
z_grid = mountain_function(x_grid, y_grid)

# Initialize water volume grid
water_volume = np.zeros_like(x_grid)

# Define rain volume
rain_volume_per_cell = 10  # 10 liters of rain per cell

# Step 7: Simulate flood paths from each grid cell center (stored for plotting)
flood_paths = []
for i in range(x_grid.shape[0] - 1):
    for j in range(y_grid.shape[1] - 1):
        # Center of the grid cell
        center_x = (x_grid[i, j] + x_grid[i+1, j] + x_grid[i, j+1] + x_grid[i+1, j+1]) / 4
        center_y = (y_grid[i, j] + y_grid[i+1, j] + y_grid[i, j+1] + y_grid[i+1, j+1]) / 4
        start_point = np.array([center_x, center_y])
        
        # Simulate the flood path for this starting point with boundary check
        flood_path = simulate_flood_path(start_point, tri, gradients, z_values, boundary)
        flood_paths.append(flood_path)
        
        if flood_path.size > 0:
            stopping_point = flood_path[-1]
            
            # Determine the cell in which the stopping point lies
            cell_x = int((stopping_point[0] - boundary[0]) / grid_size)
            cell_y = int((stopping_point[1] - boundary[2]) / grid_size)
            
            if 0 <= cell_x < x_grid.shape[0] - 1 and 0 <= cell_y < y_grid.shape[1] - 1:
                water_volume[cell_x, cell_y] += rain_volume_per_cell

# Step 8: Plot the terrain, flood paths, and water volume heatmap
fig = plt.figure(figsize=(14, 10))

# Plot the 3D terrain and flood paths
ax = fig.add_subplot(121, projection='3d')

# Plot the terrain
ax.plot_trisurf(points[:, 0], points[:, 1], z_values, triangles=tri.simplices, cmap='terrain', edgecolor='none', alpha=0.8)

# Plot the grid overlay
for i in range(x_grid.shape[0]):
    ax.plot(x_grid[i, :], y_grid[i, :], z_grid[i, :], color='black', linestyle='--', linewidth=0.5)
for j in range(y_grid.shape[1]):
    ax.plot(x_grid[:, j], y_grid[:, j], z_grid[:, j], color='black', linestyle='--', linewidth=0.5)

# Plot the flood paths (already traced above)
for flood_path in flood_paths:
    path_z_values = mountain_function(flood_path[:, 0], flood_path[:, 1])
    ax.plot(flood_path[:, 0], flood_path[:, 1], path_z_values, color='blue', linewidth=1, alpha=0.7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Elevation (Z)')
ax.set_title('3D Mountain Simulation with Flood Paths')

# Plot the water volume heatmap
ax2 = fig.add_subplot(122)
cax = ax2.imshow(water_volume.T, cmap='Blues', interpolation='nearest', origin='lower')
ax2.set_xlabel('Grid X')
ax2.set_ylabel('Grid Y')
ax2.set_title('Accumulated Water Volume (Liters)')
fig.colorbar(cax, ax=ax2, label='Water Volume (Liters)')

plt.show()
