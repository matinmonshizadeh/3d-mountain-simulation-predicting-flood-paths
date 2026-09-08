import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

# Step 1: Generate random points in 2D space
num_points = 400
points = np.random.rand(num_points, 2)
# print(points)

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
        # Get the points of the simplex (triangle)
        pts = tri.points[simplex]
        z = z_values[simplex]
        
        # Calculate the gradient (slope)
        A = np.c_[pts[:, 0], pts[:, 1], np.ones(3)]
        coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        gradients[i] = -coef[:2]  # Gradient (negative because it points downhill)
    
    return gradients

gradients = calculate_gradient(tri, z_values)

# Step 5: Simulate the flood path
def simulate_flood_path(start_point, tri, gradients, z_values):
    path = [start_point]
    current_point = start_point
    
    for _ in range(100):  # Limit to 100 steps to avoid infinite loops
        simplex_index = tri.find_simplex(current_point)
        if simplex_index == -1:
            break  # Outside the triangulation
        
        gradient = gradients[simplex_index]
        if np.linalg.norm(gradient) < 1e-6:
            break  # Flat area, stop
        
        next_point = current_point + 0.01 * gradient
        path.append(next_point)
        
        current_point = next_point
    
    return np.array(path)

# Choose a starting point (rainfall point)
start_point = np.array([0.5, 0.5])

# Simulate the flood path
flood_path = simulate_flood_path(start_point, tri, gradients, z_values)

# Step 6: Plot the triangulated surface and flood path
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the terrain
ax.plot_trisurf(points[:, 0], points[:, 1], z_values, triangles=tri.simplices, cmap='terrain', edgecolor='none', alpha=0.8)

# Plot the flood path
path_z_values = mountain_function(flood_path[:, 0], flood_path[:, 1])
ax.plot(flood_path[:, 0], flood_path[:, 1], path_z_values, color='blue', marker='o', linewidth=2, markersize=4, label='Flood Path')

# Customize the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Elevation (Z)')
ax.set_title('3D Mountain Simulation with Predicted Flood Path')
ax.legend()

plt.show()
