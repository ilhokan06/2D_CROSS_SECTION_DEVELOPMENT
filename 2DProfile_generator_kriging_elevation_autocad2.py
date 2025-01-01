import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykrige.uk import UniversalKriging
import ezdxf
from shapely.geometry import LineString, Polygon

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# Example DataFrame (Distance along section, Altitude, Depth, Vs)
data = pd.read_excel('E:\\TOSHIBA-BACK-UP\\MARAS_EQ\\2D_CROSS_SECTIONs\\Input_P3.xlsx')

# Extracting distance, altitude, depth, and Vs values from the DataFrame
distance = data['Distance'].values
altitude = data['Altitude'].values
depth = data['Depth'].values
vs_values = data['Vs'].values

# Adjust depths relative to topography
corrected_depth = altitude - depth

# Define the grid for distance and corrected depth
distance_grid = np.linspace(data['Distance'].min(), data['Distance'].max(), 1000)
# corrected_depth_grid = np.linspace(corrected_depth.min(), corrected_depth.max(), 1000)

#Max altitude
corrected_depth_grid = np.linspace(40, 163, 200)
distance_mesh, corrected_depth_mesh = np.meshgrid(distance_grid, corrected_depth_grid)

# Universal Kriging with linear drift terms
UK = UniversalKriging(
    distance, corrected_depth, vs_values,
    variogram_model='exponential',  # Try 'spherical', 'exponential', 'linear', etc.
    drift_terms=['regional_linear']  # Account for linear trends
)

# Perform Kriging interpolation on the grid
vs_kriged, ss = UK.execute('grid', distance_grid, corrected_depth_grid)

# Plot the interpolated Vs values
plt.figure(figsize=(10, 6))
contour = plt.contour(distance_mesh, corrected_depth_mesh, vs_kriged, levels=30, cmap='viridis')
plt.colorbar(contour, label='Vs (m/s)')
plt.scatter(distance, corrected_depth, c=vs_values, edgecolors='k', label='Known Data Points')
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')
plt.show()

# Define boundaries for clipping (aligned with corrected_depth)
x_min, x_max = distance.min(), distance.max()
y_min, y_max = corrected_depth.min(), corrected_depth.max()
boundary_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

# Export contour lines to DXF
dxf_filename = 'E:\\TOSHIBA-BACK-UP\\MARAS_EQ\\2D_CROSS_SECTIONs\\Vs_Profile333.dxf'
doc = ezdxf.new()
msp = doc.modelspace()

# Loop through each contour line
for i, collection in enumerate(contour.collections):
    layer_name = f"Contour_Level_{i}"
    doc.layers.new(name=layer_name)
    for path in collection.get_paths():
        points = path.vertices
        if points is not None:
            # Create a polyline for each path
            msp.add_lwpolyline(points, close=False, dxfattribs={'layer': layer_name})

# Save DXF file
doc.saveas(dxf_filename)
print(f"DXF file saved to {dxf_filename}")
