import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykrige.uk import UniversalKriging
import ezdxf
from shapely.geometry import LineString, Polygon
import math

def parse_input_data(data):
    
    # Extracting distance, altitude, depth, and Vs values from the DataFrame
    distance = data['Distance'].values
    altitude = data['Altitude'].values
    depth = data['Depth'].values
    vs_values = data['Vs'].values

    # Adjust altitude relative to topography
    corrected_altitude = altitude - depth
    
    return distance, altitude, depth, vs_values, corrected_altitude

def get_dist_and_alt_boundaries(dist_cs_1, dist_cs_2, dist_cs_3, corr_alt_cs_1, corr_alt_cs_2, corr_alt_cs_3):
    
    # Get global min and max altitudes for all cross-sections
    
    min_cr_alt_cs_1 = min(corr_alt_cs_1)
    max_cr_alt_cs_1 = max(corr_alt_cs_1)
    
    min_cr_alt_cs_2 = min(corr_alt_cs_2)
    max_cr_alt_cs_2 = max(corr_alt_cs_2)
    
    min_cr_alt_cs_3 = min(corr_alt_cs_3)
    max_cr_alt_cs_3 = max(corr_alt_cs_3)
    
    # Get distance boundaries
    
    min_dist_cs_1 = min(dist_cs_1)
    max_dist_cs_1 = max(dist_cs_1)
    
    min_dist_cs_2 = min(dist_cs_2)
    max_dist_cs_2 = max(dist_cs_2)
    
    min_dist_cs_3 = min(dist_cs_3)
    max_dist_cs_3 = max(dist_cs_3)
    
    global_min = min(min_cr_alt_cs_1, min_cr_alt_cs_2, min_cr_alt_cs_3)
    global_max = max(max_cr_alt_cs_1, max_cr_alt_cs_2, max_cr_alt_cs_3)
    
    min_max_values = {"Min_dist_cs_1": min_dist_cs_1,
                      "Max_dist_cs_1": max_dist_cs_1,
                      "Min_dist_cs_2": min_dist_cs_2,
                      "Max_dist_cs_2": max_dist_cs_2,
                      "Min_dist_cs_3": min_dist_cs_3,
                      "Max_dist_cs_3": max_dist_cs_3,
                      "Min_cr_alt_cs_1": min_cr_alt_cs_1,
                      "Max_cr_alt_cs_1": max_cr_alt_cs_1,
                      "Min_cr_alt_cs_2": min_cr_alt_cs_2,
                      "Max_cr_alt_cs_2": max_cr_alt_cs_2,
                      "Min_cr_alt_cs_3": min_cr_alt_cs_3,
                      "Max_cr_alt_cs_3": max_cr_alt_cs_3,
                      "Global_min": global_min,
                      "Global_max": global_max
                     }
    
    return min_max_values

def produce_mesh(distance, global_min, global_max, n_dist_int, n_alt_int):
    
    # Define the grid for distance and corrected depth
    distance_grid = np.linspace(min(distance), max(distance), n_dist_int)
    
    corrected_alt_grid = np.linspace(math.floor(global_min), math.ceil(global_max), n_alt_int)
    distance_mesh, corrected_alt_mesh = np.meshgrid(distance_grid, corrected_alt_grid)
    
    return distance_grid, corrected_alt_grid, distance_mesh, corrected_alt_mesh

def perform_kriging(distance, corrected_altitude, vs_values, distance_grid, corrected_alt_grid):
    
    # Universal Kriging with linear drift terms
    UK = UniversalKriging(
        distance, corrected_altitude, vs_values,
        variogram_model='exponential',  # Try 'spherical', 'exponential', 'linear', etc.
        drift_terms=['regional_linear']  # Account for linear trends
    )

    # Perform Kriging interpolation on the grid
    vs_kriged, ss = UK.execute('grid', distance_grid, corrected_alt_grid)
    
    return vs_kriged, ss

def boundary_polygon_def(min_max_values, cross_sec):
    
    # Define boundaries for clipping (aligned with corrected_depth)
    x_min, x_max = min_max_values["Min_dist_cs_"+str(cross_sec)], min_max_values["Max_dist_cs_"+str(cross_sec)]
    y_min, y_max = min_max_values["Min_cr_alt_cs_"+str(cross_sec)], min_max_values["Max_cr_alt_cs_"+str(cross_sec)]
    boundary_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    
    return boundary_polygon

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

inputloc = "E:\\TOSHIBA-BACK-UP\\MARAS_EQ\\2D_CROSS_SECTIONs"
os.chdir(inputloc)

input_cs_1_fn = "Input_p1.xlsx"
input_cs_2_fn = "Input_p2.xlsx"
input_cs_3_fn = "Input_p3.xlsx"

n_dist_int = 1000   # The number of grids for distance parameter
n_alt_int = 200     # The number of grids for corrected altitude parameter

# Example DataFrame (Distance along section, Altitude, Depth, Vs)
data_cs_1 = pd.read_excel(input_cs_1_fn, sheet_name = "Sayfa1")
data_cs_2 = pd.read_excel(input_cs_2_fn, sheet_name = "Sayfa1")
data_cs_3 = pd.read_excel(input_cs_3_fn, sheet_name = "Sayfa1")

dist_cs_1, alt_cs_1, depth_cs_1, vs_cs_1, corr_alt_cs_1 = parse_input_data(data_cs_1)
dist_cs_2, alt_cs_2, depth_cs_2, vs_cs_2, corr_alt_cs_2 = parse_input_data(data_cs_2)
dist_cs_3, alt_cs_3, depth_cs_3, vs_cs_3, corr_alt_cs_3 = parse_input_data(data_cs_3)

min_max_values = get_dist_and_alt_boundaries(dist_cs_1, dist_cs_2, dist_cs_3, corr_alt_cs_1,
                                             corr_alt_cs_2, corr_alt_cs_3)

dist_grid, corr_alt_grid, dist_mesh, corr_alt_mesh = produce_mesh(dist_cs_3, min_max_values["Global_min"],
                                                                  min_max_values["Global_max"], n_dist_int, n_alt_int)

vs_kriged, ss = perform_kriging(dist_cs_3, corr_alt_cs_3, vs_cs_3, dist_grid, corr_alt_grid)

# Plot the interpolated Vs values
plt.figure(figsize=(10, 6))
contour = plt.contour(dist_mesh, corr_alt_mesh, vs_kriged, levels=100, cmap='viridis')
plt.colorbar(contour, label='Vs (m/s)')
plt.scatter(dist_cs_3, corr_alt_cs_3, c=vs_cs_3, edgecolors='k', label='Known Data Points')
plt.xlabel('Distance (m)')
plt.ylabel('Altitude (m)')
#plt.gca().invert_yaxis()
plt.show()

cross_sec = 3 # Cross-section to be exported as .dxf file.

boundary_polygon = boundary_polygon_def(min_max_values, cross_sec)

dxf_filename = "Vs_Profile_Cross_Section_Clipped_"+str(cross_sec)+".dxf"
doc = ezdxf.new()
msp = doc.modelspace()

for i, (collection, level) in enumerate(zip(contour.collections, contour.levels)):
    # This is the Vs value for this contour
    vs_value = level

    # Name each layer based on that Vs value
    layer_name = f"Vs_{int(round(vs_value, 0))}"
    if layer_name not in doc.layers:
        doc.layers.new(name=layer_name)

    # For each path (line) in this contour level
    for path in collection.get_paths():
        points = path.vertices
        if (points is not None) and (len(points) > 1):
            # Convert to a Shapely LineString and CLIP it
            line = LineString(points)
            clipped_geom = line.intersection(boundary_polygon)

            # The intersection can yield:
            # - An empty geometry
            # - A single LineString
            # - A MultiLineString
            # - Potentially polygons if lines lie exactly on the boundary
            if clipped_geom.is_empty:
                # Nothing to export
                continue

            # Collect the final lines to be exported
            if clipped_geom.geom_type == "LineString":
                lines_to_export = [clipped_geom]
            elif clipped_geom.geom_type == "MultiLineString":
                # Extract each sub-line
                lines_to_export = [
                    geom for geom in clipped_geom.geoms 
                    if geom.geom_type == "LineString"
                ]
            else:
                # E.g., a Polygon or something else. Skip or handle differently.
                lines_to_export = []

            # Export clipped lines to DXF
            for clipped_line in lines_to_export:
                # (a) Add the line itself
                msp.add_lwpolyline(
                    list(clipped_line.coords),
                    close=False,
                    dxfattribs={'layer': layer_name}
                )

                # (b) Optionally, label the line with text
                #     (For example, place a text label at the midpoint)
                coords_list = list(clipped_line.coords)
                mid_index = len(coords_list) // 2
                mx, my = coords_list[mid_index]
                label_text = f"{int(round(vs_value, 0))}"
                text = msp.add_text(
                    label_text,
                    dxfattribs={
                        'layer': layer_name,
                        'height': 2.5,
                        # 'style': 'STANDARD',  # or another text style if desired
                        # 'rotation': 0.0,
                        # 'color': 7,
                        # etc.
                    }
                )
                
                # Set the primary insertion point:
                text.set_dxf_attrib('insert', (mx, my))
                
                # If you also want text centered horizontally and vertically:
                # (halign=1 => center; valign=2 => middle, for single-line TEXT entities)
                text.set_dxf_attrib('halign', 1)
                text.set_dxf_attrib('valign', 2)
                
                # To get alignment to actually work for single-line TEXT,
                # you must also set the alignment point. 
                # In older versions of ezdxf, that is the same as 'insert' or a separate param:
                text.set_dxf_attrib('align_point', (mx, my))

doc.saveas(dxf_filename)
print(f"DXF file saved to {dxf_filename}")

