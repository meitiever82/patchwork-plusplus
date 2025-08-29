#!/usr/bin/env python3
"""
PNG to PGM Grid Map Converter with Origin Alignment
Converts PNG format grid maps to PGM format with proper world coordinate handling
and alignment based on known origin pixel position.
Compatible with ROS map_server format and vSLAM coordinate systems.
"""

import cv2
import numpy as np
import argparse
import os
from typing import Tuple, Optional


class GridMapParams:
    def __init__(self, resolution: float = 0.1, ground_value: int = 0, 
                 unknown_value: int = 100, obstacle_value: int = 200):
        self.resolution = resolution
        self.ground_value = ground_value
        self.unknown_value = unknown_value
        self.obstacle_value = obstacle_value
        
        # World coordinate bounds (to be calculated from origin alignment)
        self.world_x_min = 0.0
        self.world_x_max = 0.0
        self.world_y_min = 0.0
        self.world_y_max = 0.0
        
        # Origin pixel position in PNG (where point cloud origin projects to)
        self.origin_pixel_x = 0
        self.origin_pixel_y = 0
        
        # Derived parameters
        self.map_width = 0.0
        self.map_height = 0.0


def png_to_grid_values(png_image: np.ndarray, params: GridMapParams) -> np.ndarray:
    """
    Convert PNG image values to grid map values.
    Assumes: 0 (black) = ground, 127 (gray) = unknown, 255 (white) = obstacle
    """
    grid_map = np.zeros_like(png_image, dtype=np.uint8)
    
    # Convert visualization values back to grid values
    grid_map[png_image == 0] = params.ground_value      # black -> ground (0)
    grid_map[png_image == 127] = params.unknown_value   # gray -> unknown (100)  
    grid_map[png_image == 255] = params.obstacle_value  # white -> obstacle (200)
    
    # Handle other values by mapping to closest category
    mask_other = (png_image != 0) & (png_image != 127) & (png_image != 255)
    if np.any(mask_other):
        print(f"Warning: Found {np.sum(mask_other)} pixels with non-standard values, mapping to closest category")
        # Map based on brightness
        grid_map[mask_other & (png_image < 64)] = params.ground_value
        grid_map[mask_other & (png_image >= 64) & (png_image < 192)] = params.unknown_value
        grid_map[mask_other & (png_image >= 192)] = params.obstacle_value
    
    return grid_map


def calculate_world_bounds_from_origin(png_shape: Tuple[int, int], 
                                     origin_pixel_x: int, origin_pixel_y: int,
                                     origin_world_x: float, origin_world_y: float,
                                     resolution: float) -> Tuple[float, float, float, float]:
    """
    Calculate world coordinate bounds based on origin alignment.
    Keep same coordinate system as PNG (no flipping).
    
    Args:
        png_shape: (height, width) of PNG image
        origin_pixel_x: X pixel coordinate where point cloud origin projects to
        origin_pixel_y: Y pixel coordinate where point cloud origin projects to  
        origin_world_x: World X coordinate of the point cloud origin (usually 0.0)
        origin_world_y: World Y coordinate of the point cloud origin (usually 0.0)
        resolution: meters per pixel
        
    Returns:
        (world_x_min, world_x_max, world_y_min, world_y_max)
    """
    height, width = png_shape
    
    # Calculate world bounds based on origin alignment
    # Keep PNG coordinate system: (0,0) at top-left, x->right, y->down
    
    # Distance from origin to image boundaries in pixels
    pixels_to_left = origin_pixel_x
    pixels_to_right = width - origin_pixel_x
    pixels_to_top = origin_pixel_y  
    pixels_to_bottom = height - origin_pixel_y
    
    # Convert to world coordinates - keep same orientation as PNG
    world_x_min = origin_world_x - pixels_to_left * resolution
    world_x_max = origin_world_x + pixels_to_right * resolution
    world_y_min = origin_world_y - pixels_to_top * resolution    # top in PNG = smaller Y values
    world_y_max = origin_world_y + pixels_to_bottom * resolution # bottom in PNG = larger Y values
    
    return world_x_min, world_x_max, world_y_min, world_y_max


def save_pgm_file(grid_map: np.ndarray, pgm_path: str, params: GridMapParams) -> None:
    """Save grid map as PGM file keeping same coordinate system as PNG (no flipping)."""
    
    # Calculate map center and grid origin
    map_center_x = (params.world_x_min + params.world_x_max) / 2.0
    map_center_y = (params.world_y_min + params.world_y_max) / 2.0
    grid_origin_x = params.world_x_min
    grid_origin_y = params.world_y_min
    
    with open(pgm_path, 'w') as f:
        # PGM header (ASCII format P2)
        f.write("P2\n")
        f.write("# Grid map converted from PNG with origin alignment\n")
        f.write(f"# Resolution: {params.resolution} m/pixel\n")
        f.write(f"# Map center: ({map_center_x:.3f}, {map_center_y:.3f})\n")
        f.write(f"# Grid origin: ({grid_origin_x:.3f}, {grid_origin_y:.3f})\n")
        f.write(f"# Point cloud origin pixel: ({params.origin_pixel_x}, {params.origin_pixel_y})\n")
        f.write(f"# Map size: {params.map_width:.3f}m x {params.map_height:.3f}m\n")
        f.write("# Coordinate system: same as PNG (top-left origin, x->right, y->down)\n")
        f.write(f"# Values: {params.ground_value}=ground, {params.unknown_value}=unknown, {params.obstacle_value}=obstacle\n")
        f.write(f"{grid_map.shape[1]} {grid_map.shape[0]}\n")
        f.write("255\n")  # max value
        
        # Write data in original PNG order (no coordinate system conversion)
        # Keep the same orientation as the input PNG
        for row in range(grid_map.shape[0]):  # From top to bottom, same as PNG
            row_values = []
            for col in range(grid_map.shape[1]):
                row_values.append(str(grid_map[row, col]))
            f.write(" ".join(row_values) + "\n")


def save_yaml_file(yaml_path: str, pgm_filename: str, params: GridMapParams) -> None:
    """Save map metadata as YAML file with correct world coordinates."""
    
    with open(yaml_path, 'w') as f:
        f.write(f"image: {pgm_filename}\n")
        f.write(f"resolution: {params.resolution}\n")
        
        # Grid map origin is at world_x_min, world_y_min
        # This is where grid coordinate (0,0) maps to in world coordinates
        f.write(f"origin: [{params.world_x_min:.6f}, {params.world_y_min:.6f}, 0.0]\n")
        
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")
        
        # Add coordinate transform information for vSLAM system
        f.write("# Coordinate Transform Information\n")
        f.write("pcl_to_grid_transform:\n")
        f.write(f"  resolution: {params.resolution}\n")
        f.write(f"  grid_origin_world_x: {params.world_x_min:.6f}\n")
        f.write(f"  grid_origin_world_y: {params.world_y_min:.6f}\n")
        f.write(f"  world_bounds: [{params.world_x_min:.6f}, {params.world_x_max:.6f}, {params.world_y_min:.6f}, {params.world_y_max:.6f}]\n")
        f.write("  coordinate_system: top_left_origin\n")
        f.write(f"  origin_pixel_position: [{params.origin_pixel_x}, {params.origin_pixel_y}]\n")
        
        # Grid map coordinate system explanation
        f.write("# Grid map coordinate system:\n")
        f.write("# - Origin at top-left corner (same as PNG)\n")
        f.write("# - X-axis points right\n")
        f.write("# - Y-axis points down\n")
        f.write(f"# - Grid (0,0) = World ({params.world_x_min:.6f}, {params.world_y_min:.6f})\n")
        f.write(f"# - Point cloud origin (0,0,0) projects to pixel ({params.origin_pixel_x}, {params.origin_pixel_y})\n")
        f.write(f"# - Values: {params.ground_value}=ground, {params.unknown_value}=unknown, {params.obstacle_value}=obstacle\n")


def convert_png_to_pgm_with_origin(png_path: str, output_path: str, params: GridMapParams,
                                  origin_pixel_x: int, origin_pixel_y: int,
                                  origin_world_x: float = 0.0, origin_world_y: float = 0.0) -> None:
    """Main conversion function with origin-based alignment."""
    
    # Load PNG image
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"PNG file not found: {png_path}")
    
    # Read as grayscale
    png_image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if png_image is None:
        raise ValueError(f"Cannot read PNG image: {png_path}")
    
    print(f"Loaded PNG image: {png_image.shape[1]}x{png_image.shape[0]} pixels")
    print(f"Point cloud origin projects to pixel: ({origin_pixel_x}, {origin_pixel_y})")
    print(f"Point cloud origin world coordinate: ({origin_world_x}, {origin_world_y})")
    
    # Store origin information
    params.origin_pixel_x = origin_pixel_x
    params.origin_pixel_y = origin_pixel_y
    
    # Calculate world coordinate bounds based on origin alignment
    world_x_min, world_x_max, world_y_min, world_y_max = calculate_world_bounds_from_origin(
        png_image.shape, origin_pixel_x, origin_pixel_y,
        origin_world_x, origin_world_y, params.resolution
    )
    
    # Set world coordinate bounds
    params.world_x_min = world_x_min
    params.world_x_max = world_x_max
    params.world_y_min = world_y_min
    params.world_y_max = world_y_max
    
    # Calculate map dimensions
    params.map_width = world_x_max - world_x_min
    params.map_height = world_y_max - world_y_min
    
    print(f"\nCalculated world coordinate bounds:")
    print(f"  X: [{world_x_min:.3f}, {world_x_max:.3f}] (width: {params.map_width:.3f}m)")
    print(f"  Y: [{world_y_min:.3f}, {world_y_max:.3f}] (height: {params.map_height:.3f}m)")
    
    # Verify the alignment calculation
    print(f"\nAlignment verification:")
    world_origin_pixel_x = (origin_world_x - world_x_min) / params.resolution
    world_origin_pixel_y = (origin_world_y - world_y_min) / params.resolution  # Keep same Y orientation as PNG
    print(f"  World origin ({origin_world_x}, {origin_world_y}) should map to pixel ({world_origin_pixel_x:.1f}, {world_origin_pixel_y:.1f})")
    print(f"  Input origin pixel position: ({origin_pixel_x}, {origin_pixel_y})")
    if abs(world_origin_pixel_x - origin_pixel_x) < 0.1 and abs(world_origin_pixel_y - origin_pixel_y) < 0.1:
        print("  ✓ Alignment calculation is correct!")
    else:
        print("  ⚠ Warning: Alignment calculation may have issues!")
    
    # Convert PNG values to grid map values
    grid_map = png_to_grid_values(png_image, params)
    
    # Generate output filenames
    base_name = os.path.splitext(output_path)[0]
    pgm_path = base_name + "_aligned_gridmap.pgm"
    yaml_path = base_name + "_aligned_gridmap.yaml"
    pgm_filename = os.path.basename(pgm_path)
    
    # Save PGM file
    save_pgm_file(grid_map, pgm_path, params)
    print(f"\nGrid map saved as PGM: {pgm_path}")
    
    # Save YAML metadata
    save_yaml_file(yaml_path, pgm_filename, params)
    print(f"Map metadata saved as YAML: {yaml_path}")
    
    # Print statistics
    total_pixels = grid_map.size
    ground_pixels = np.sum(grid_map == params.ground_value)
    unknown_pixels = np.sum(grid_map == params.unknown_value)
    obstacle_pixels = np.sum(grid_map == params.obstacle_value)
    
    print(f"\nConversion statistics:")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Ground pixels: {ground_pixels} ({100*ground_pixels/total_pixels:.1f}%)")
    print(f"  Unknown pixels: {unknown_pixels} ({100*unknown_pixels/total_pixels:.1f}%)")
    print(f"  Obstacle pixels: {obstacle_pixels} ({100*obstacle_pixels/total_pixels:.1f}%)")
    
    # Print coordinate transformation information for vSLAM system
    print(f"\nCoordinate Transform Information:")
    print(f"  Grid origin world coordinates: ({world_x_min:.6f}, {world_y_min:.6f})")
    print(f"  Map center: ({(world_x_min + world_x_max)/2.0:.6f}, {(world_y_min + world_y_max)/2.0:.6f})")
    print(f"  Point cloud origin pixel: ({origin_pixel_x}, {origin_pixel_y})")
    print(f"  Resolution: {params.resolution} m/pixel")


def main():
    parser = argparse.ArgumentParser(description="Convert PNG grid map to PGM format with origin-based alignment")
    parser.add_argument("input_png", help="Input PNG file path")
    parser.add_argument("output_path", help="Output file path (without extension)")
    
    # Origin-based alignment parameters (required)
    parser.add_argument("--origin-pixel-x", type=int, required=True,
                       help="X pixel coordinate where point cloud origin (0,0,0) projects to in PNG")
    parser.add_argument("--origin-pixel-y", type=int, required=True,
                       help="Y pixel coordinate where point cloud origin (0,0,0) projects to in PNG")
    
    # World coordinate of the origin (optional, defaults to 0,0)
    parser.add_argument("--origin-world-x", type=float, default=0.0,
                       help="World X coordinate of point cloud origin (default: 0.0)")
    parser.add_argument("--origin-world-y", type=float, default=0.0,
                       help="World Y coordinate of point cloud origin (default: 0.0)")
    
    # Grid map parameters
    parser.add_argument("--resolution", "-r", type=float, required=True,
                       help="Map resolution in meters per pixel (from CloudCompare step)")
    parser.add_argument("--ground-value", type=int, default=0,
                       help="Grid value for ground/free space (default: 0)")
    parser.add_argument("--unknown-value", type=int, default=100,
                       help="Grid value for unknown space (default: 100)")
    parser.add_argument("--obstacle-value", type=int, default=200,
                       help="Grid value for obstacles (default: 200)")

    args = parser.parse_args()
    
    # Create parameters object
    params = GridMapParams(
        resolution=args.resolution,
        ground_value=args.ground_value,
        unknown_value=args.unknown_value,
        obstacle_value=args.obstacle_value
    )
    
    try:
        convert_png_to_pgm_with_origin(
            args.input_png, 
            args.output_path, 
            params,
            args.origin_pixel_x,
            args.origin_pixel_y,
            args.origin_world_x,
            args.origin_world_y
        )
        print("\nConversion completed successfully!")
        print("\nUsage example:")
        print(f"python {os.path.basename(__file__)} gridmap.png output \\")
        print(f"    --resolution 0.1 \\")
        print(f"    --origin-pixel-x 150 --origin-pixel-y 200 \\")
        print(f"    --origin-world-x 0.0 --origin-world-y 0.0")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())