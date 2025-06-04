#!/usr/bin/env python3
"""
Standalone Cyst Analysis Script

This script contains the core functions for analyzing cyst masks:
- calculate_cyst_volume: Calculate volume of specified label groups
- calculate_blob_count: Count connected components/blobs
- calculate_circularity: Calculate circularity metric using 2D projection
- calculate_ellipsoid_volume: Calculate volume by fitting to ellipsoid

Dependencies:
pip install nibabel numpy scipy scikit-image shapely
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from scipy.spatial.distance import pdist, squareform
from skimage.measure import find_contours
from shapely.geometry import Polygon
from shapely.ops import unary_union


class CystAnalyzer:
    def __init__(self, label_groups):
        """
        Initialize the cyst analyzer.
        
        Args:
            label_groups: List of tuples (group_name, [label_values])
                         e.g., [('solid', [1, 2]), ('fluid', [3, 4])]
        """
        self.label_groups = label_groups
        
        # Collect all labels and create mapping
        self.labels = []
        self.label_to_group = {}
        for group_name, labels in self.label_groups:
            self.labels.extend(labels)
            for label in labels:
                self.label_to_group[label] = group_name

    def calculate_cyst_volume(self, mask_path):
        """
        Calculate the volume of specified label groups in a cyst mask.
        
        Args:
            mask_path: Path to the NIfTI mask file
            
        Returns:
            dict: Dictionary containing volume measurements and additional metrics
                 Keys: group volumes, 'Total', 'blob_count', 'circularity', 'ellipsoid_volume'
        """
        try:
            # Load mask data
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            mask_data_rounded = np.round(mask_data)
            mask_data = mask_data_rounded.astype(int)

            # Create a mask including only the specified labels
            mask = np.isin(mask_data, self.labels)

            # Ensure there are voxels to process
            if not np.any(mask):
                print(f"Warning: No specified labels found in mask {mask_path}")
                return None

            # Get voxel spacing and calculate voxel volume
            voxel_spacing = mask_img.header.get_zooms()[:3]
            voxel_spacing = np.array(voxel_spacing)
            voxel_volume = np.prod(voxel_spacing)

            # Calculate volume for each label group
            group_volumes = {}
            total_volume = 0
            for group_name, labels in self.label_groups:
                group_mask = np.isin(mask_data, labels)
                group_volume = np.sum(group_mask) * voxel_volume
                group_volumes[group_name] = group_volume
                total_volume += group_volume
            group_volumes['Total'] = total_volume

            # Calculate additional metrics
            blob_count = self.calculate_blob_count(mask)
            circularity = self.calculate_circularity(mask_data, voxel_spacing)
            ellipsoid_volume = self.calculate_ellipsoid_volume(mask_data, voxel_spacing)

            # Combine all results
            results = group_volumes.copy()
            results.update({
                'blob_count': blob_count,
                'circularity': circularity,
                'ellipsoid_volume': ellipsoid_volume
            })

            return results

        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            return None

    def calculate_blob_count(self, mask_data):
        """
        Calculate the number of connected components (blobs) in the mask data.
        
        Args:
            mask_data: Binary mask array
            
        Returns:
            int: Number of connected components
        """
        # Count connected regions
        labeled_array, num_features = label(mask_data > 0)
        return num_features

    def calculate_circularity(self, mask_data, voxel_spacing):
        """
        Calculate the circularity of the cyst mask using 2D projection.
        
        Args:
            mask_data: 3D mask array
            voxel_spacing: Array of voxel spacings along each axis
            
        Returns:
            float: Circularity value (0-1, where 1 is perfect circle)
        """
        try:
            # Create a mask including only the specified labels
            mask = np.isin(mask_data, self.labels)

            # Determine the axis with the largest voxel spacing for projection
            axis_to_project = np.argmax(voxel_spacing)

            # Project the 3D mask into 2D by summing along the chosen axis
            projection = np.sum(mask, axis=axis_to_project)
            # Binarize the projection
            projection = projection > 0

            # Find contours in the projection
            contours = find_contours(projection, level=0.5)
            
            if not contours:
                return 0.0

            polygons = []
            for contour in contours:
                # Need at least 3 points for a valid polygon
                if len(contour) < 3:
                    continue
                
                # Convert (row, col) -> (x, y) for Shapely
                coords = [(c[1], c[0]) for c in contour]
                poly = Polygon(coords)
                
                # Filter out invalid or empty polygons
                if not poly.is_valid or poly.area == 0:
                    continue
                polygons.append(poly)

            if not polygons:
                return 0.0

            # Combine multiple polygons if they exist
            union_poly = unary_union(polygons)
            
            # Calculate circularity using standard formula
            perimeter_shapely = union_poly.length
            area_shapely = union_poly.area
            
            if perimeter_shapely > 0:
                circularity = 4 * np.pi * area_shapely / (perimeter_shapely ** 2)
            else:
                circularity = 0.0

            # Clamp circularity to [0, 1] range
            return min(circularity, 1.0)

        except Exception as e:
            print(f"Error calculating circularity: {e}")
            return 0.0

    def calculate_ellipsoid_volume(self, mask_data, voxel_spacing):
        """
        Calculate the volume by fitting the mask to an ellipsoid using maximal area slice method.
        
        Args:
            mask_data: 3D mask array
            voxel_spacing: Array of voxel spacings along each axis
            
        Returns:
            float: Calculated ellipsoid volume
        """
        try:
            # Create a mask including only the specified labels
            mask = np.isin(mask_data, self.labels)

            if not np.any(mask):
                return 0.0

            voxel_spacing = np.array(voxel_spacing)
            
            # Determine the axis with the largest voxel spacing (lowest resolution)
            primary_axis = np.argmax(voxel_spacing)

            # Sum over the other two axes to get area for each slice
            other_axes = tuple(i for i in range(mask_data.ndim) if i != primary_axis)
            areas = np.sum(mask, axis=other_axes)

            # Find the slice with maximum area
            max_area_slice_index = np.argmax(areas)
            max_area = areas[max_area_slice_index]

            if max_area == 0:
                return 0.0

            # Extract the slice with maximum area
            if primary_axis == 0:
                max_slice_mask = mask[max_area_slice_index, :, :]
                spacing_2d = voxel_spacing[1:]
            elif primary_axis == 1:
                max_slice_mask = mask[:, max_area_slice_index, :]
                spacing_2d = voxel_spacing[::2]
            else:
                max_slice_mask = mask[:, :, max_area_slice_index]
                spacing_2d = voxel_spacing[:2]

            # Find coordinates of mask pixels in physical units
            coords = np.argwhere(max_slice_mask)
            if coords.shape[0] < 2:
                return 0.0

            coords_physical = coords * spacing_2d

            # Find the two furthest points (major axis)
            distances = squareform(pdist(coords_physical))
            idx_max = np.unravel_index(np.argmax(distances), distances.shape)
            point1 = coords_physical[idx_max[0]]
            point2 = coords_physical[idx_max[1]]
            r1 = np.linalg.norm(point1 - point2) / 2  # Major axis radius

            # Find minor axis length
            vector = point2 - point1
            if np.linalg.norm(vector) == 0:
                return 0.0
                
            vector_norm = vector / np.linalg.norm(vector)

            # Project all points onto major axis and find perpendicular distances
            projections = np.dot(coords_physical - point1, vector_norm)
            projected_points = np.outer(projections, vector_norm) + point1
            distances_perp = np.linalg.norm(coords_physical - projected_points, axis=1)
            r2 = distances_perp.max()  # Minor axis radius

            # Compute extent along primary axis
            coords_3d = np.argwhere(mask)
            axis_coords = coords_3d[:, primary_axis]
            z_min = axis_coords.min()
            z_max = axis_coords.max()
            Lz = (z_max - z_min + 1) * voxel_spacing[primary_axis]
            r_z = Lz / 2

            # Calculate ellipsoid volume
            volume = (4 / 3) * np.pi * r1 * r2 * r_z

            return volume

        except Exception as e:
            print(f"Error calculating ellipsoid volume: {e}")
            return 0.0


def example_usage():
    """
    Example of how to use the CystAnalyzer class.
    """
    # Define label groups (customize based on your labeling scheme)
    label_groups = [
        ('solid', [1, 2]),    # Labels 1 and 2 represent solid tissue
        ('fluid', [3, 4]),    # Labels 3 and 4 represent fluid
        ('other', [5])        # Label 5 represents other tissue
    ]
    
    # Initialize analyzer
    analyzer = CystAnalyzer(label_groups)
    
    # Analyze a single mask file
    mask_path = "path/to/your/cyst_mask.nii.gz"
    
    if os.path.exists(mask_path):
        results = analyzer.calculate_cyst_volume(mask_path)
        
        if results:
            print("Analysis Results:")
            print("-" * 40)
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
    else:
        print(f"File not found: {mask_path}")
        print("Please update the mask_path variable with the correct file path.")


if __name__ == "__main__":
    example_usage()
