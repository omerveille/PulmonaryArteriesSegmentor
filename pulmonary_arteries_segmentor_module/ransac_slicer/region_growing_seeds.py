from . import CustomStatusDialog
import slicer
import numpy as np
import vtk
from skimage.morphology import binary_dilation, ball
from .color_palettes import vessel_colors, contour_color
import time

def update_segment(
    segment_id: str,
    labelmap_node: slicer.vtkMRMLSegmentationNode,
    data: np.ndarray,
    segmentation_node: slicer.vtkMRMLLabelMapVolumeNode,
):
    vtk_segment_id = vtk.vtkStringArray()
    vtk_segment_id.InsertNextValue(segment_id)

    slicer.util.updateVolumeFromArray(labelmap_node, data)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap_node, segmentation_node, vtk_segment_id)

def compute_bbox(centerline_points: list[np.ndarray], radius: list[np.ndarray], lower_bound : np.ndarray, dimensions: np.ndarray):

    centerline_points = np.array(centerline_points)
    radius = np.array(radius)

    min_point = np.maximum(np.floor(np.min(centerline_points - radius, axis=0)).astype(int), lower_bound)
    max_point = np.minimum(np.ceil(np.max(centerline_points + radius, axis=0)).astype(int), dimensions)

    return min_point, max_point

def adapt_radius(radius: float, reduction_threshold : float, reduction_factor : float) -> float:
    return radius if radius <= reduction_threshold else (radius - reduction_threshold) * reduction_factor + reduction_threshold

def split_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def paint_segments(
    volume_node: slicer.vtkMRMLScalarVolumeNode,
    centerlines: list[np.ndarray],
    centerline_names: list[str],
    radius: list[list[float]],
    segmentation_node: slicer.vtkMRMLSegmentationNode,
    reduction_factor : float,
    reduction_threshold : float,
    contour_distance : int,
    merge_all_vessels: bool
) -> None:

    # Important variables
    voxel_spacing = np.array(volume_node.GetSpacing()[::-1])
    volume_dimensions = np.array(volume_node.GetImageData().GetDimensions()[::-1])
    segmentation = segmentation_node.GetSegmentation()

    # Get the ras to ijk matrix as numpy array
    ras_to_ijk = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(ras_to_ijk)
    np_ras_to_ijk = np.zeros(shape=(4, 4))
    ras_to_ijk.DeepCopy(np_ras_to_ijk.ravel(), ras_to_ijk)

    # Clear all segments of segmentation
    segmentation.RemoveAllSegments()

    # Ensure the labelmap has the same dimensions, spacing, orientation, localisation as the volume
    labelmap_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    labelmap_node.CopyOrientation(volume_node)
    labelmap_node.SetOrigin(volume_node.GetOrigin())
    labelmap_node.SetSpacing(voxel_spacing)
    ijk_to_ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras)
    labelmap_node.SetIJKToRASMatrix(ijk_to_ras)

    labelmap_node.CreateDefaultDisplayNodes()
    labelmap_node.SetAndObserveImageData(vtk.vtkImageData())
    labelmap_node.GetImageData().SetDimensions(volume_dimensions)

    # Set the labelmap pixel values to uint8 with one channel, it might become a problem later if there are more than 255 labels
    labelmap_node.GetImageData().AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Transform ras coordinates (real world coordinates) into ijk coordinates (voxel coordinates)
    centerlines = [[[(np_ras_to_ijk @ np.array([*point, 1]))[-2::-1] for point in list_part] for list_part in split_list(centerline, 3)] for centerline in centerlines]
    radius = [list(split_list(radius_list, 3)) for radius_list in radius]

    # Constants
    x, y, z = np.indices(volume_dimensions)
    low_bound = np.array([0, 0, 0], dtype=int)
    high_bound = volume_dimensions - 1


    # Paint centerlines
    treated_radius = [[[np.array([adapt_radius(r, reduction_threshold, reduction_factor)] * 3) / voxel_spacing for r in radius_part] for radius_part in radius_per_centerline] for radius_per_centerline in radius]
    contour_map = np.zeros(volume_dimensions, dtype=np.bool_)

    for centerline_idx, (centerline, radius_per_centerline, centerline_name) in enumerate(zip(centerlines, treated_radius, centerline_names)):
        segment_map = np.zeros(volume_dimensions, dtype=np.bool_)
        for (points, points_radius) in zip(centerline, radius_per_centerline):
            lower_edge, highter_edge = compute_bbox(points, points_radius, low_bound, volume_dimensions)
            for (point, radius_) in zip(points, points_radius):
                center_x, center_y, center_z = point
                radius_x, radius_y, radius_z = radius_

                sphere_map = (((x[lower_edge[0]:highter_edge[0], lower_edge[1]:highter_edge[1], lower_edge[2]:highter_edge[2]] - center_x)**2 / radius_x**2 + (y[lower_edge[0]:highter_edge[0], lower_edge[1]:highter_edge[1], lower_edge[2]:highter_edge[2]]  - center_y)**2 / radius_y**2 + (z[lower_edge[0]:highter_edge[0], lower_edge[1]:highter_edge[1], lower_edge[2]:highter_edge[2]] - center_z)**2 / radius_z**2) <= 1).astype(np.bool_)
                closest_pixel_to_paint = np.maximum(np.minimum([round(coord) for coord in point], high_bound, dtype=int), low_bound, dtype=int)

                segment_map[closest_pixel_to_paint[0], closest_pixel_to_paint[1], closest_pixel_to_paint[2]] = True
                segment_map[lower_edge[0]:highter_edge[0], lower_edge[1]:highter_edge[1], lower_edge[2]:highter_edge[2]] += sphere_map

                contour_map[closest_pixel_to_paint[0], closest_pixel_to_paint[1], closest_pixel_to_paint[2]] = True
                contour_map[lower_edge[0]:highter_edge[0], lower_edge[1]:highter_edge[1], lower_edge[2]:highter_edge[2]] += sphere_map

        if not merge_all_vessels:
            # Each vessels has its segment
            segment_id = segmentation.AddEmptySegment("", centerline_name, vessel_colors[centerline_idx % len(vessel_colors)])
            update_segment(segment_id, labelmap_node, segment_map.astype(np.uint8), segmentation_node)

    del segment_map
    del treated_radius

    if merge_all_vessels:
        # Each vessels has been merges into one single segment
        segment_id = segmentation.AddEmptySegment("", "vessels", vessel_colors[centerline_idx % len(vessel_colors)])
        update_segment(segment_id, labelmap_node, contour_map.astype(np.uint8), segmentation_node)

    # Paint contours

    # Filtering point that have not been shrunk
    centerline_and_radius = [[centerline_part, [np.array([r] * 3) / voxel_spacing for r in radius_part]] for (centerline, radius_per_centerline) in zip(centerlines, radius) for (centerline_part, radius_part) in zip(centerline, radius_per_centerline) if any(map(lambda x: x > reduction_threshold, radius_part))]

    for (points, points_radius) in centerline_and_radius:
        lower_edge, highter_edge = compute_bbox(points, points_radius, low_bound, volume_dimensions)
        for (point, radius_) in zip(points, points_radius):

            center_x, center_y, center_z = point
            radius_x, radius_y, radius_z = radius_

            sphere_map = (((x[lower_edge[0]:highter_edge[0], lower_edge[1]:highter_edge[1], lower_edge[2]:highter_edge[2]] - center_x)**2 / radius_x**2 + (y[lower_edge[0]:highter_edge[0], lower_edge[1]:highter_edge[1], lower_edge[2]:highter_edge[2]]  - center_y)**2 / radius_y**2 + (z[lower_edge[0]:highter_edge[0], lower_edge[1]:highter_edge[1], lower_edge[2]:highter_edge[2]] - center_z)**2 / radius_z**2) <= 1).astype(np.bool_)
            contour_map[lower_edge[0]:highter_edge[0], lower_edge[1]:highter_edge[1], lower_edge[2]:highter_edge[2]] += sphere_map

    progress_dialog = CustomStatusDialog(windowTitle="Computing contours ...", text="Please wait", width=300, height=50)
    # Timer added to let the interface load
    time.sleep(0.1)

    # Outter edge
    progress_dialog.setText("Computing the outter edge ...")
    contours_dilated = binary_dilation(contour_map, ball(radius=contour_distance + 2))

    # Inner edge
    progress_dialog.setText("Computing the inner edge ...")
    contours_dilated[binary_dilation(contour_map, ball(radius=contour_distance))] = False
    del contour_map

    progress_dialog.close()

    segment_id = segmentation.AddEmptySegment("", "Contour", contour_color)
    segmentation_node.GetDisplayNode().SetSegmentOpacity3D(segment_id, 0.1)
    update_segment(segment_id, labelmap_node, contours_dilated.astype(np.uint8), segmentation_node)

    slicer.mrmlScene.RemoveNode(labelmap_node)