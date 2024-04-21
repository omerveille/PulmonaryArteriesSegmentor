import math
import skimage.morphology
from . import make_custom_progress_bar, CustomStatusDialog
import slicer
import numpy as np
import vtk
import skimage
from .color_palettes import vessel_colors, contour_color
import time

def paint_segments(volume_node : slicer.vtkMRMLScalarVolumeNode, centerlines : list[np.ndarray], centerline_names : list[str], radius : list[list[float]], segmentation_node : slicer.vtkMRMLSegmentationNode) -> None:
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
    centerlines = [[(np_ras_to_ijk @ np.array([*point, 1]))[-2::-1] for point in centerline] for centerline in centerlines]

    centerlines_seeds_np_array, centerline_segment_ids = place_centerlines_seeds(centerlines, centerline_names, segmentation, volume_dimensions)
    contour_seed_np_array, contour_segment_id = place_contours_seeds(centerlines_seeds_np_array, centerlines, radius, voxel_spacing, segmentation_node, segmentation)

    slicer.util.updateVolumeFromArray(labelmap_node, centerlines_seeds_np_array)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap_node, segmentation_node, centerline_segment_ids)

    slicer.util.updateVolumeFromArray(labelmap_node, contour_seed_np_array)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap_node, segmentation_node, contour_segment_id)

    slicer.mrmlScene.RemoveNode(labelmap_node)

def place_centerlines_seeds(
        centerlines : list[list[np.ndarray]],
        centerline_names : list[str],
        segmentation : slicer.vtkSegmentation,
        volume_dimensions : np.ndarray
    ) -> None:

    segment_ids = vtk.vtkStringArray()

    max_values = np.array(volume_dimensions, dtype=int) - 1
    min_values = [0, 0, 0]
    numpy_label_map = np.zeros(shape=volume_dimensions, dtype=np.uint8)

    for centerline_idx in range(len(centerlines)):
        segment_ids.InsertNextValue(segmentation.AddEmptySegment("", centerline_names[centerline_idx], vessel_colors[centerline_idx % len(vessel_colors)]))
        for point in centerlines[centerline_idx]:
            # Clamps the the pixel coordinates to make sure it's inside the volume
            ijk_closest_point = np.maximum(np.minimum([round(coord) for coord in point], max_values, dtype=int), min_values, dtype=int)
            numpy_label_map[ijk_closest_point[0], ijk_closest_point[1], ijk_closest_point[2]] = centerline_idx + 1

    return numpy_label_map, segment_ids

def place_contours_seeds(
    centerline_seeds: np.ndarray,
    centerlines: list[list[float, float, float]],
    radius: list[float],
    spacing: list[float, float, float],
    segmentationNode: slicer.vtkMRMLSegmentationNode,
    segmentation: slicer.vtkSegmentation
) -> np.ndarray:

    contours = (centerline_seeds > 0).astype(np.bool_)
    x, y, z = np.indices(contours.shape)

    # Unwrapping to ease loading bar
    centerlines = [point for centerline in centerlines for point in centerline]
    radius = [radius_per_point for radius_list in radius for radius_per_point in radius_list]

    # Radius modified with spacing in case of non-isotropic volume
    radius = [np.array([r, r, r]) / spacing for r in radius]

    segment_id = vtk.vtkStringArray()
    segment_id.InsertNextValue(segmentation.AddEmptySegment("", "Contours", contour_color))
    segmentationNode.GetDisplayNode().SetSegmentOpacity3D(segment_id.GetValue(0), 0.1)

    progress_bar = make_custom_progress_bar(labelText="Computing contours ...", windowTitle="Computing contours ...", width=250)
    time.sleep(0.25)
    slicer.app.processEvents()

    for idx in range(len(centerlines)):
        center_x, center_y, center_z = centerlines[idx]
        radius_x, radius_y, radius_z = radius[idx]
        contours_tmp = (((x - center_x)**2 / radius_x**2 + (y - center_y)**2 / radius_y**2 + (z - center_z)**2 / radius_z**2) <= 1).astype(np.bool_)

        contours += contours_tmp

        progress_value = math.floor(((idx + 1) / len(centerlines)) * 100)
        if progress_value != progress_bar.value:
            progress_bar.value = progress_value
            slicer.app.processEvents()

    progress_bar.close()

    progress_dialog = CustomStatusDialog(windowTitle="Computing contours ...", text="Please wait", width=300, height=50)
    time.sleep(0.25)

    # Outter edge
    progress_dialog.setText("Computing the outter edge ...")
    contours_ball_6 = skimage.morphology.binary_dilation(contours, skimage.morphology.ball(radius=6))

    # Inner edge
    progress_dialog.setText("Computing the inner edge ...")
    contours_ball_6[skimage.morphology.binary_dilation(contours, skimage.morphology.ball(radius=4))] = False

    progress_dialog.setText("Processing done !")
    progress_dialog.close()
    return contours_ball_6.astype(np.uint8), segment_id