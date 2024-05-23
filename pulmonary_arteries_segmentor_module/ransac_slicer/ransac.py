#!/usr/bin/env python-real
from .volume import volume
from .popup_utils import CustomProgressBar, CustomStatusDialog
from .cylinder_ransac import (
    sample_around_cylinder,
    track_branch,
    config,
)

from .cylinder import cylinder, closest_branch
import numpy as np
from ransac_slicer.graph_branches import GraphBranches
import qt


def interpolate_point(
    cyl_0: cylinder, cyl_1: cylinder, vol: volume, cfg: config, distance: float
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """
    Interpolate points between two cylinders.
    Points must be sparsed from at least a certain distance.
    Each new centerline point comes with its associated contour points.

    Parameters
    ----------
    cyl_0: first cylinder to interpolate from (excluded).
    cyl_1: second cylinder to interpolate to (excluded).
    vol: the volume from which points are sampled.
    cfg: configuration regarding the RANSAC algorithm.
    distance: minimum distance between interpolated points allowed.

    Returns
    ----------

    list[np.ndarray]
    A list of center points.

    list[list[np.ndarray]]
    A list of contours points, each center point has its list of contours.
    """
    direction = cyl_1.center - cyl_0.center
    radius_diff = cyl_1.radius - cyl_0.radius
    nb_points = int(np.linalg.norm(direction) // distance)

    centers, contour_points = [], []

    centers_radius = zip(
        [
            cyl_0.center + (idx / (nb_points + 1)) * direction
            for idx in range(1, nb_points + 1)
        ],
        [
            cyl_0.radius + (idx / (nb_points + 1)) * radius_diff
            for idx in range(1, nb_points + 1)
        ],
    )
    for center, radius in centers_radius:
        cyl = cylinder(center=center, radius=radius, direction=cyl_1.center - center)
        inliers = None
        tries = 0

        # Maximum number of interpolation attempt is 5
        while inliers is None and tries < 5:
            inliers = sample_around_cylinder(vol, cyl, cfg)
            tries += 1

        if inliers is not None:
            centers.append(center)
            contour_points.append(inliers)

    return centers, contour_points


def interpolate_centerline(
    cylinders: list[cylinder],
    contour_points: list[list[np.ndarray]],
    vol: volume,
    cfg: config,
    distance: float,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[float]]:
    """
    Refine the points of a centerline according to a certain minimum distance between points.

    Parameters
    ----------
    cylinders: cylinder fitted through RANSAC algorithm.
    contour_points: list of contour point of each cylinder.
    vol: the volume from which points are sampled.
    cfg: configuration regarding the RANSAC algorithm.
    distance: minimum distance between interpolated points allowed.

    Returns
    ----------

    list[np.ndarray]
    A list of center points.

    list[list[np.ndarray]]
    A list of contours points, each center point has its list of contours.

    list[float]
    A list of underestimated radius, each center point has an underestimated radius.
    """
    new_centerline, new_contour = [cylinders[0].center], [contour_points[0]]

    for idx in CustomProgressBar(
        iterable=range(len(cylinders) - 1),
        quantity_to_measure="segments to interpolate",
        windowTitle="Interpolating centerline points...",
        width=300,
    ):
        tmp_centerline, tmp_contour_points = interpolate_point(
            cylinders[idx],
            cylinders[idx + 1],
            vol,
            cfg,
            distance,
        )
        tmp_centerline.append(cylinders[idx + 1].center)
        tmp_contour_points.append(contour_points[idx + 1])
        new_centerline.extend(tmp_centerline)
        new_contour.extend(tmp_contour_points)

    return (
        np.array(new_centerline),
        new_contour,
        [
            np.linalg.norm(cp - c, axis=1).min()
            for cp, c in zip(new_contour, new_centerline)
        ],
    )


def run_ransac(
    vol: volume,
    starting_point: np.ndarray,
    direction_point: np.ndarray,
    starting_radius: float,
    percent_inlier_points: int,
    threshold: int,
    centerline_resolution: float,
    graph_branches: GraphBranches,
    isNewBranch: float,
    progress_dialog: CustomStatusDialog,
) -> GraphBranches:
    """
    Run the RANSAC algorithm to fit a cylinder according to the parameters indicated by the user.

    Apply a post-processing by refining the centerline points up to a certain resolution and updates the
    graph branch.

    Parameters
    ----------
    vol: the volume from which points are sampled.
    starting_point: starting point of the first cylinder.
    direction_point: point indicating the direction of the first cylinder.
    starting_radius: radius in mm of the first cylinder to fit.
    percent_inlier_points: percent of inlier required to be considered a correct model.
    threshold: threshold from which points are considered inlier.
    centerline_resolution: minimum distance between centerline points.
    graph_branches: the graph branch object.
    isNewBranch: flag to tell if it is the first branch or not.
    progress_dialog: UI window to inform the user on the state of the branch tracking.

    Returns
    ----------

    GraphBranches
    Updated graph
    """
    # Input info for branch tracking (in RAS coordinates)
    if isNewBranch:
        _, _, idx_cb, idx_cyl = closest_branch(
            starting_point, graph_branches.branch_list
        )
        if idx_cyl == len(graph_branches.centerlines[idx_cb]) - 2:
            idx_cyl = len(graph_branches.centerlines[idx_cb]) - 1

        # Update Graph
        parent_node = graph_branches.names[idx_cb]
        # Case when the closest node is the last point of a branch, we concatenate the two branches
        if idx_cyl == len(graph_branches.centerlines[idx_cb]) - 1:
            end_centerline, end_center_radius, end_contour_point = (
                graph_branches.centerlines[idx_cb][idx_cyl : idx_cyl + 1],
                graph_branches.centerline_radius[idx_cb][idx_cyl : idx_cyl + 1],
                graph_branches.contours_points[idx_cb][idx_cyl : idx_cyl + 1],
            )
        # Case when the closest node is the first point of a branch, we had the branch to the parent of the closest branch, thus the branch way have more than 2 childs
        elif idx_cyl == 0:
            parent_node = graph_branches.tree_widget.getParentNodeId(parent_node)
            end_centerline, end_center_radius, end_contour_point = (
                graph_branches.centerlines[idx_cb][:1],
                graph_branches.centerline_radius[idx_cb][:1],
                graph_branches.contours_points[idx_cb][:1],
            )
        # Case when the closest node is in the middle of a branch, we split the branch at the intersection point
        else:
            (
                end_centerline,
                end_center_radius,
                end_contour_point,
            ) = graph_branches.split_branch(idx_cb, idx_cyl, parent_node)
    else:
        parent_node = None
        graph_branches.nodes.append(starting_point)
        end_centerline = np.empty((0, 3))
        end_center_radius = []
        end_contour_point = []

    direction_point = direction_point - starting_point

    init_radius = starting_radius

    # Tracking configuration
    pct_inl = percent_inlier_points / 100.0
    err = threshold / 100.0
    cfg = config(percent_inliers=pct_inl, threshold=err)

    # Initialize tracking
    cyl = cylinder(starting_point, init_radius, direction_point, height=0)

    # Perform tracking
    centerline, contour_points, centerline_radius, cylinders = track_branch(
        vol,
        cyl,
        cfg,
        end_centerline,
        end_center_radius,
        end_contour_point,
        [elt for branch in graph_branches.branch_list for elt in branch],
        progress_dialog,
    )

    if len(centerline) <= 1:
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText("Could not find any branch")
        msg.exec_()
        graph_branches.on_merge_only_child(parent_node)
        return graph_branches

    # In the case of a split, we had the closest point to the cylinders list so it can be interpolated aswell
    if len(centerline) - 1 == len(cylinders):
        cylinders.insert(
            0,
            cylinder(
                center=centerline[0],
                radius=centerline_radius[0],
                direction=centerline[1] - centerline[0],
            ),
        )

    # Interpolate points
    centerline, contour_points, centerline_radius = interpolate_centerline(
        cylinders,
        contour_points,
        vol,
        cfg,
        distance=centerline_resolution,
    )
    del cylinders

    graph_branches.nodes.append(centerline[-1])
    edge_begin = (
        graph_branches.edges[graph_branches.names.index(parent_node)][1]
        if isNewBranch
        else len(graph_branches.nodes) - 2
    )
    graph_branches.create_new_branch(
        (edge_begin, len(graph_branches.nodes) - 1),
        centerline,
        contour_points,
        centerline_radius,
        parent_node,
    )

    return graph_branches
