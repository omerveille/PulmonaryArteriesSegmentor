#!/usr/bin/env python-real
from .cylinder_ransac import (
    sample_around_cylinder,
    track_branch,
    config,
)

from . import ProgressBarTimer, make_custom_progress_bar
from .cylinder import cylinder, closest_branch
import numpy as np
import slicer
from ransac_slicer.graph_branches import GraphBranches
import qt


def interpolate_point(cyl_0, cyl_1, vol, cfg, distance):
    direction = cyl_1.center - cyl_0.center
    radius_diff = cyl_1.radius - cyl_0.radius
    nb_points = int(np.linalg.norm(direction) // distance)

    print(f"point every {distance} mm | number of points : {nb_points}")

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
    cylinders, contour_points, vol, cfg, distance=3
):
    progress_bar = progress_bar = make_custom_progress_bar(
        labelText="Interpolating centerline points...",
        windowTitle="Interpolating centerline points...",
        width=300,
    )

    new_centerline, new_contour = [cylinders[0].center], [contour_points[0]]
    last_percent = 0

    with ProgressBarTimer(total=len(cylinders) - 1) as timer:
        for idx in range(len(cylinders) - 1):
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

            elapsed_time, remaining_time, percent_done = timer.update()
            if percent_done - last_percent >= 1 or last_percent == 0 :
                last_percent = percent_done
                progress_bar.value = percent_done
                progress_bar.labelText = f"{timer.count}/{timer.total} segments to interpolate\nElapsed time: {ProgressBarTimer.format_time(elapsed_time)}\nRemaining time: {ProgressBarTimer.format_time(remaining_time)}"
                slicer.app.processEvents()

    progress_bar.hide()
    progress_bar.close()
    return (
        np.array(new_centerline),
        new_contour,
        [
            np.linalg.norm(cp - c, axis=1).min()
            for cp, c in zip(new_contour, new_centerline)
        ],
    )


def run_ransac(
    vol,
    starting_point,
    direction_point,
    starting_radius,
    pct_inlier_points,
    threshold,
    centerline_resolution,
    graph_branches: GraphBranches,
    isNewBranch,
    progress_dialog,
):
    # Input info for branch tracking (in RAS coordinates)
    if isNewBranch:
        _, _, idx_cb, idx_cyl = closest_branch(
            starting_point, graph_branches.branch_list
        )
        if idx_cyl == len(graph_branches.centers_lines[idx_cb]) - 2:
            idx_cyl = len(graph_branches.centers_lines[idx_cb]) - 1

        # Update Graph
        parent_node = graph_branches.names[idx_cb]
        # Case when the closest node is the last point of a branch, we concatenate the two branches
        if idx_cyl == len(graph_branches.centers_lines[idx_cb]) - 1:
            end_center_line, end_center_radius, end_contour_point = (
                graph_branches.centers_lines[idx_cb][idx_cyl : idx_cyl + 1],
                graph_branches.centers_line_radius[idx_cb][idx_cyl : idx_cyl + 1],
                graph_branches.contours_points[idx_cb][idx_cyl : idx_cyl + 1],
            )
        # Case when the closest node is the first point of a branch, we had the branch to the parent of the closest branch, thus the branch way have more than 2 childs
        elif idx_cyl == 0:
            parent_node = graph_branches.tree_widget.getParentNodeId(parent_node)
            end_center_line, end_center_radius, end_contour_point = (
                graph_branches.centers_lines[idx_cb][:1],
                graph_branches.centers_line_radius[idx_cb][:1],
                graph_branches.contours_points[idx_cb][:1],
            )
        # Case when the closest node is in the middle of a branch, we split the branch at the intersection point
        else:
            end_center_line, end_center_radius, end_contour_point = (
                graph_branches.split_branch(idx_cb, idx_cyl, parent_node)
            )
    else:
        parent_node = None
        graph_branches.nodes.append(starting_point)
        end_center_line = np.empty((0, 3))
        end_center_radius = []
        end_contour_point = []

    direction_point = direction_point - starting_point

    init_radius = starting_radius

    # Tracking configuration
    pct_inl = pct_inlier_points / 100
    err = threshold / 100
    cfg = config(percent_inliers=pct_inl, threshold=err)

    # Initialize tracking
    cyl = cylinder(starting_point, init_radius, direction_point, height=0)

    # Perform tracking
    centerline, contour_points, centerline_radius, cylinders = track_branch(
        vol,
        cyl,
        cfg,
        end_center_line,
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
