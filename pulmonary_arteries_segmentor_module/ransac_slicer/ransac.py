#!/usr/bin/env python-real
from .cylinder_ransac import (track_branch, config)
from .cylinder import cylinder, closest_branch
import numpy as np
from ransac_slicer.graph_branches import GraphBranches
import qt


def run_ransac(vol, starting_point, direction_point, starting_radius, pct_inlier_points,
               threshold, graph_branches: GraphBranches, isNewBranch, progress_dialog):
    # Input info for branch tracking (in RAS coordinates)
    if isNewBranch:
        _, _, idx_cb, idx_cyl = closest_branch(starting_point, graph_branches.branch_list)
        if idx_cyl == len(graph_branches.centers_lines[idx_cb]) - 2:
            idx_cyl = len(graph_branches.centers_lines[idx_cb]) - 1

        # Update Graph
        parent_node = graph_branches.names[idx_cb]
        # Case when the closest node is the last point of a branch, we concatenate the two branches
        if idx_cyl == len(graph_branches.centers_lines[idx_cb]) - 1:
            end_center_line, end_center_radius, end_contour_point = graph_branches.centers_lines[idx_cb][idx_cyl:idx_cyl+1], graph_branches.centers_line_radius[idx_cb][idx_cyl:idx_cyl+1], graph_branches.contours_points[idx_cb][idx_cyl:idx_cyl+1]
        # Case when the closest node is the first point of a branch, we had the branch to the parent of the closest branch, thus the branch way have more than 2 childs
        elif idx_cyl == 0:
            parent_node = graph_branches.tree_widget.getParentNodeId(parent_node)
            end_center_line, end_center_radius, end_contour_point = graph_branches.centers_lines[idx_cb][:1], graph_branches.centers_line_radius[idx_cb][:1], graph_branches.contours_points[idx_cb][:1]
        # Case when the closest node is in the middle of a branch, we split the branch at the intersection point
        else:
            end_center_line, end_center_radius, end_contour_point = graph_branches.split_branch(idx_cb, idx_cyl, parent_node)
    else:
        parent_node = None
        graph_branches.nodes.append(starting_point)
        end_center_line = np.empty((0,3))
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
    centers_line, contour_points, center_line_radius = track_branch(vol, cyl, cfg, end_center_line, end_center_radius, end_contour_point, [elt for branch in graph_branches.branch_list for elt in branch], progress_dialog)

    if len(centers_line) <= 1:
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText("Could not find any branch")
        msg.exec_()
        graph_branches.on_merge_only_child(parent_node)
    else:
        graph_branches.nodes.append(centers_line[-1])
        edge_begin = graph_branches.edges[graph_branches.names.index(parent_node)][1] if isNewBranch else len(graph_branches.nodes) - 2
        graph_branches.create_new_branch((edge_begin, len(graph_branches.nodes) - 1), centers_line, contour_points, center_line_radius, parent_node)

    return graph_branches
