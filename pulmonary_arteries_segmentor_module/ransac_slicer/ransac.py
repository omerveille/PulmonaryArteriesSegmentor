#!/usr/bin/env python-real
from .cylinder_ransac import (track_branch, config)
from .cylinder import cylinder, closest_branch
import numpy as np
from ransac_slicer.graph_branches import GraphBranches


def run_ransac(vol, starting_point, direction_point, starting_radius, pct_inlier_points,
               threshold, graph_branches: GraphBranches, isNewBranch):
    # Input info for branch tracking (in RAS coordinates)
    if isNewBranch:
        _, cb, idx_cb, idx_cyl = closest_branch(direction_point, graph_branches.branch_list)
        starting_point = cb[idx_cyl].center

        # Update Graph
        parent_node = graph_branches.names[idx_cb]
        end_center_line, end_center_radius = graph_branches.update_graph(idx_cb, idx_cyl, parent_node)
    else:
        parent_node = None
        graph_branches.nodes.append(starting_point)
        end_center_line = np.empty((0,3))
        end_center_radius = []

    direction_point = direction_point - starting_point

    init_radius = starting_radius

    # Tracking configuration
    pct_inl = pct_inlier_points / 100
    err = threshold / 100
    cfg = config(percent_inliers=pct_inl, threshold=err)

    # Initialize tracking
    cyl = cylinder(starting_point, init_radius, direction_point, height=0)

    # Perform tracking
    centers_line, contour_points, center_line_radius = track_branch(vol, cyl, cfg, end_center_line, end_center_radius, [elt for branch in graph_branches.branch_list for elt in branch])

    graph_branches.nodes.append(centers_line[-1])
    graph_branches.create_new_branch((len(graph_branches.nodes) - 2, len(graph_branches.nodes) - 1), centers_line, contour_points, center_line_radius, parent_node)

    return graph_branches
