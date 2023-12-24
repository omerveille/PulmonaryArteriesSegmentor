#!/usr/bin/env python-real
from .io import (copy_curve)
from .cylinder_ransac import (track_branch, config)
from .cylinder import cylinder, closest_branch
from .volume import volume
import numpy as np
from ransac_slicer.graph_branches import *


def run_ransac(input_volume_path, starting_point, direction_point, starting_radius, pct_inlier_points,
               threshold, graph_branches: Graph_branches, isNewBranch):
    # Input volume
    vol = volume.from_nrrd(input_volume_path)

    # Input info for branch tracking (in RAS coordinates)
    if isNewBranch:
        _, cb, idx_cb, idx_cyl = closest_branch(direction_point, graph_branches.branch_list)
        starting_point = cb[idx_cyl].center

        # Update Graph
        end_center_line = graph_branches.updateGraph(idx_cb, idx_cyl)
    else:
        graph_branches.nodes.append(starting_point)
        end_center_line = np.empty((0,3))
        
    direction_point = direction_point - starting_point

    init_radius = starting_radius

    # Tracking configuration
    pct_inl = pct_inlier_points / 100
    err = threshold / 100
    cfg = config(percent_inliers=pct_inl, threshold=err)

    # Initialize tracking
    cyl = cylinder(starting_point, init_radius, direction_point, height=0)

    # Perform tracking
    centers_line, contour_points = track_branch(vol, cyl, cfg, end_center_line, [elt for branch in graph_branches.branch_list for elt in branch])

    new_branch_list = []
    for cp in centers_line:
        new_branch_list.append(cylinder(center=np.array(cp)))
    graph_branches.branch_list.append(new_branch_list)
    graph_branches.nodes.append(centers_line[-1])
    graph_branches.createNewBranch((len(graph_branches.nodes) - 2, len(graph_branches.nodes) - 1), centers_line, contour_points)
    
    return graph_branches
