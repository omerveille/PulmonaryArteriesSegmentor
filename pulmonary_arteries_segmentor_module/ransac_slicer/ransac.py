#!/usr/bin/env python-real
from .io import (copy_curve)
from .cylinder_ransac import (track_branch, config)
from .cylinder import cylinder, closest_branch
from .volume import volume
import numpy as np
from ransac_slicer.graph_branches import *


def run_ransac(input_volume_path, starting_point, direction_point, starting_radius, pct_inlier_points,
               threshold, graph_branches, isNewBranch):
    # Input volume
    vol = volume.from_nrrd(input_volume_path)

    # Input info for branch tracking (in RAS coordinates)
    if isNewBranch:
        _, cb, idx_cb, idx_cyl = closest_branch(direction_point, graph_branches.branch_list)
        starting_point = cb[idx_cyl].center

        # Update Graph
        end_center_line = graph_branches.updateGraph(idx_cb, idx_cyl)
        
    direction_point = direction_point - starting_point

    init_radius = starting_radius

    # Tracking configuration
    pct_inl = pct_inlier_points / 100
    err = threshold / 100
    cfg = config(percent_inliers=pct_inl, threshold=err)

    # Initialize tracking
    cyl = cylinder(starting_point, init_radius, direction_point, height=0)

    # Perform tracking
    if isNewBranch:
        centers_line, contour_points, centers2contour = track_branch(vol, cyl, cfg, end_center_line, graph_branches.branch_total)
    else:
        centers_line, contour_points, centers2contour = track_branch(vol, cyl, cfg, np.empty((0,3)), graph_branches.branch_total)

    new_branch_list = []
    for cp in centers_line:
        new_branch_list.append(cylinder(center=np.array(cp)))
    graph_branches.branch_list.append(new_branch_list)
    graph_branches.branch_total += new_branch_list

    new_centers_line_markups, new_contour_points_markups = graph_branches.createNewMarkups(centers_line, contour_points)
    graph_branches.branch_graph.add_node(graph_branches.branch_graph.number_of_nodes(), name="n"+str(graph_branches.branch_graph.number_of_nodes()), centers_line=centers_line, contour_points=contour_points, centers2contour=centers2contour, centers_line_markups=new_centers_line_markups, contour_points_markups=new_contour_points_markups)

    if isNewBranch:
        graph_branches.branch_graph.add_edge(idx_cb, graph_branches.branch_graph.number_of_nodes()-1)
    
    return graph_branches
