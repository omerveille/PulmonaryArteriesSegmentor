#!/usr/bin/env python-real
from .io import (copy_curve)
from .cylinder_ransac import (track_branch, config)
from .cylinder import cylinder, closest_branch
from .volume import volume
import numpy as np
import slicer
import networkx as nx


def createNewMarkups(output_center_line, output_contour_points):
    new_output_center_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
    slicer.util.updateMarkupsControlPointsFromArray(new_output_center_line, output_center_line)
    new_output_center_line.GetDisplayNode().SetTextScale(0)

    new_output_countour_points = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    slicer.util.updateMarkupsControlPointsFromArray(new_output_countour_points, output_contour_points)
    new_output_countour_points.GetDisplayNode().SetTextScale(0)
    new_output_countour_points.GetDisplayNode().SetVisibility(False)

    return new_output_center_line, new_output_countour_points


def run_ransac(input_volume_path, starting_point, direction_point, starting_radius, pct_inlier_points,
               threshold, isNewBranch, branch_list: list, branch_graph: nx.Graph):
    # Input volume
    vol = volume.from_nrrd(input_volume_path)

    branch = []
    # Input info for branch tracking (in RAS coordinates)
    if isNewBranch:
        _, cb, idx_cb, idx_cyl = closest_branch(direction_point, branch_list)
        branch = cb[:idx_cyl] # To check if we can't let everything in the branch
        starting_point = cb[idx_cyl].center

        # Update Graph
        centers_line = branch_graph.nodes[idx_cb]["centers_line"]
        contour_points = branch_graph.nodes[idx_cb]["contour_points"]
        centers2contour = branch_graph.nodes[idx_cb]["centers2contour"]
        branch_graph.nodes[idx_cb]["centers_line"] = centers_line[:idx_cyl]
        branch_graph.nodes[idx_cb]["contour_points"] = contour_points[:centers2contour[idx_cyl]]
        branch_graph.nodes[idx_cb]["centers2contour"] = centers2contour[:idx_cyl]
        slicer.util.updateMarkupsControlPointsFromArray(branch_graph.nodes[idx_cb]["centers_line_markups"], branch_graph.nodes[idx_cb]["centers_line"])
        slicer.util.updateMarkupsControlPointsFromArray(branch_graph.nodes[idx_cb]["contour_points_markups"], branch_graph.nodes[idx_cb]["contour_points"])
        new_centers_line_markups, new_contour_points_markups = createNewMarkups(centers_line[max(0,idx_cyl-1):], contour_points[centers2contour[idx_cyl]:])
        branch_graph.add_node(branch_graph.number_of_nodes(), name="n"+str(branch_graph.number_of_nodes()), centers_line=centers_line[max(0,idx_cyl-1):], contour_points=contour_points[centers2contour[idx_cyl]:], centers2contour=centers2contour[idx_cyl:], centers_line_markups=new_centers_line_markups, contour_points_markups=new_contour_points_markups)
        branch_graph.add_edge(idx_cb, branch_graph.number_of_nodes()-1)
        
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
        centers_line, contour_points, centers2contour = track_branch(vol, cyl, cfg, centers_line[max(0,idx_cyl-1):idx_cyl], branch)
    else:
        centers_line, contour_points, centers2contour = track_branch(vol, cyl, cfg, np.empty((0,3)), branch)

    new_branch_list = []
    for cp in centers_line:
        new_branch_list.append(cylinder(center=np.array(cp)))
    branch_list.append(new_branch_list)

    new_centers_line_markups, new_contour_points_markups = createNewMarkups(centers_line, contour_points)
    branch_graph.add_node(branch_graph.number_of_nodes(), name="n"+str(branch_graph.number_of_nodes()), centers_line=centers_line, contour_points=contour_points, centers2contour=centers2contour, centers_line_markups=new_centers_line_markups, contour_points_markups=new_contour_points_markups)

    if isNewBranch:
        branch_graph.add_edge(idx_cb, branch_graph.number_of_nodes()-1)
    
    return branch_list, branch_graph
