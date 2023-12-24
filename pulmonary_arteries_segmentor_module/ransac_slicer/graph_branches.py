import networkx as nx
import numpy as np
import slicer

class Graph_branches():
    def __init__(self) -> None:
        self.branch_list = []            # list of shape (n,m) with n = number of branches and m = number of cylinder in the current branch
        self.nodes = []                  # list of nodes which are the birfucation + root + leafs
        self.edges = []                  # list of tuple for edges between nodes
        self.names = []                  # list of names in each edges
        self.centers_lines = []          # list of shape (n,m,3) with n = number of branches and m = number of points in the current center line
        self.contours_points = []        # list of shape (n,m,l,3) with n = number of branches, m = number of points in the current center line and l = number of points in the current contour
        self.centers_line_markups = []   # list of murkups for centers line
        self.contour_points_markups = [] # list of markups for contour points


    def createNewMarkups(self, centers_line, contour_points):
        new_center_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        slicer.util.updateMarkupsControlPointsFromArray(new_center_line, centers_line)
        new_center_line.GetDisplayNode().SetTextScale(0)

        new_countour_points = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        slicer.util.updateMarkupsControlPointsFromArray(new_countour_points, np.array([elt for pts in contour_points for elt in pts]))
        new_countour_points.GetDisplayNode().SetTextScale(0)
        new_countour_points.GetDisplayNode().SetVisibility(False)

        self.centers_line_markups.append(new_center_line)
        self.contour_points_markups.append(new_countour_points)
    

    def createNewBranch(self, edge, centers_line, contour_points):
        self.edges.append(edge)
        self.names.append("b"+str(len(self.edges)))
        self.centers_lines.append(centers_line)
        self.contours_points.append(contour_points)
        self.createNewMarkups(centers_line, contour_points)


    def updateGraph(self, idx_cb, idx_cyl):
        centers_line = self.centers_lines[idx_cb]
        contour_points = self.contours_points[idx_cb]

        # Modify old branch which became a parent   
        self.centers_lines[idx_cb] = centers_line[:min(idx_cyl+1, len(centers_line)-1)]
        self.contours_points[idx_cb] = contour_points[:min(idx_cyl+1, len(centers_line)-1)]
        slicer.util.updateMarkupsControlPointsFromArray(self.centers_line_markups[idx_cb], self.centers_lines[idx_cb])
        slicer.util.updateMarkupsControlPointsFromArray(self.contour_points_markups[idx_cb], np.array([elt for pts in self.contours_points[idx_cb] for elt in pts]))

        # Update edges
        self.nodes.append(centers_line[idx_cyl])
        old_end = self.edges[idx_cb][1]
        self.edges[idx_cb] = (self.edges[idx_cb][0], len(self.nodes)-1)

        # Create new branch from the old one but as a child
        self.createNewBranch((len(self.nodes)-1, old_end), centers_line[idx_cyl:], contour_points[min(idx_cyl+1, len(centers_line)-1):])

        return centers_line[idx_cyl:min(idx_cyl+1, len(centers_line)-1)]
    

    def createNetworkX(self):
        # Create graph Network X with node = bifurcation and edges = branches
        branch_graph = nx.DiGraph()

        for i, n in enumerate(self.nodes):
            branch_graph.add_node(i, pos=n)
        for i, e in enumerate(self.edges):
            branch_graph.add_edge(e[0], e[1], name=self.names[i], centers_line=self.centers_lines[i], contour_points=self.contours_points[i])

        return branch_graph
