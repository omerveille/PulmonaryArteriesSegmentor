import networkx as nx
import slicer

class Graph_branches():
    def __init__(self) -> None:
        self.branch_graph = nx.Graph()
        self.branch_list = []
        self.branch_total = []


    def createNewMarkups(self, output_center_line, output_contour_points):
        new_output_center_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        slicer.util.updateMarkupsControlPointsFromArray(new_output_center_line, output_center_line)
        new_output_center_line.GetDisplayNode().SetTextScale(0)

        new_output_countour_points = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        slicer.util.updateMarkupsControlPointsFromArray(new_output_countour_points, output_contour_points)
        new_output_countour_points.GetDisplayNode().SetTextScale(0)
        new_output_countour_points.GetDisplayNode().SetVisibility(False)

        return new_output_center_line, new_output_countour_points
    

    def updateGraph(self, idx_cb, idx_cyl):
        centers_line = self.branch_graph.nodes[idx_cb]["centers_line"]
        contour_points = self.branch_graph.nodes[idx_cb]["contour_points"]
        centers2contour = self.branch_graph.nodes[idx_cb]["centers2contour"]

        self.branch_graph.nodes[idx_cb]["centers_line"] = centers_line[:idx_cyl]
        self.branch_graph.nodes[idx_cb]["contour_points"] = contour_points[:centers2contour[idx_cyl]]
        self.branch_graph.nodes[idx_cb]["centers2contour"] = centers2contour[:idx_cyl]

        slicer.util.updateMarkupsControlPointsFromArray(self.branch_graph.nodes[idx_cb]["centers_line_markups"], self.branch_graph.nodes[idx_cb]["centers_line"])
        slicer.util.updateMarkupsControlPointsFromArray(self.branch_graph.nodes[idx_cb]["contour_points_markups"], self.branch_graph.nodes[idx_cb]["contour_points"])
        new_centers_line_markups, new_contour_points_markups = self.createNewMarkups(centers_line[max(0,idx_cyl-1):], contour_points[centers2contour[idx_cyl]:])
        self.branch_graph.add_node(self.branch_graph.number_of_nodes(), name="n"+str(self.branch_graph.number_of_nodes()), centers_line=centers_line[max(0,idx_cyl-1):], contour_points=contour_points[centers2contour[idx_cyl]:], centers2contour=centers2contour[idx_cyl:], centers_line_markups=new_centers_line_markups, contour_points_markups=new_contour_points_markups)
        self.branch_graph.add_edge(idx_cb, self.branch_graph.number_of_nodes()-1)

        return centers_line[max(0,idx_cyl-1):idx_cyl]