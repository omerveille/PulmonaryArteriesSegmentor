import networkx as nx
from networkx.readwrite import json_graph
import pickle
import json
import numpy as np
import slicer
import qt
from .cylinder import cylinder
from .branch_tree import BranchTree, TreeColumnRole, Icons

class GraphBranches():
    def __init__(self, tree_widget: BranchTree) -> None:
        self.branch_list = []            # list of shape (n,m) with n = number of branches and m = number of cylinder in the current branch
        self.nodes = []                  # list of nodes which are the birfucation + root + leafs
        self.edges = []                  # list of tuple for edges between nodes
        self.names = []                  # list of names in each edges
        self.centers_lines = []          # list of shape (n,m,3) with n = number of branches and m = number of points in the current center line
        self.contours_points = []        # list of shape (n,m,l,3) with n = number of branches, m = number of points in the current center line and l = number of points in the current contour
        self.centers_line_markups = []   # list of markups for centers line
        self.contour_points_markups = [] # list of markups for contour points

        self.tree_widget = tree_widget
        self.current_tree_item = None
        self.tree_widget.connect("itemClicked(QTreeWidgetItem *, int)", self.on_item_clicked)
        self.tree_widget.itemRenamed.connect(self.on_item_renamed)
        self.tree_widget.itemDeleted.connect(self.on_delete_item)
        self.tree_widget.keyPressed.connect(self.on_key_pressed)


    def create_new_markups(self, name, centers_line, contour_points):
        new_center_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        slicer.util.updateMarkupsControlPointsFromArray(new_center_line, centers_line)
        # new_center_line.GetDisplayNode().SetTextScale(0)
        new_center_line.SetName(name+"_centers")

        new_contour_points = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        slicer.util.updateMarkupsControlPointsFromArray(new_contour_points, np.array([elt for pts in contour_points for elt in pts]))
        new_contour_points.GetDisplayNode().SetTextScale(0)
        new_contour_points.GetDisplayNode().SetVisibility(False)
        new_contour_points.SetName(name+"_contours")

        self.centers_line_markups.append(new_center_line)
        self.contour_points_markups.append(new_contour_points)


    def create_new_branch(self, edge, centers_line, contour_points, parent_node=None):
        new_branch_list = []
        for cp in centers_line:
            new_branch_list.append(cylinder(center=np.array(cp)))
        self.branch_list.append(new_branch_list)

        self.edges.append(edge)
        new_name = "b"+str(len(self.edges))
        self.names.append(new_name)
        self.centers_lines.append(centers_line)
        self.contours_points.append(contour_points)
        self.create_new_markups(new_name, centers_line, contour_points)
        self.tree_widget.insertAfterNode(nodeId=new_name, parentNodeId=parent_node)


    def update_graph(self, idx_cb, idx_cyl, parent_node):
        centers_line = self.centers_lines[idx_cb]
        contour_points = self.contours_points[idx_cb]
        branch_list = self.branch_list[idx_cb]

        # Modify old branch which became a parent
        self.centers_lines[idx_cb] = centers_line[:min(idx_cyl+1, len(centers_line)-1)]
        self.contours_points[idx_cb] = contour_points[:min(idx_cyl+1, len(centers_line)-1)]
        slicer.util.updateMarkupsControlPointsFromArray(self.centers_line_markups[idx_cb], self.centers_lines[idx_cb])
        slicer.util.updateMarkupsControlPointsFromArray(self.contour_points_markups[idx_cb], np.array([elt for pts in self.contours_points[idx_cb] for elt in pts]))
        self.branch_list[idx_cb] = branch_list[:min(idx_cyl+1, len(centers_line)-1)]

        # Update edges
        self.nodes.append(centers_line[idx_cyl])
        old_end = self.edges[idx_cb][1]
        self.edges[idx_cb] = (self.edges[idx_cb][0], len(self.nodes)-1)

        # Create new branch from the old one but as a child
        self.create_new_branch((len(self.nodes)-1, old_end), centers_line[idx_cyl:], contour_points[min(idx_cyl+1, len(centers_line)-1):], parent_node)

        return centers_line[idx_cyl:min(idx_cyl+1, len(centers_line)-1)]


    def save_networkX(self):
        # Create graph Network X with node = bifurcation and edges = branches
        branch_graph = nx.DiGraph()

        for i, n in enumerate(self.nodes):
            branch_graph.add_node(i, pos=n)
        for i, e in enumerate(self.edges):
            branch_graph.add_edge(e[0], e[1], name=self.names[i], centers_line=self.centers_lines[i], contour_points=self.contours_points[i])


        dialog = qt.QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Sélectionnez un dossier")
        # save with pickle
        with open(f'{folder_path}/graph_tree.pickle', 'wb') as f:
            pickle.dump(branch_graph, f, pickle.HIGHEST_PROTOCOL)

        def ndarray_to_list(data):
            if isinstance(data, np.ndarray):
                return data.tolist()
            if isinstance(data, list):
                return [ndarray_to_list(item) for item in data]
            elif isinstance(data, dict):
                return {k: ndarray_to_list(v) for k, v in data.items()}
            else:
                return data

        # save to json
        data = json_graph.node_link_data(branch_graph)
        data_list = ndarray_to_list(data)
        with open(f"{folder_path}/graph_tree.json", "w") as outfile:
            json.dump(data_list, outfile, indent=4)


        return branch_graph


    def clear_all(self):
        self.branch_list = []
        self.nodes = []
        self.edges = []
        self.names = []
        self.centers_lines = []
        self.contours_points = []

        size = len(self.centers_line_markups)
        for _ in range(size):
            slicer.mrmlScene.RemoveNode(self.centers_line_markups.pop())
            slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop())

        self.tree_widget.clear()


    def on_stop_interaction(self):
        self.tree_widget.editing_node = False
        if self.current_tree_item is not None:
            self.current_tree_item.updateText()

    def on_item_clicked(self, treeItem, column):
        """
    On item clicked, start placing item if necessary.
    Delete item if delete column was selected
    """
        self.current_tree_item = treeItem
        node_id = treeItem.nodeId
        branch_id = self.names.index(node_id)
        if column == TreeColumnRole.VISIBILITY_CENTER:
            is_visible = self.centers_line_markups[branch_id].GetDisplayNode().GetVisibility()
            self.centers_line_markups[branch_id].GetDisplayNode().SetVisibility(not is_visible)
            self.tree_widget._branchDict[node_id].setIcon(TreeColumnRole.VISIBILITY_CENTER, Icons.visibleOff if is_visible else Icons.visibleOn)
        if column == TreeColumnRole.VISIBILITY_CONTOUR:
            is_visible = self.contour_points_markups[branch_id].GetDisplayNode().GetVisibility()
            self.contour_points_markups[branch_id].GetDisplayNode().SetVisibility(not is_visible)
            self.tree_widget._branchDict[node_id].setIcon(TreeColumnRole.VISIBILITY_CONTOUR, Icons.visibleOff if is_visible else Icons.visibleOn)
        if column == TreeColumnRole.DELETE:
            self.on_delete_item(treeItem)

    def on_item_renamed(self, previous, new):
        branch_id = self.names.index(previous)
        self.names[branch_id] = new
        self.centers_line_markups[branch_id].SetName(new+"_centers")
        self.contour_points_markups[branch_id].SetName(new+"_contours")

    def on_key_pressed(self, treeItem, key):
        """
    On delete key pressed, delete the current item if any selected
    """
        if key == qt.Qt.Key_Delete:
            self.on_delete_item(treeItem)


    def delete_node(self, index):
        self.nodes.pop(index)
        for i in range(len(self.edges)):
            n1, n2 = self.edges[i]
            if n1 > index:
                n1 -= 1
            if n2 > index:
                n2 -= 1
            self.edges[i] = n1, n2

    def on_delete_item(self, treeItem):
        """
    Remove the item from the tree and hide the associated markup
    """
        self.on_stop_interaction()
        node_id = treeItem.nodeId
        if self.tree_widget.isRoot(node_id):
            return
        branch_id = self.names.index(node_id)
        self.names.pop(branch_id)
        self.branch_list.pop(branch_id)
        self.centers_lines.pop(branch_id)
        self.contours_points.pop(branch_id)
        slicer.mrmlScene.RemoveNode(self.centers_line_markups.pop(branch_id))
        slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop(branch_id))

        if self.tree_widget.isLeaf(node_id):
            self.delete_node(self.edges[branch_id][1])
        self.edges.pop(branch_id)

        self.tree_widget.removeNode(node_id)

        if self.current_tree_item == treeItem:
            self.current_tree_item = None