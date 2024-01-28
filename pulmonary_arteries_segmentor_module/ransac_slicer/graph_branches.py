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
        self.centers_line_radius = []    # list of shape (n,m) with n = number of branches and m = the radius of each points of the center line
        self.centers_line_markups = []   # list of markups for centers line
        self.contour_points_markups = [] # list of markups for contour points

        self.tree_widget = tree_widget
        self.current_tree_item = None
        self.tree_widget.connect("itemClicked(QTreeWidgetItem *, int)", self.on_item_clicked)
        self.tree_widget.itemRenamed.connect(self.on_item_renamed)
        self.tree_widget.itemRemoveEnd.connect(self.on_remove_end)
        self.tree_widget.itemDeleted.connect(self.on_delete_item)
        self.tree_widget.keyPressed.connect(self.on_key_pressed)

        self.node_selected = (-1, -1)


    def create_new_markups(self, name, centers_line, contour_points):
        new_center_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        slicer.util.updateMarkupsControlPointsFromArray(new_center_line, centers_line)
        # new_center_line.GetDisplayNode().SetTextScale(0)
        new_center_line.SetName(name+"_centers")
        new_center_line.AddObserver(slicer.vtkMRMLMarkupsNode.PointStartInteractionEvent , self.on_node_clicked)

        new_contour_points = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        slicer.util.updateMarkupsControlPointsFromArray(new_contour_points, np.array([elt for pts in contour_points for elt in pts]))
        new_contour_points.GetDisplayNode().SetTextScale(0)
        new_contour_points.GetDisplayNode().SetVisibility(False)
        new_contour_points.SetName(name+"_contours")

        self.centers_line_markups.append(new_center_line)
        self.contour_points_markups.append(new_contour_points)


    def create_new_branch(self, edge, centers_line, contour_points, center_line_radius, parent_node=None, isFromUpdate=False):
        new_branch_list = []
        for cp in centers_line:
            new_branch_list.append(cylinder(center=np.array(cp)))
        self.branch_list.append(new_branch_list)

        self.edges.append(edge)
        new_name = "b"+str(len(self.edges))

        self.names.append(new_name)
        self.centers_lines.append(centers_line)
        self.contours_points.append(contour_points)
        self.centers_line_radius.append(center_line_radius)

        self.create_new_markups(new_name, centers_line, contour_points)
        self.tree_widget.insertAfterNode(nodeId=new_name, parentNodeId=parent_node)

        if not isFromUpdate:
            self.on_merge_only_child(parent_node)


    def update_graph(self, branch_id, node_id):
        self.branch_list[branch_id] = self.branch_list[branch_id][:node_id]
        self.centers_lines[branch_id] = self.centers_lines[branch_id][:node_id]
        self.contours_points[branch_id] = self.contours_points[branch_id][:node_id]
        self.centers_line_radius[branch_id] = self.centers_line_radius[branch_id][:node_id]

        slicer.util.updateMarkupsControlPointsFromArray(self.centers_line_markups[branch_id], self.centers_lines[branch_id])
        slicer.util.updateMarkupsControlPointsFromArray(self.contour_points_markups[branch_id], np.array([elt for pts in self.contours_points[branch_id] for elt in pts]))


    def split_branch(self, idx_cb, idx_cyl, parent_node):
        # Modify old branch which became a parent
        centers_line = self.centers_lines[idx_cb]
        contour_points = self.contours_points[idx_cb]
        centers_line_radius = self.centers_line_radius[idx_cb]
        self.update_graph(idx_cb, idx_cyl+1)

        # Update edges
        self.nodes.append(centers_line[idx_cyl])
        old_end = self.edges[idx_cb][1]
        self.edges[idx_cb] = (self.edges[idx_cb][0], len(self.nodes)-1)

        # Create new branch from the old one but as a child
        self.create_new_branch((len(self.nodes)-1, old_end), centers_line[idx_cyl:], contour_points[idx_cyl+1:], centers_line_radius[idx_cyl:], parent_node, True)

        return centers_line[idx_cyl:idx_cyl+1], centers_line_radius[idx_cyl:idx_cyl+1]


    def save_networkX(self):
        # Create graph Network X with node = bifurcation and edges = branches
        branch_graph = nx.DiGraph()

        for i, n in enumerate(self.nodes):
            branch_graph.add_node(i, pos=n)
        for i, e in enumerate(self.edges):
            branch_graph.add_edge(e[0], e[1], name=self.names[i], centers_line=self.centers_lines[i], contour_points=self.contours_points[i])


        dialog = qt.QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Choose a folder")
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
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Warning)
        msg.setWindowTitle("Confirmation")
        msg.setText("Are you sure you want to clear the tree ?")
        msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)

        if msg.exec_() != qt.QMessageBox.Yes:
            return

        self.branch_list = []
        self.nodes = []
        self.edges = []
        self.names = []
        self.centers_lines = []
        self.contours_points = []
        self.centers_line_radius = []

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

    def on_node_clicked(self, caller, event):
        displayNode = caller.GetDisplayNode()
        if displayNode.GetActiveComponentType() == slicer.vtkMRMLMarkupsDisplayNode.ComponentControlPoint:
            node_id = displayNode.GetActiveComponentIndex()
            branch_name = caller.GetName()
            branch_id = self.names.index(branch_name.split('_')[0])
            self.node_selected = (branch_id, node_id)

    def on_remove_end(self, treeItem):
        node_id = treeItem.nodeId
        branch_id = self.names.index(node_id)
        branch_selected, branch_node_id = self.node_selected
        if branch_id != branch_selected:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("No node selected or not in the good branch")
            msg.exec_()
            return

        # Nothing to delete
        if branch_node_id == len(self.centers_lines[branch_id]) - 1:
            return

        edges_node_id = self.edges[branch_id][1]
        self.nodes[edges_node_id] = self.centers_lines[branch_id][branch_node_id]
        self.update_graph(branch_id, branch_node_id+1)

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
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("You can't delete the root")
            msg.exec_()
            return
        branch_id = self.names.index(node_id)
        self.names.pop(branch_id)
        self.branch_list.pop(branch_id)
        self.centers_lines.pop(branch_id)
        self.contours_points.pop(branch_id)
        self.centers_line_radius.pop(branch_id)
        slicer.mrmlScene.RemoveNode(self.centers_line_markups.pop(branch_id))
        slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop(branch_id))

        if self.tree_widget.isLeaf(node_id):
            self.delete_node(self.edges[branch_id][1])
        self.edges.pop(branch_id)

        parent_id = self.tree_widget.getParentNodeId(node_id)
        self.tree_widget.removeNode(node_id)

        if self.current_tree_item == treeItem:
            self.current_tree_item = None

        self.on_merge_only_child(parent_id)

    def on_merge_only_child(self, node_id):
        if node_id is None:
            return
        child_list = self.tree_widget.getChildrenNodeId(node_id)
        if len(child_list) != 1:
            return

        parent_idx = self.names.index(node_id)
        child_idx = self.names.index(child_list[0])

        # Modify parent branch to add child branch
        self.centers_lines[parent_idx] = np.vstack((self.centers_lines[parent_idx], self.centers_lines[child_idx][1:]))
        self.centers_lines.pop(child_idx)
        self.contours_points[parent_idx] += self.contours_points[child_idx]
        self.contours_points.pop(child_idx)
        self.centers_line_radius[parent_idx] += self.centers_line_radius[child_idx]
        self.centers_line_radius.pop(child_idx)
        slicer.util.updateMarkupsControlPointsFromArray(self.centers_line_markups[parent_idx], self.centers_lines[parent_idx])
        slicer.util.updateMarkupsControlPointsFromArray(self.contour_points_markups[parent_idx], np.array([elt for pts in self.contours_points[parent_idx] for elt in pts]))
        slicer.mrmlScene.RemoveNode(self.centers_line_markups.pop(child_idx))
        slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop(child_idx))
        self.branch_list[parent_idx] += self.branch_list[child_idx]
        self.branch_list.pop(child_idx)

        # Delete old child
        self.delete_node(self.edges[child_idx][0])
        self.edges[parent_idx] = self.edges[parent_idx][0], self.edges[child_idx][1]
        self.edges.pop(child_idx)
        self.names.pop(child_idx)

        self.tree_widget.removeNode(child_list[0])