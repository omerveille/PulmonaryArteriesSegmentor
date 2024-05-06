import networkx as nx
from networkx.readwrite import json_graph
import pickle
import json
import numpy as np
import slicer
import qt
import os
from .cylinder import cylinder
from .branch_tree import BranchTree, TreeColumnRole, Icons
from .color_palettes import centerline_color, contour_points_color

class GraphBranches():
    def __init__(self, tree_widget: BranchTree, centerline_button, contour_point_button, lock_button) -> None:
        self.branch_list = []            # list of shape (n,m) with n = number of branches and m = number of cylinder in the current branch
        self.nodes = []                  # list of nodes which are the birfucation + root + leafs
        self.edges = []                  # list of tuple for edges between nodes
        self.names = []                  # list of names in each edges
        self.centers_lines : list[np.ndarray] = []          # list of shape (n,m,3) with n = number of branches and m = number of points in the current center line
        self.contours_points : list[list] = []        # list of shape (n,m,l,3) with n = number of branches, m = number of points in the current center line and l = number of points in the current contour
        self.centers_line_radius = []    # list of shape (n,m) with n = number of branches and m = the radius of each points of the center line
        self.centers_line_markups = []   # list of markups for centers line
        self.contour_points_markups = [] # list of markups for contour points

        self.tree_widget = tree_widget
        self.centerline_button = centerline_button
        self.contour_point_button = contour_point_button
        self.lock_button = lock_button

        self.current_tree_item = None
        self.tree_widget.connect("itemClicked(QTreeWidgetItem *, int)", self.on_item_clicked)
        self.tree_widget.itemRenamed.connect(self.on_item_renamed)
        self.tree_widget.itemRemoveEnd.connect(self.on_remove_end)
        self.tree_widget.itemDeleted.connect(self.on_delete_item)
        self.tree_widget.keyPressed.connect(self.on_key_pressed)
        self.tree_widget.headerClicked.connect(self.on_header_clicked)

        self.node_selected = (-1, -1)


    def create_new_markups(self, name, centers_line, contour_points):
        new_center_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        slicer.util.updateMarkupsControlPointsFromArray(new_center_line, centers_line)

        new_center_line.SetName(name+"_centers")
        new_center_line.AddObserver(slicer.vtkMRMLMarkupsNode.PointStartInteractionEvent , self.on_node_clicked)
        new_center_line.GetDisplayNode().SetSelectedColor(*centerline_color)

        new_contour_points = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        slicer.util.updateMarkupsControlPointsFromArray(new_contour_points, np.array([elt for pts in contour_points for elt in pts]))
        new_contour_points.GetDisplayNode().SetTextScale(0)
        new_contour_points.GetDisplayNode().SetVisibility(False)
        new_contour_points.GetDisplayNode().SetSelectedColor(*contour_points_color)
        new_contour_points.SetName(name+"_contours")

        self.centers_line_markups.append(new_center_line)
        self.contour_points_markups.append(new_contour_points)

        if self.lock_button.checked:
            new_center_line.LockedOn()
            new_contour_points.LockedOn()

        self.update_visibility_button(TreeColumnRole.VISIBILITY_CENTER)
        self.update_visibility_button(TreeColumnRole.VISIBILITY_CONTOUR)


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

        self.tree_widget.insertAfterNode(nodeId=new_name, parentNodeId=parent_node, becomeIntermediaryParent=isFromUpdate)

        if not isFromUpdate:
            self.on_merge_only_child(parent_node)


    def update_parent_branch(self, branch_id, node_id):
        self.branch_list[branch_id] = self.branch_list[branch_id][:node_id]
        self.centers_lines[branch_id] = self.centers_lines[branch_id][:node_id]
        self.contours_points[branch_id] = self.contours_points[branch_id][:node_id]
        self.centers_line_radius[branch_id] = self.centers_line_radius[branch_id][:node_id]

        slicer.util.updateMarkupsControlPointsFromArray(self.centers_line_markups[branch_id], self.centers_lines[branch_id])
        slicer.util.updateMarkupsControlPointsFromArray(self.contour_points_markups[branch_id], np.array([elt for pts in self.contours_points[branch_id] for elt in pts]))

    def update_visibility_button(self, column):
        markup_list, button = (self.centers_line_markups, self.centerline_button) if column == TreeColumnRole.VISIBILITY_CENTER else (self.contour_points_markups, self.contour_point_button)
        majority_visibility = not (np.sum([markup.GetDisplayNode().GetVisibility() for markup in markup_list]) >= max(1, (len(markup_list) // 2)))
        button.text = button.text.replace("Hide", "Show") if majority_visibility else button.text.replace("Show", "Hide")




    def split_branch(self, idx_cb, idx_cyl, parent_node):
        # Modify old branch which became a parent
        centers_line = self.centers_lines[idx_cb]
        contour_points = self.contours_points[idx_cb]
        centers_line_radius = self.centers_line_radius[idx_cb]
        self.update_parent_branch(idx_cb, idx_cyl+1)

        # Update edges
        self.nodes.append(centers_line[idx_cyl])
        old_end = self.edges[idx_cb][1]
        self.edges[idx_cb] = (self.edges[idx_cb][0], len(self.nodes)-1)

        # Create new branch from the old one but as a child
        self.create_new_branch((len(self.nodes)-1, old_end), centers_line[idx_cyl:], contour_points[idx_cyl:], centers_line_radius[idx_cyl:], parent_node, True)

        return centers_line[idx_cyl:idx_cyl+1], centers_line_radius[idx_cyl:idx_cyl+1], contour_points[idx_cyl:idx_cyl+1]


    def save_networkX(self):
        # Create graph Network X with node = bifurcation and edges = branches
        branch_graph = nx.DiGraph()

        for i, n in enumerate(self.nodes):
            branch_graph.add_node(i, pos=n)
        for i, e in enumerate(self.edges):
            branch_graph.add_edge(e[0], e[1], name=self.names[i], center_line=self.centers_lines[i], contour_points=self.contours_points[i])

        dialog = qt.QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Choose a folder")

        # cancel any action if the user cancel / close the window / press escape
        if not folder_path:
            return

        # save with pickle
        with open(os.path.join(folder_path, "graph_tree.pickle"), 'wb') as f:
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
        with open(os.path.join(folder_path, "graph_tree.json"), "w") as outfile:
            json.dump(data_list, outfile, indent=4)

        slicer.util.infoDisplay(f"The graph has been successfully exported to :\n{folder_path}", windowTitle="Success")


    def clear_all(self) -> bool:
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Warning)
        msg.setWindowTitle("Confirmation")
        msg.setText("Are you sure you want to clear the tree ?")
        msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)

        if msg.exec_() != qt.QMessageBox.Yes:
            return False

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
        self.update_visibility_button(TreeColumnRole.VISIBILITY_CENTER)
        self.update_visibility_button(TreeColumnRole.VISIBILITY_CONTOUR)
        return True


    def on_stop_interaction(self):
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
            self.update_visibility_button(column)
        elif column == TreeColumnRole.VISIBILITY_CONTOUR:
            is_visible = self.contour_points_markups[branch_id].GetDisplayNode().GetVisibility()
            self.contour_points_markups[branch_id].GetDisplayNode().SetVisibility(not is_visible)
            self.tree_widget._branchDict[node_id].setIcon(TreeColumnRole.VISIBILITY_CONTOUR, Icons.visibleOff if is_visible else Icons.visibleOn)
            self.update_visibility_button(column)
        elif column == TreeColumnRole.DELETE:
            self.on_delete_item(treeItem)
            self.update_visibility_button(TreeColumnRole.VISIBILITY_CENTER)
            self.update_visibility_button(TreeColumnRole.VISIBILITY_CONTOUR)

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

    def on_header_clicked(self, column):
        def change_majority_visibility(markup_list, column, button):
            majority_visibility = not (np.sum([markup.GetDisplayNode().GetVisibility() for markup in markup_list]) >= max(1, (len(markup_list) // 2)))
            icon = Icons.visibleOn if majority_visibility else Icons.visibleOff
            for markup in markup_list:
                markup.GetDisplayNode().SetVisibility(majority_visibility)
            for branch in self.tree_widget._branchDict.values():
                branch.setIcon(column, icon)
            self.update_visibility_button(column)


        if column == TreeColumnRole.VISIBILITY_CENTER:
            change_majority_visibility(self.centers_line_markups, column, self.centerline_button)
        elif column == TreeColumnRole.VISIBILITY_CONTOUR:
            change_majority_visibility(self.contour_points_markups, column, self.contour_point_button)

    def on_node_clicked(self, caller, event):
        displayNode = caller.GetDisplayNode()
        if displayNode.GetActiveComponentType() == slicer.vtkMRMLMarkupsDisplayNode.ComponentControlPoint:
            node_id = displayNode.GetActiveComponentIndex()

            branch_name = "_".join(caller.GetName().split('_')[:-1])
            branch_id = self.names.index(branch_name)
            self.node_selected = (branch_id, node_id)

            tree_item = self.tree_widget.getTreeWidgetItem(branch_name)
            self.tree_widget.scrollToItem(tree_item)
            self.tree_widget.setCurrentItem(tree_item)

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
        self.update_parent_branch(branch_id, branch_node_id+1)

    def delete_node(self, index):
        self.nodes.pop(index)
        for i in range(len(self.edges)):
            n1, n2 = self.edges[i]
            if n1 > index:
                n1 -= 1
            if n2 > index:
                n2 -= 1
            self.edges[i] = n1, n2

    def on_delete_item(self, treeItem, showPopupForNonLeaf=True):
        """
    Remove the item from the tree and hide the associated markup
    """
        self.on_stop_interaction()
        node_id = treeItem.nodeId

        if self.tree_widget.isRoot(node_id):
            slicer.util.errorDisplay(text="You can't delete the root", windowTitle="Error")
            return

        children = [self.tree_widget.getTreeWidgetItem(n_id) for n_id in self.tree_widget.getChildrenNodeId(node_id)]

        if len(children) != 0 and showPopupForNonLeaf:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setWindowTitle("Confirmation")
            msg.setText(f"Are you sure you want to delete {node_id} and all its children ?")
            msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
            if msg.exec_() != qt.QMessageBox.Yes:
                return

        for child in children:
            self.on_delete_item(child, showPopupForNonLeaf=False)

        branch_id = self.names.index(node_id)
        self.delete_node(self.edges[branch_id][1])

        self.names.pop(branch_id)
        self.branch_list.pop(branch_id)
        self.centers_lines.pop(branch_id)
        self.contours_points.pop(branch_id)
        self.centers_line_radius.pop(branch_id)
        slicer.mrmlScene.RemoveNode(self.centers_line_markups.pop(branch_id))
        slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop(branch_id))

        self.edges.pop(branch_id)
        parent_id = self.tree_widget.getParentNodeId(node_id)
        self.tree_widget.removeNode(node_id)

        if self.current_tree_item == treeItem:
            self.current_tree_item = None

        if showPopupForNonLeaf:
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
        self.contours_points[parent_idx] += self.contours_points[child_idx][1:]
        self.contours_points.pop(child_idx)
        self.centers_line_radius[parent_idx] += self.centers_line_radius[child_idx][1:]
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