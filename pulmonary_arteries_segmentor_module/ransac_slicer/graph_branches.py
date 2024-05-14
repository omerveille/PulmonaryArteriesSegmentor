from typing import Union
import networkx as nx
from networkx.readwrite import json_graph
import pickle
import json
import numpy as np
from .popup_utils import CustomProgressBar
import slicer
import qt
import os
from .cylinder import cylinder
from .branch_tree import BranchTree, TreeColumnRole, Icons
from .color_palettes import centerline_color, contour_points_color


class GraphBranches:
    """
    Class which hold the graph of all the vessels segmented.

    The edges of the graph hold the points, and the node denotes bifurcation.

    Note:
    A branch can split in an infinite amount of childs.
    """

    def __init__(
        self,
        tree_widget: BranchTree,
        centerline_button,
        contour_point_button,
        lock_button,
    ) -> None:
        self.branch_list = []  # list of shape (n,m) with n = number of branches and m = number of cylinder in the current branch
        self.nodes = []  # list of nodes which are the birfucation + root + leafs
        self.edges = []  # list of tuple for edges between nodes
        self.names = []  # list of names in each edges
        self.centerlines: list[
            np.ndarray
        ] = []  # list of shape (n,m,3) with n = number of branches and m = number of points in the current center line
        self.contours_points: list[
            list
        ] = []  # list of shape (n,m,l,3) with n = number of branches, m = number of points in the current center line and l = number of points in the current contour
        self.centerline_radius = []  # list of shape (n,m) with n = number of branches and m = the radius of each points of the center line
        self.centerline_markups = []  # list of markups for centers line
        self.contour_points_markups = []  # list of markups for contour points

        self.tree_widget = tree_widget
        self.centerline_button = centerline_button
        self.contour_point_button = contour_point_button
        self.lock_button = lock_button
        self.centerline_text_size = 3.0

        self.current_tree_item = None
        self.tree_widget.connect(
            "itemClicked(QTreeWidgetItem *, int)", self.on_item_clicked
        )
        self.tree_widget.itemRenamed.connect(self.on_item_renamed)
        self.tree_widget.itemRemoveEnd.connect(self.on_remove_end)
        self.tree_widget.itemDeleted.connect(self.on_delete_item)
        self.tree_widget.keyPressed.connect(self.on_key_pressed)
        self.tree_widget.headerClicked.connect(self.on_header_clicked)

        self.node_selected = (-1, -1)

    def create_new_markups(
        self, name: str, centerline: np.ndarray, contour_points: list[list[np.ndarray]]
    ):
        """
        Create a new markup for the centerline and the associated contour points.

        Parameters
        ----------

        name: name of the branch.
        centerline: array containing the points of the centerline.
        contour_points: array of array of contour points, contour_points[0] are the points sampled around centerline[0],
        contour_points[1] are the points sampled around centerline[1] etc...
        """
        centerline_markup = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode"
        )
        slicer.util.updateMarkupsControlPointsFromArray(centerline_markup, centerline)

        centerline_markup.SetName(name + "_centers")
        centerline_markup.GetDisplayNode().SetTextScale(self.centerline_text_size)
        centerline_markup.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointStartInteractionEvent, self.on_node_clicked
        )
        centerline_markup.GetDisplayNode().SetSelectedColor(*centerline_color)

        contour_points_markup = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode"
        )
        slicer.util.updateMarkupsControlPointsFromArray(
            contour_points_markup,
            np.array([elt for pts in contour_points for elt in pts]),
        )
        contour_points_markup.GetDisplayNode().SetTextScale(0)
        contour_points_markup.GetDisplayNode().SetVisibility(False)
        contour_points_markup.GetDisplayNode().SetSelectedColor(*contour_points_color)
        contour_points_markup.SetName(name + "_contours")

        self.centerline_markups.append(centerline_markup)
        self.contour_points_markups.append(contour_points_markup)

        if self.lock_button.checked:
            centerline_markup.LockedOn()
            contour_points_markup.LockedOn()

        self.update_visibility_button(TreeColumnRole.VISIBILITY_CENTER)
        self.update_visibility_button(TreeColumnRole.VISIBILITY_CONTOUR)

    def create_new_branch(
        self,
        edge,
        centerline: np.ndarray,
        contour_points: list[list[np.ndarray]],
        centerline_radius: list[float],
        parent_node: Union[str, None] = None,
        isFromSplitBranch: bool = False,
    ):
        """
        Update the graph with the new edge and create associated markups for the centerline and the contour points.

        Parameters
        ----------

        edge: the edge to be added to the graph.
        centerline: array containing the points of the centerline.
        contour_points: array of array of contour points, contour_points[0] are the points sampled around centerline[0],
        contour_points[1] are the points sampled around centerline[1] etc...
        centerline_radius: array containing the underestimated radius of each point of the centerline.
        parent_node: id of its parent node, None if it is the root.
        isFromSplitBranch: flag to check if this new branch is from a split, if it is not from a split we may
        merge branch with its single children.
        """
        new_branch_list = []
        for point in centerline:
            new_branch_list.append(cylinder(center=np.array(point)))
        self.branch_list.append(new_branch_list)

        self.edges.append(edge)
        new_name = "b" + str(len(self.edges))
        self.names.append(new_name)
        self.centerlines.append(centerline)
        self.contours_points.append(contour_points)
        self.centerline_radius.append(centerline_radius)

        self.create_new_markups(new_name, centerline, contour_points)

        self.tree_widget.insertAfterNode(
            nodeId=new_name,
            parentNodeId=parent_node,
            becomeIntermediaryParent=isFromSplitBranch,
        )

        if not isFromSplitBranch:
            self.on_merge_only_child(parent_node)

    def update_parent_branch(self, branch_idx: int, node_idx: int):
        """
        Update the graph when a split occurs.

        Parameters
        ----------

        branch_idx: index of the branch updated.
        node_idx: index of the last point of the branch.
        """
        self.branch_list[branch_idx] = self.branch_list[branch_idx][:node_idx]
        self.centerlines[branch_idx] = self.centerlines[branch_idx][:node_idx]
        self.contours_points[branch_idx] = self.contours_points[branch_idx][:node_idx]
        self.centerline_radius[branch_idx] = self.centerline_radius[branch_idx][
            :node_idx
        ]

        slicer.util.updateMarkupsControlPointsFromArray(
            self.centerline_markups[branch_idx], self.centerlines[branch_idx]
        )
        slicer.util.updateMarkupsControlPointsFromArray(
            self.contour_points_markups[branch_idx],
            np.array([elt for pts in self.contours_points[branch_idx] for elt in pts]),
        )

    def update_visibility_button(self, column: TreeColumnRole):
        """
        Update the text of the button according to the action being the most annoying to do.
        For example, if two out of three items are visible, the action will be to turn them invisible.

        Parameters
        ----------

        column: flag to indicate the column to be updated.
        """
        markup_list, button = (
            (self.centerline_markups, self.centerline_button)
            if column == TreeColumnRole.VISIBILITY_CENTER
            else (self.contour_points_markups, self.contour_point_button)
        )
        majority_visibility = not (
            np.sum([markup.GetDisplayNode().GetVisibility() for markup in markup_list])
            >= max(1, (len(markup_list) // 2))
        )
        button.text = (
            button.text.replace("Hide", "Show")
            if majority_visibility
            else button.text.replace("Show", "Hide")
        )

    def split_branch(
        self, idx_branch: int, idx_cyl: int, parent_node: Union[str, None]
    ):
        """
        Split a branch into two parts.
        Triggered when the user adds a new branch, and the closest cylinder to that branch
        is located in the middle of a branch. Thus, this branch first part become parent of its second part
        and the newly branch created.

        Parameters
        ----------

        idx_branch: index of the branch splited.
        idx_cyl: index of the closest cylinder to the new branch created.
        parent_node: name of the parent of the branch splited, None if it is the root.

        Returns
        ----------

        The beginning parts of the newly branch created.

        """
        # Modify old branch which became a parent
        centerline = self.centerlines[idx_branch]
        contour_points = self.contours_points[idx_branch]
        centerline_radius = self.centerline_radius[idx_branch]
        self.update_parent_branch(idx_branch, idx_cyl + 1)

        # Update edges
        self.nodes.append(centerline[idx_cyl])
        old_end = self.edges[idx_branch][1]
        self.edges[idx_branch] = (self.edges[idx_branch][0], len(self.nodes) - 1)

        # Create new branch from the old one but as a child
        self.create_new_branch(
            (len(self.nodes) - 1, old_end),
            centerline[idx_cyl:],
            contour_points[idx_cyl:],
            centerline_radius[idx_cyl:],
            parent_node,
            True,
        )

        return centerline[idx_cyl : idx_cyl + 1], centerline_radius[
            idx_cyl : idx_cyl + 1
        ], contour_points[idx_cyl : idx_cyl + 1]

    def save_networkX(self):
        """
        Save the graph created as a networkx .JSON and .pickle file if the user select a valid directory.
        """
        dialog = qt.QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Choose a folder")

        # cancel any action if the user cancel / close the window / press escape
        if not folder_path:
            return

        # Create graph Network X with node = bifurcation and edges = branches
        branch_graph = nx.DiGraph()

        for i, n in enumerate(self.nodes):
            branch_graph.add_node(i, pos=n)
        for i, e in enumerate(self.edges):
            branch_graph.add_edge(
                e[0],
                e[1],
                name=self.names[i],
                centerline=self.centerlines[i],
                contour_points=self.contours_points[i],
            )

        # save with pickle
        with open(os.path.join(folder_path, "graph_tree.pickle"), "wb") as f:
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

        # save to JSON
        data = json_graph.node_link_data(branch_graph)
        data_list = ndarray_to_list(data)
        with open(os.path.join(folder_path, "graph_tree.json"), "w") as outfile:
            json.dump(data_list, outfile, indent=4)

        slicer.util.infoDisplay(
            f"The graph has been successfully exported to :\n{folder_path}",
            windowTitle="Success",
        )

    def clear_all(self) -> bool:
        """
        Clear the whole graph after confirmation.

        Returns
        ----------

        True if the whole graph has been deleted else False.
        """
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
        self.centerlines = []
        self.contours_points = []
        self.centerline_radius = []

        with CustomProgressBar(
            total=len(self.centerline_markups),
            quantity_to_measure="branch deleted",
            windowTitle="Clearing tree architecture...",
            width=300,
        ) as progress_bar:
            for _ in range(len(self.centerline_markups)):
                slicer.mrmlScene.RemoveNode(self.centerline_markups.pop())
                slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop())
                progress_bar.update()

        self.tree_widget.clear()
        self.update_visibility_button(TreeColumnRole.VISIBILITY_CENTER)
        self.update_visibility_button(TreeColumnRole.VISIBILITY_CONTOUR)
        return True

    def on_stop_interaction(self):
        if self.current_tree_item is not None:
            self.current_tree_item.updateText()

    def on_item_clicked(self, treeItem, column: TreeColumnRole):
        """
        On item clicked in the tree view do the associated action.

        Parameters
        ----------

        treeItem: tree item on which the user clicked a column.
        column: flag the indicate which column has been clicked.
        """
        self.current_tree_item = treeItem
        node_id = treeItem.nodeId
        branch_id = self.names.index(node_id)
        if column == TreeColumnRole.VISIBILITY_CENTER:
            is_visible = (
                self.centerline_markups[branch_id].GetDisplayNode().GetVisibility()
            )
            self.centerline_markups[branch_id].GetDisplayNode().SetVisibility(
                not is_visible
            )
            self.tree_widget._branchDict[node_id].setIcon(
                TreeColumnRole.VISIBILITY_CENTER,
                Icons.visibleOff if is_visible else Icons.visibleOn,
            )
            self.update_visibility_button(column)
        elif column == TreeColumnRole.VISIBILITY_CONTOUR:
            is_visible = (
                self.contour_points_markups[branch_id].GetDisplayNode().GetVisibility()
            )
            self.contour_points_markups[branch_id].GetDisplayNode().SetVisibility(
                not is_visible
            )
            self.tree_widget._branchDict[node_id].setIcon(
                TreeColumnRole.VISIBILITY_CONTOUR,
                Icons.visibleOff if is_visible else Icons.visibleOn,
            )
            self.update_visibility_button(column)
        elif column == TreeColumnRole.DELETE:
            self.on_delete_item(treeItem)
            self.update_visibility_button(TreeColumnRole.VISIBILITY_CENTER)
            self.update_visibility_button(TreeColumnRole.VISIBILITY_CONTOUR)

    def on_item_renamed(self, previous: str, new: str):
        """
        Rename the markup when the associated branch is renamed.

        Parameters
        ----------

        previous: previous name of the branch.
        new: new name of the branch.
        """
        branch_id = self.names.index(previous)
        self.names[branch_id] = new
        self.centerline_markups[branch_id].SetName(new + "_centers")
        self.contour_points_markups[branch_id].SetName(new + "_contours")

    def on_key_pressed(self, treeItem, key):
        """
        On delete key pressed, delete the current item if any selected.
        Can be modified to manage shortcuts.
        """
        if key == qt.Qt.Key_Delete:
            self.on_delete_item(treeItem)

    def on_header_clicked(self, column: TreeColumnRole):
        """
        On header clicked in the tree view do the associated action.

        Parameters
        ----------

        column: flag to indicate the column clicked.
        """

        def change_majority_visibility(markup_list, column, button):
            majority_visibility = not (
                np.sum(
                    [markup.GetDisplayNode().GetVisibility() for markup in markup_list]
                )
                >= max(1, (len(markup_list) // 2))
            )
            icon = Icons.visibleOn if majority_visibility else Icons.visibleOff
            for markup in markup_list:
                markup.GetDisplayNode().SetVisibility(majority_visibility)
            for branch in self.tree_widget._branchDict.values():
                branch.setIcon(column, icon)
            self.update_visibility_button(column)

        if column == TreeColumnRole.VISIBILITY_CENTER:
            change_majority_visibility(
                self.centerline_markups, column, self.centerline_button
            )
        elif column == TreeColumnRole.VISIBILITY_CONTOUR:
            change_majority_visibility(
                self.contour_points_markups, column, self.contour_point_button
            )

    def on_node_clicked(self, caller, event):
        """
        Callback function when the user click a node in the 3D slicer's 3D view.

        Parameters
        ----------

        caller: markup node the user clicked.
        """
        displayNode = caller.GetDisplayNode()
        if (
            displayNode.GetActiveComponentType()
            == slicer.vtkMRMLMarkupsDisplayNode.ComponentControlPoint
        ):
            node_id = displayNode.GetActiveComponentIndex()

            branch_name = "_".join(caller.GetName().split("_")[:-1])
            branch_id = self.names.index(branch_name)
            self.node_selected = (branch_id, node_id)

            tree_item = self.tree_widget.getTreeWidgetItem(branch_name)
            self.tree_widget.scrollToItem(tree_item)
            self.tree_widget.setCurrentItem(tree_item)

    def on_remove_end(self, treeItem):
        """
        Callback function when the user choose to delete the end of a branch.

        Parameters
        ----------

        treeItem: tree item in which the user wants to delete the end.
        """
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
        if branch_node_id == len(self.centerlines[branch_id]) - 1:
            return

        edges_node_id = self.edges[branch_id][1]
        self.nodes[edges_node_id] = self.centerlines[branch_id][branch_node_id]
        self.update_parent_branch(branch_id, branch_node_id + 1)

    def delete_node(self, index: int):
        """
        Delete a node from the graph

        Parameters
        ----------

        index: index of the node to be removed.
        """
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
        Callback function when the user choose to delete a branch from the tree view.
        If the branch has childs, a confirmation is required.

        Parameters
        ----------

        treeItem: tree item the user wish to delete.
        showPopupForNonLeaf: flag to indicate that it is the initial call to the function, default is True
        so that it does not recursivly ask for deletion confirmation.
        """
        self.on_stop_interaction()
        node_id = treeItem.nodeId

        if self.tree_widget.isRoot(node_id):
            slicer.util.errorDisplay(
                text="You can't delete the root", windowTitle="Error"
            )
            return

        children = [
            self.tree_widget.getTreeWidgetItem(n_id)
            for n_id in self.tree_widget.getChildrenNodeId(node_id)
        ]

        if len(children) != 0 and showPopupForNonLeaf:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setWindowTitle("Confirmation")
            msg.setText(
                f"Are you sure you want to delete {node_id} and all its children ?"
            )
            msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
            if msg.exec_() != qt.QMessageBox.Yes:
                return

        for child in children:
            self.on_delete_item(child, showPopupForNonLeaf=False)

        branch_id = self.names.index(node_id)
        self.delete_node(self.edges[branch_id][1])

        self.names.pop(branch_id)
        self.branch_list.pop(branch_id)
        self.centerlines.pop(branch_id)
        self.contours_points.pop(branch_id)
        self.centerline_radius.pop(branch_id)
        slicer.mrmlScene.RemoveNode(self.centerline_markups.pop(branch_id))
        slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop(branch_id))

        self.edges.pop(branch_id)
        parent_id = self.tree_widget.getParentNodeId(node_id)
        self.tree_widget.removeNode(node_id)

        if self.current_tree_item == treeItem:
            self.current_tree_item = None

        if showPopupForNonLeaf:
            self.on_merge_only_child(parent_id)

    def on_merge_only_child(self, branch_id: str):
        """
        Merge branch if it contains a single child.

        Parameters
        ----------

        branch_id: id of the branch which should be checked for merge.
        """
        if branch_id is None:
            return
        child_list = self.tree_widget.getChildrenNodeId(branch_id)
        if len(child_list) != 1:
            return

        parent_idx = self.names.index(branch_id)
        child_idx = self.names.index(child_list[0])

        # Modify parent branch to add child branch
        self.centerlines[parent_idx] = np.vstack(
            (self.centerlines[parent_idx], self.centerlines[child_idx][1:])
        )
        self.centerlines.pop(child_idx)
        self.contours_points[parent_idx] += self.contours_points[child_idx][1:]
        self.contours_points.pop(child_idx)
        self.centerline_radius[parent_idx] += self.centerline_radius[child_idx][1:]
        self.centerline_radius.pop(child_idx)
        slicer.util.updateMarkupsControlPointsFromArray(
            self.centerline_markups[parent_idx], self.centerlines[parent_idx]
        )
        slicer.util.updateMarkupsControlPointsFromArray(
            self.contour_points_markups[parent_idx],
            np.array([elt for pts in self.contours_points[parent_idx] for elt in pts]),
        )
        slicer.mrmlScene.RemoveNode(self.centerline_markups.pop(child_idx))
        slicer.mrmlScene.RemoveNode(self.contour_points_markups.pop(child_idx))
        self.branch_list[parent_idx] += self.branch_list[child_idx]
        self.branch_list.pop(child_idx)

        # Delete old child
        self.delete_node(self.edges[child_idx][0])
        self.edges[parent_idx] = self.edges[parent_idx][0], self.edges[child_idx][1]
        self.edges.pop(child_idx)
        self.names.pop(child_idx)

        self.tree_widget.removeNode(child_list[0])
