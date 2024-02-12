import qt

from .segmentation_utils import Signal, Icons

class TreeColumnRole:
    NODE_ID = 0
    VISIBILITY_CENTER = 1
    VISIBILITY_CONTOUR = 2
    DELETE = 3

class BranchTreeItem(qt.QTreeWidgetItem):
  """Helper class holding nodeId and nodeName in the VesselBranchTree
  """

  def __init__(self, nodeId):
    qt.QTreeWidgetItem.__init__(self)
    self.nodeId = nodeId
    self.setIcon(TreeColumnRole.VISIBILITY_CENTER, Icons.visibleOn)
    self.setIcon(TreeColumnRole.VISIBILITY_CONTOUR, Icons.visibleOff)
    self.setIcon(TreeColumnRole.DELETE, Icons.delete)
    self.setFlags(self.flags() | qt.Qt.ItemIsEditable)
    self.updateText()

  @property
  def status(self):
    return self._status

  @status.setter
  def status(self, status):
    self._status = status
    self.updateText()

  def updateText(self):
    self.setText(0, f"{self.nodeId}")

class BranchTree(qt.QTreeWidget):
  """Tree representation of vessel branch nodes.

  Class enables inserting new vessel node branches after or before existing nodes.
  Class signals when modified or user interacts with the UI.
  """

  def __init__(self, parent=None):
    qt.QTreeWidget.__init__(self, parent)

    self.keyPressed = Signal("VesselBranchTreeItem, qt.Qt.Key")
    self.setContextMenuPolicy(qt.Qt.CustomContextMenu)
    self.customContextMenuRequested.connect(self.onContextMenu)
    self.itemChanged.connect(self.onItemChange)
    self.itemRenamed = Signal(str, str)
    self.itemDropped = Signal()
    self.itemRemoveEnd = Signal("VesselBranchTreeItem")
    self.itemDeleted = Signal("VesselBranchTreeItem")

    self._branchDict = {}

    # Configure tree widget
    self.setColumnCount(3)
    self.setHeaderLabels(["Branch Name", " Center", " Contour", ""])

    # Configure tree to have first section stretched and last sections to be at right of the layout
    # other columns will always be at minimum size fitting the icons
    self.header().setSectionResizeMode(0, qt.QHeaderView.Stretch)
    self.header().setStretchLastSection(False)
    self.header().setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
    self.header().setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
    self.header().setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
    self.headerItem().setIcon(TreeColumnRole.VISIBILITY_CENTER, Icons.toggleVisibility)
    self.headerItem().setIcon(TreeColumnRole.VISIBILITY_CONTOUR, Icons.toggleVisibility)
    self.headerItem().setIcon(TreeColumnRole.DELETE, Icons.delete)

    # Enable reordering by drag and drop
    self.setDragEnabled(False)
    self.setDropIndicatorShown(True)
    self.setDragDropMode(qt.QAbstractItemView.InternalMove)
    self.setAccessibleName("branch_tree")

  def clear(self):
    self._branchDict = {}
    qt.QTreeWidget.clear(self)

  def clickItem(self, item):
    item = self.getTreeWidgetItem(item) if isinstance(item, str) else item
    self.setItemSelected(item)
    if item is not None:
      self.itemClicked.emit(item, 0)

  def setItemSelected(self, item):
    if item is not None:
      self.selectionModel().clearSelection()
      item.setSelected(True)

  def isInTree(self, nodeId):
    """
    Parameters
    ----------
    nodeId: str
      Id of the node which may ne in the tree.

    Returns
    -------
    bool
      True if nodeId is part of the tree, False otherwise.
    """
    return nodeId in self._branchDict.keys()

  def isRoot(self, nodeId):
    """
    :return: True if node doesn't have any parents
    """
    return self.getParentNodeId(nodeId) is None

  def dropEvent(self, event):
    """On drop event, enforce structure of the tree is not broken.
    """
    qt.QTreeWidget.dropEvent(self, event)
    self.enforceOneRoot()
    self.itemDropped.emit()

  def keyPressEvent(self, event):
    """Overridden from qt.QTreeWidget to notify listeners of key event

    Parameters
    ----------
    event: qt.QKeyEvent
    """
    if self.currentItem():
      self.keyPressed.emit(self.currentItem(), event.key())

    qt.QTreeWidget.keyPressEvent(self, event)

  def onContextMenu(self, position):
    item = self.itemAt(position)

    renameAction = qt.QAction("Rename")
    renameAction.triggered.connect(self.renameItem)

    removeEndAction = qt.QAction("Remove end of the branch")
    removeEndAction.triggered.connect(lambda :self.itemRemoveEnd.emit(self.currentItem()))
    if self.isLeaf(item.text(0)):
        removeEndAction.setEnabled(True)
    else:
        removeEndAction.setEnabled(False)

    deleteAction = qt.QAction("Delete")
    deleteAction.triggered.connect(lambda :self.itemDeleted.emit(self.currentItem()))

    menu = qt.QMenu(self)
    menu.addAction(renameAction)
    menu.addAction(removeEndAction)
    menu.addAction(deleteAction)

    menu.exec_(self.mapToGlobal(position))

  def onItemChange(self, item, column):
    previous = item.nodeId
    new = item.text(0)

    # Forbid renaming with existing name
    if self.isInTree(new):
      item.updateText()
      return

    self._branchDict[item.text(0)] = self._branchDict.pop(item.nodeId)
    item.nodeId = new
    item.updateText()
    self.itemRenamed.emit(previous, new)

  def renameItem(self):
    item: BranchTreeItem = self.currentItem()
    self.editItem(item, 0)

  def _takeItem(self, nodeId):
    """Remove item with given item id from the tree. Removes it from its parent if necessary
    """
    if nodeId is None:
      return None
    elif nodeId in self._branchDict:
      nodeItem = self._branchDict[nodeId]
      self._removeFromParent(nodeItem)
      return nodeItem
    else:
      return BranchTreeItem(nodeId)

  def _removeFromParent(self, nodeItem):
    """Remove input node item from its parent if it is attached to an item or from the TreeWidget if at the root
    """
    parent = nodeItem.parent()
    if parent is not None:
      parent.removeChild(nodeItem)
    else:
      self.takeTopLevelItem(self.indexOfTopLevelItem(nodeItem))

  def _insertNode(self, nodeId, parentId):
    """Insert the nodeId with input node name as child of the item whose name is parentId. If parentId is None, the item
    will be added as a root of the tree

    Parameters
    ----------
    nodeId: str
      Unique id of the node to add to the tree
    parentId: str or None
      Unique id of the parent node. If None or "" will add node as root
    """
    nodeItem = self._takeItem(nodeId)
    if not parentId:
      hasRoot = self.topLevelItemCount > 0
      self.addTopLevelItem(nodeItem)
      if hasRoot:
        rootItem = self.takeTopLevelItem(0)
        nodeItem.addChild(rootItem)
      self._branchDict[nodeId] = nodeItem
    else:
      children = self.getChildrenNodeId(parentId)
      self._branchDict[parentId].addChild(nodeItem)
      self._branchDict[nodeId] = nodeItem
      if len(children) == 2:
        for child in children:
          self._insertNode(child, nodeId)

    return nodeItem

  def insertAfterNode(self, nodeId, parentNodeId):
    """Insert given node after the input parent Id. Inserts new node as root if parentNodeId is None.
    If root is already present in the tree and insert after None is used, new node will become the parent of existing
    root node.

    Parameters
    ----------
    nodeId: str
      Unique ID of the node to insert in the tree
    parentNodeId: str or None
      Unique ID of the parent node. If None, new node will be inserted as root.
    status: PlaceStatus

    Raises
    ------
      ValueError
        If parentNodeId is not None and doesn't exist in the tree
    """
    self._insertNode(nodeId, parentNodeId)
    self.expandAll()

  def removeNode(self, nodeId):
    """Remove given node from tree.

    If node is root, only remove if it has exactly one direct child and replace root by child. Else does nothing.
    If intermediate item, move each child of node to node parent.

    Parameters
    ----------
    nodeId: str
      Id of the node to remove from tree

    Returns
    -------
    bool - True if node was removed, False otherwise
    """
    nodeItem = self._branchDict[nodeId]
    if nodeItem.parent() is None:
      return False
    else:
      self._removeIntermediateItem(nodeItem, nodeId)
      return True

  def _removeIntermediateItem(self, nodeItem, nodeId):
    """Move each child of node to node parent and remove item.
    """
    parentItem = nodeItem.parent()
    parentItem.takeChild(parentItem.indexOfChild(nodeItem))
    for child in nodeItem.takeChildren():
      parentItem.addChild(child)
    del self._branchDict[nodeId]

  def getParentNodeId(self, childNodeId):
    """

    Parameters
    ----------
    childNodeId: str
      Node for which we want the parent id

    Returns
    -------
    str or None
      Id of the parent item or None if node has no parent
    """
    parentItem = self._branchDict[childNodeId].parent()
    return parentItem.nodeId if parentItem is not None else None

  def getChildrenNodeId(self, parentNodeId):
    """
    Returns
    -------
    List[str]
      List of nodeIds of every children associated with parentNodeId
    """
    parent = self._branchDict[parentNodeId]
    return [parent.child(i).nodeId for i in range(parent.childCount())]

  def getNodeList(self):
    """
    Returns
    -------
    List[str]
      List of every nodeIds referenced in the tree
    """
    return self._branchDict.keys()

  def getTreeWidgetItem(self, nodeId):
    return self._branchDict[nodeId] if nodeId in self._branchDict else None

  def getText(self, nodeId):
    item = self.getTreeWidgetItem(nodeId)
    return item.text(0) if item is not None else ""

  def isLeaf(self, nodeId):
    """
    Returns
    -------
    bool
      True if nodeId has no children item, False otherwise
    """
    return len(self.getChildrenNodeId(nodeId)) == 0

  def enforceOneRoot(self):
    """Reorders tree to have only one root item. If elements are defined after root, they will be inserted before
    current root. Methods is called during drop events.
    """
    # Early return if tree has at most one root
    if self.topLevelItemCount <= 1:
      return

    # Set current root as second item child
    newRoot = self.takeTopLevelItem(1)
    currentRoot = self.takeTopLevelItem(0)
    newRoot.addChild(currentRoot)

    # Add the new root to the tree
    self.insertTopLevelItem(0, newRoot)

    # Expand both items
    newRoot.setExpanded(True)
    currentRoot.setExpanded(True)

    # Call recursively until the whole tree has only one root
    self.enforceOneRoot()
