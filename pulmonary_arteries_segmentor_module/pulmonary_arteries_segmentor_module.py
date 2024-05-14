import importlib
import sys
from typing import Annotated, Optional

import numpy as np
import qt
import slicer
import slicer.util
import vtk
import json

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLMarkupsFiducialNode
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModule,
    ScriptedLoadableModuleTest,
)
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer.util import VTKObservationMixin
from vtkSlicerMarkupsModuleMRMLPython import vtkMRMLMarkupsNode

# Recursive reload, when you hit the "reload" button in 3D slicer, force all submodules to be reloaded (which is not the case by default).
try:
    to_reload = [key for key in sys.modules.keys() if "ransac_slicer." in key]
    for file_to_reload in to_reload:
        sys.modules[file_to_reload] = importlib.reload(sys.modules[file_to_reload])
except Exception as e:
    print(f"Exception occurred while reloading\n{e}")

from ransac_slicer.cylinder import cylinder
from ransac_slicer.ransac import run_ransac
from ransac_slicer.graph_branches import GraphBranches
from ransac_slicer.branch_tree import BranchTree, TreeColumnRole, Icons
from ransac_slicer.color_palettes import direction_points_color
from ransac_slicer.popup_utils import (
    CustomProgressBar,
    CustomStatusDialog,
)
from ransac_slicer.volume import volume
from ransac_slicer.region_growing_seeds import paint_segments

from networkx.readwrite import json_graph
import networkx as nx

#
# pulmonary_arteries_segmentor_module
#


class pulmonary_arteries_segmentor_module(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Pulmonary Arteries Segmentor"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Azéline Aillet", "Gabriel Jacquinot"]
        self.parent.helpText = """
A 3D Slicer plugin for pulmonary artery extraction from angiography images.
"""
        self.parent.acknowledgementText = """
This plugin is an end-of-study project, made by Azéline Aillet (Student at EPITA) and Gabriel Jacquinot (Student at EPITA), under the direction of Odyssée Merveille (CREATIS) and Morgane Des Ligneris (CREATIS).
The RANSAC code is based on the previous work of Jack CARBONERO (CReSTIC), Guillaume DOLLE (LMR) and Nicolas PASSAT (CReSTIC) on the plugin vestract.
The hierarchy code is based on the work of Lucie Macron (Kitware SAS), Thibault Pelletier (Kitware SAS), Camille Huet (Kitware SAS), Leo Sanchez (Kitware SAS) from the RVesselX plugin.
"""


#
# pulmonary_arteries_segmentor_moduleParameterNode
#


@parameterNodeWrapper
class pulmonary_arteries_segmentor_moduleParameterNode:
    """
    Class to wrap the inputs of the RANSAC algorithm, the fields are automaticaly updated.

    Fields
    ----------

    inputVolume: input volume to extract the arteries from.
    startingPoint: starting point list for RANSAC cylinders.
    directionPoint: direction point list for RANSAC cylinders.
    percentInlierPoints: percentage of inlier points to validate a cylinder.
    percentThreshold: percentage of last cylinders radius to make a point inlier of a cylinder.
    startingRadius: initiale radius of the first cylinder to fit
    """

    # Begin tab
    inputVolume: vtkMRMLScalarVolumeNode
    startingPoint: vtkMRMLMarkupsFiducialNode
    directionPoint: vtkMRMLMarkupsFiducialNode
    percentInlierPoints: Annotated[float, WithinRange(0, 100)] = 60.0
    percentThreshold: Annotated[float, WithinRange(0, 100)] = 30.0
    startingRadius: Annotated[float, WithinRange(0, 100)] = 0.0


#
# pulmonary_arteries_segmentor_moduleWidget
#


class pulmonary_arteries_segmentor_moduleWidget(
    ScriptedLoadableModuleWidget, VTKObservationMixin
):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """

        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.graph_branches = None
        self.segmentationNode = None
        self.nodeDeletionObserverTag = None
        self.isPlacingPoints = False

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(
            self.resourcePath("UI/pulmonary_arteries_segmentor_module.ui")
        )
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(
            segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode"
        )
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass(
                "vtkMRMLSegmentEditorNode"
            )
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        if (
            not hasattr(self, "segmentEditorNode")
            or self.segmentEditorNode != segmentEditorNode
        ):
            self.segmentEditorNode = segmentEditorNode
            self.ui.SegmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = pulmonary_arteries_segmentor_moduleLogic()

        self.branch_tree = BranchTree()
        begin_tab = self.ui.tabWidget.widget(0)

        # Insert the branch tree widget defined in code
        begin_tab.layout().insertWidget(6, self.branch_tree)

        self.graph_branches = GraphBranches(
            self.branch_tree,
            self.ui.showCenterlineButton,
            self.ui.showContourPointsButton,
            self.ui.lockButton,
        )
        # Connections / Callbacks

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # Sliders callbacks
        self.ui.centerlineTextSize.connect("valueChanged(double)", self.changeTextSize)

        # Buttons callbacks
        self.ui.placePointButton.connect("clicked(bool)", self.startPlacePointProcedure)

        self.ui.createBranch.connect("clicked(bool)", self.create_branch)
        self.ui.clearTree.connect(
            "clicked(bool)",
            lambda: (
                self.graph_branches.clear_all(),
                self.updateSegmentationButtonState(),
                self._checkCanStartRansac(),
            ),
        )

        self.ui.exportTreeButton.connect(
            "clicked(bool)", self.graph_branches.save_networkX
        )
        self.ui.loadTreeArchitectureButton.connect(
            "clicked(bool)", self.onLoadTreeArchitecture
        )

        self.ui.paintButton.connect("clicked(bool)", self.onStartSegmentationButton)

        self.ui.lockButton.connect("clicked(bool)", self.onLockButton)
        self.ui.showCenterlineButton.connect(
            "clicked(bool)",
            lambda: self.graph_branches.on_header_clicked(
                TreeColumnRole.VISIBILITY_CENTER
            ),
        )
        self.ui.showContourPointsButton.connect(
            "clicked(bool)",
            lambda: self.graph_branches.on_header_clicked(
                TreeColumnRole.VISIBILITY_CONTOUR
            ),
        )

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.checkCanPlacePoint()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._checkCanStartRansac,
            )

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and are observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if node:
                self._parameterNode.inputVolume = node

        # Create starting and direction point parameters if they do not already exist and select them
        if not self._parameterNode.startingPoint:
            node = slicer.util.getFirstNodeByClassByName(
                "vtkMRMLMarkupsFiducialNode", "s"
            )
            if node is None or not isinstance(node, slicer.vtkMRMLMarkupsFiducialNode):
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
                node.SetName("s")
            node.GetDisplayNode().SetSelectedColor(*direction_points_color)
            self._parameterNode.startingPoint = node

        if not self._parameterNode.directionPoint:
            node = slicer.util.getFirstNodeByClassByName(
                "vtkMRMLMarkupsFiducialNode", "d"
            )
            if node is None or not isinstance(node, slicer.vtkMRMLMarkupsFiducialNode):
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
                node.SetName("d")
            node.GetDisplayNode().SetSelectedColor(*direction_points_color)
            self._parameterNode.directionPoint = node

    def setParameterNode(
        self,
        inputParameterNode: Optional[pulmonary_arteries_segmentor_moduleParameterNode],
    ) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._checkCanStartRansac,
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._checkCanStartRansac,
            )
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateSegmentationButtonState,
            )
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.checkCanPlacePoint,
            )
            self._checkCanStartRansac()

    def _getParametersRansac(self) -> list:
        """
        Return parameters of the parameter node as a list.
        """
        return [
            self._parameterNode.inputVolume,
            self._parameterNode.startingPoint,
            self._parameterNode.directionPoint,
            self._parameterNode.percentInlierPoints,
            self._parameterNode.percentThreshold,
            self._parameterNode.startingRadius,
        ]

    def _addObserver(self, obj, event, fct):
        """
        Wrapper of addObserver function, does nothing if the observer already exists.
        """
        if not self.hasObserver(obj, event, fct):
            self.addObserver(obj, event, fct)

    def _removeObserver(self, obj, event, fct):
        """
        Wrapper of removeObserver function, does nothing if the observer does not exists.
        """
        if self.hasObserver(obj, event, fct):
            self.removeObserver(obj, event, fct)

    def _checkCanStartRansac(self, caller=None, event=None) -> None:
        """
        Update the create and delete button state depending on the parameters state.
        """
        starting_point = self._parameterNode.startingPoint
        direction_point = self._parameterNode.directionPoint

        if starting_point:
            self._addObserver(
                starting_point,
                vtkMRMLMarkupsNode.PointAddedEvent,
                self._checkCanStartRansac,
            )
            self._addObserver(
                starting_point,
                vtkMRMLMarkupsNode.PointRemovedEvent,
                self._checkCanStartRansac,
            )

        if direction_point:
            self._addObserver(
                direction_point,
                vtkMRMLMarkupsNode.PointAddedEvent,
                self._checkCanStartRansac,
            )
            self._addObserver(
                direction_point,
                vtkMRMLMarkupsNode.PointRemovedEvent,
                self._checkCanStartRansac,
            )

        if (
            self._parameterNode
            and not self.isPlacingPoints
            and all(self._getParametersRansac())
            and starting_point.GetNumberOfControlPoints()
            and direction_point.GetNumberOfControlPoints()
        ):
            self.ui.createBranch.enabled = True
            if len(self.graph_branches.names) == 0:
                self.ui.createBranch.text = "Create Root"
                self.ui.createBranch.toolTip = "Create Root."
            else:
                self.ui.createBranch.text = "Create New Branch"
                self.ui.createBranch.toolTip = "Create New Branch."
        else:
            self.ui.createBranch.toolTip = "Select all input before creating branch."
            self.ui.createBranch.enabled = False

        if len(self.graph_branches.names) != 0:
            self.ui.clearTree.toolTip = "Clear all tree."
            self.ui.clearTree.enabled = True
            self.ui.exportTreeButton.toolTip = "Export the network X graph of the centerlines and contour points as JSON and pickle."
            self.ui.exportTreeButton.enabled = True
        else:
            self.ui.clearTree.toolTip = "Tree is already empty."
            self.ui.clearTree.enabled = False
            self.ui.exportTreeButton.toolTip = "There is nothing to save."
            self.ui.exportTreeButton.enabled = False

    def checkCanPlacePoint(self, *args):
        """
        Enables the place button if the starting and direction point parameters exist.
        """
        self.ui.placePointButton.enabled = (
            self._parameterNode.startingPoint and self._parameterNode.directionPoint
        )

    def startPlacePointProcedure(self):
        """
        Start a point placement procedure.

        Each function of this procedure are called through observers callbacks.
        """
        self.startingPointPlaced = False
        self.directionPointPlaced = False
        self.isPlacingPoints = True
        self.ui.createBranch.enabled = False

        starting_point = self._parameterNode.startingPoint
        starting_point.GetDisplayNode().SetSelectedColor(*direction_points_color)

        # Prepare the case where the user place the first point
        self._addObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.directionPointPlacement,
        )

        # Prepare the case where the user cancel the point placement
        self._addObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.resetPlacementState,
        )

        # Start placing procedure
        slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(
            starting_point.GetID()
        )
        slicer.modules.markups.logic().StartPlaceMode(1)

    def directionPointPlacement(self, *args):
        """
        First state of a point placement procedure.

        Each function of this procedure are called through observers callbacks.
        """
        self.startingPointPlaced = True
        starting_point = self._parameterNode.startingPoint

        if not starting_point:
            return

        self._removeObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.directionPointPlacement,
        )
        self._removeObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.resetPlacementState,
        )

        direction_point = self._parameterNode.directionPoint
        direction_point.GetDisplayNode().SetSelectedColor(*direction_points_color)

        # Prepare the case where the user placed the last point
        self._addObserver(
            direction_point,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.validate_last_point,
        )

        # Prepare the case where the user cancel the point placement
        self._addObserver(
            direction_point,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.resetPlacementState,
        )

        # Place direction point
        slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(
            direction_point.GetID()
        )
        slicer.modules.markups.logic().StartPlaceMode(1)

    def validate_last_point(self, *args):
        """
        Finish a point placement procedure.

        Each function of this procedure are called through observers callbacks.
        """
        self.directionPointPlaced = True
        slicer.modules.markups.logic().StartPlaceMode(0)
        self.resetPlacementState()

    def resetPlacementState(self, *args):
        """
        Reset the state of the point placement procedure.

        Meaning if the user placed only half of points or cancels point placement,
        put the system into a stable state.

        If all points have been placed, remove the extra points.

        Each function of this procedure are called through observers callbacks.
        """
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()

        if (
            interactionNode.GetPlaceModePersistence() == 1
            and interactionNode.GetCurrentInteractionMode() == 1
        ):
            # We do not go further because if those conditions are met, it means that the user moved cursor out of window
            return

        self.isPlacingPoints = False
        self._checkCanStartRansac()

        starting_point = self._parameterNode.startingPoint
        self._removeObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.directionPointPlacement,
        )
        self._removeObserver(
            starting_point,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.resetPlacementState,
        )

        direction_point = self._parameterNode.directionPoint
        self._removeObserver(
            direction_point,
            vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.validate_last_point,
        )
        self._removeObserver(
            direction_point,
            vtkMRMLMarkupsNode.PointRemovedEvent,
            self.resetPlacementState,
        )

        if self.startingPointPlaced and not self.directionPointPlaced:
            starting_point.RemoveNthControlPoint(
                starting_point.GetNumberOfControlPoints() - 1
            )

        while starting_point.GetNumberOfControlPoints() >= 2:
            starting_point.RemoveNthControlPoint(0)

        while direction_point.GetNumberOfControlPoints() >= 2:
            direction_point.RemoveNthControlPoint(0)

        self.startingPointPlaced = False
        self.directionPointPlaced = False

    def recenter3dView(self) -> None:
        """
        Recenter the 3D slicer's 3D view on the subject.
        """
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()

    def create_branch(self) -> None:
        """
        Start the RANSAC algorithm according to user input parameters.
        """
        with slicer.util.tryWithErrorDisplay(
            "Failed to compute tracking.", waitCursor=True
        ):
            progress_dialog = CustomStatusDialog(
                windowTitle="Computing centerline...",
                text="Please wait",
                width=300,
                height=50,
            )
            self.graph_branches = self.logic.processBranch(
                self._getParametersRansac(),
                self.ui.centerlineResolutionDoubleSpinBox.value,
                self.graph_branches,
                self.ui.createBranch.text == "Create New Branch",
                progress_dialog,
            )

            self.recenter3dView()

            # Select the starting markup node to ease future node placement
            slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(
                self._parameterNode.startingPoint.GetID()
            )

            self._checkCanStartRansac()
            self.updateSegmentationButtonState()

    def changeTextSize(self, value):
        """
        Update centerline label text size when moving the slider.
        """
        self.graph_branches.centerline_text_size = value
        for markup in self.graph_branches.centerline_markups:
            markup.GetDisplayNode().SetTextScale(value)

    def updateSegmentationButtonState(self, *args):
        """
        Enables the create segmentation button when at least one branch exist.
        """
        paintButton: qt.QPushButton = self.ui.paintButton
        paintButton.enabled = (
            len(self.graph_branches.centerline_markups) != 0
            and self._parameterNode.inputVolume
        )

        if self.segmentationNode is None or self.segmentationNode.GetScene() is None:
            self.segmentationNode = None
            if self.nodeDeletionObserverTag is not None:
                slicer.mrmlScene.RemoveObserver(self.nodeDeletionObserverTag)
                self.nodeDeletionObserverTag = None

    def onStartSegmentationButton(self) -> None:
        """
        Compute and create the segments inside the segmentation node according
        to the graph_branches architecture.
        """
        with slicer.util.tryWithErrorDisplay(
            "Failed to compute segmentation.", waitCursor=True
        ):
            if (
                self.segmentationNode is None
                or self.segmentationNode.GetScene() is None
            ):
                self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode"
                )
                self.nodeDeletionObserverTag = slicer.mrmlScene.AddObserver(
                    slicer.vtkMRMLScene.NodeAboutToBeRemovedEvent,
                    self.updateSegmentationButtonState,
                )
                self.segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(
                    self._parameterNode.inputVolume
                )

                self.segmentationNode.SetName("Segmentation")
                self.segmentationNode.CreateDefaultDisplayNodes()

            # We do pause the tracking of segmentation deletion
            slicer.mrmlScene.RemoveObserver(self.nodeDeletionObserverTag)
            self.nodeDeletionObserverTag = None

            # Compute branch drawing order, we draw in reverse bfs order, so that parent branches are always drawn on top of childs
            G = nx.DiGraph()
            for i, node in enumerate(self.graph_branches.nodes):
                G.add_node(i, pos=node)

            for i, edge in enumerate(self.graph_branches.edges):
                G.add_edge(
                    edge[0],
                    edge[1],
                    edge_idx=i,
                )
            branch_draw_order = [
                G[a][b]["edge_idx"] for a, b in nx.bfs_edges(G, source=0)
            ][::-1]
            del G

            # Create the segments and paint them
            paint_segments(
                self._parameterNode.inputVolume,
                self.graph_branches.centerlines,
                self.graph_branches.names,
                self.graph_branches.centerline_radius,
                branch_draw_order,
                self.segmentationNode,
                self.ui.reductionFactorSlider.value,
                self.ui.reductionThreshold.value,
                self.ui.contourSpinbox.value,
                self.ui.mergeAllVesselsCheckBox.checked,
            )

            # Set the current segmentation into the UI
            self.ui.SegmentEditorWidget.setSegmentationNode(self.segmentationNode)

            # Hide markup nodes
            for branch in (
                self.graph_branches.centerline_markups
                + self.graph_branches.contour_points_markups
            ):
                branch.GetDisplayNode().SetVisibility(False)

            for branch in self.graph_branches.tree_widget._branchDict.values():
                branch.setIcon(TreeColumnRole.VISIBILITY_CENTER, Icons.visibleOff)
                branch.setIcon(TreeColumnRole.VISIBILITY_CONTOUR, Icons.visibleOff)
            self.ui.showCenterlineButton.text = "Show Centerlines"
            self.ui.showContourPointsButton.text = "Show Contour Points"

            self.updateSegmentationButtonState()

            # We track segmentation deletion
            self.nodeDeletionObserverTag = slicer.mrmlScene.AddObserver(
                slicer.vtkMRMLScene.NodeAboutToBeRemovedEvent,
                self.updateSegmentationButtonState,
            )

    def onLockButton(self) -> None:
        """
        Lock / Unlock branches, disable any interactions you may have with branches.
        When locked, you cannot accidentally select / move points.
        """
        button = self.ui.lockButton
        markups = (
            self.graph_branches.centerline_markups
            + self.graph_branches.contour_points_markups
        )

        if button.checked:
            button.text = "Unlock Tree"
            for markup in markups:
                markup.LockedOn()
        else:
            button.text = "Lock Tree"
            for markup in markups:
                markup.LockedOff()

    def onLoadTreeArchitecture(self) -> None:
        """
        Loads a vessel tree architecture from a .JSON file.

        Ask first if the user wants to delete the current tree.
        If not, does nothing.
        """
        dialog = qt.QFileDialog()
        file_path = dialog.getOpenFileName(
            None, "Choose a file", "", "JSON file (*.json)"
        )

        # cancel any action if the user cancel / close the window / press escape
        if not file_path:
            return

        with slicer.util.tryWithErrorDisplay(
            "Failed to load tree architecture.", waitCursor=False
        ):
            with open(file_path) as f:
                js_graph = json.load(f)
            graph: nx.DiGraph = json_graph.node_link_graph(js_graph)

            # We ask to clear the tree before loading the new one, if not we do nothing
            if not self.graph_branches.clear_all():
                return
            self.updateSegmentationButtonState()
            self._checkCanStartRansac()

        with slicer.util.tryWithErrorDisplay(
            "Failed to restore tree architecture.", waitCursor=True
        ):
            edge_name_table = {0: None}

            with CustomProgressBar(
                total=len(graph.edges),
                quantity_to_measure="branch loaded",
                windowTitle="Restoring tree architecture...",
                width=300,
            ) as progress_bar:
                for a, b in graph.edges:
                    # Restoring lists
                    self.graph_branches.branch_list.append(
                        [
                            cylinder(center=np.array(cp))
                            for cp in graph[a][b]["centerline"]
                        ]
                    )
                    self.graph_branches.names.append(graph[a][b]["name"])
                    self.graph_branches.centerlines.append(
                        np.array(graph[a][b]["centerline"])
                    )
                    self.graph_branches.contours_points.append(
                        graph[a][b]["contour_points"]
                    )
                    # Recompute radius
                    self.graph_branches.centerline_radius.append(
                        [
                            np.linalg.norm(
                                np.array(graph[a][b]["contour_points"][k])
                                - np.array(graph[a][b]["centerline"][k]),
                                axis=1,
                            ).min()
                            for k in range(len(graph[a][b]["centerline"]))
                        ]
                    )
                    self.graph_branches.edges.append((a, b))
                    self.graph_branches.create_new_markups(
                        graph[a][b]["name"],
                        np.array(graph[a][b]["centerline"]),
                        graph[a][b]["contour_points"],
                    )
                    edge_name_table[b] = graph[a][b]["name"]

                    progress_bar.update()

            for node in graph.nodes(data=True):
                self.graph_branches.nodes.append(node[1]["pos"])

            for a, b in nx.edge_dfs(graph):
                current_edge_name = graph[a][b]["name"]
                parent_edge_name = edge_name_table[a]
                self.graph_branches.tree_widget.insertAfterNode(
                    nodeId=current_edge_name, parentNodeId=parent_edge_name
                )
            self._checkCanStartRansac()
            self.updateSegmentationButtonState()
            self.recenter3dView()


#
# pulmonary_arteries_segmentor_moduleLogic
#


class pulmonary_arteries_segmentor_moduleLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        """
        Returns parent's parameter node.
        """
        return pulmonary_arteries_segmentor_moduleParameterNode(
            super().getParameterNode()
        )

    def processBranch(
        self,
        parameters: list,
        centerline_resolution: float,
        graph_branches: GraphBranches,
        isNewBranch: bool,
        progress_dialog: CustomStatusDialog,
    ) -> GraphBranches:
        """
        Prepare the volume and run the RANSAC algorithm using the user parameters.

        Parameters
        ----------

        parameters: output of _getParametersRansac function, a list of user input parameters.
        centerline_resolution: maximum distance allowed between centerline points.
        graph_branches: object holding the graph of vessels branches.
        isNewBranch: flag to tell if it is the first branch or not.
        progress_dialog: UI window to inform the user on the state of the branch tracking.

        Returns
        ----------

        GraphBranches
        Updated graph
        """
        vol = slicer.util.array(parameters[0].GetID())
        vol = vol.swapaxes(0, 2)

        ijk_to_ras = vtk.vtkMatrix4x4()
        parameters[0].GetIJKToRASMatrix(ijk_to_ras)
        np_ijk_to_ras = np.zeros(shape=(4, 4))
        ijk_to_ras.DeepCopy(np_ijk_to_ras.ravel(), ijk_to_ras)

        vol = volume(vol, np_ijk_to_ras)

        starting_point = np.array([0, 0, 0])
        parameters[1].GetNthControlPointPosition(
            parameters[1].GetNumberOfControlPoints() - 1, starting_point
        )

        direction_point = np.array([0, 0, 0])
        parameters[2].GetNthControlPointPosition(
            parameters[2].GetNumberOfControlPoints() - 1, direction_point
        )

        graph_branches = run_ransac(
            vol,
            starting_point,
            direction_point,
            parameters[5],
            parameters[3],
            parameters[4],
            centerline_resolution,
            graph_branches,
            isNewBranch,
            progress_dialog,
        )

        return graph_branches


#
# pulmonary_arteries_segmentor_moduleTest
#


class pulmonary_arteries_segmentor_moduleTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_pulmonary_arteries_segmentor_module()

    def test_pulmonary_arteries_segmentor_module(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")
        self.delayDisplay("Test passed")
