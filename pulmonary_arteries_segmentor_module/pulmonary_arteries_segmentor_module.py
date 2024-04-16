import importlib
import math
import sys
from typing import Annotated, Optional

import numpy as np
import qt
import slicer
import slicer.util
import vtk
import json
from networkx.readwrite import json_graph
import networkx as nx
import skimage

from slicer import (vtkMRMLScalarVolumeNode, vtkMRMLMarkupsFiducialNode)
from slicer.ScriptedLoadableModule import (ScriptedLoadableModuleWidget, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModule, ScriptedLoadableModuleTest)
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer.util import VTKObservationMixin
from vtkSlicerMarkupsModuleMRMLPython import vtkMRMLMarkupsNode

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
from ransac_slicer.color_palettes import colors_float, direction_points_color
from ransac_slicer import make_custom_progress_bar, CustomStatusDialog
from ransac_slicer.volume import volume


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
        self.parent.categories = [
            "Segmentation"]
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
    params:

    inputVolume: input volume to extract the arteries from.
    startingPoint: starting point list for RANSAC cylinders.
    directionPoint: direction point list for RANSAC cylinders.
    percentInlierPoints: percentage of inlier points to validate a cylinder.
    percentThreshold: percentage of last cylinders radius to make a point inlier of a cylinder.
    """
    # Begin tab
    inputVolume: vtkMRMLScalarVolumeNode
    startingPoint: vtkMRMLMarkupsFiducialNode
    directionPoint: vtkMRMLMarkupsFiducialNode
    percentInlierPoints: Annotated[float, WithinRange(0, 100)] = 60.
    percentThreshold: Annotated[float, WithinRange(0, 100)] = 30.
    startingRadius: float


#
# pulmonary_arteries_segmentor_moduleWidget
#

class pulmonary_arteries_segmentor_moduleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/pulmonary_arteries_segmentor_module.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        if not hasattr(self, "segmentEditorNode") or self.segmentEditorNode != segmentEditorNode:
            self.segmentEditorNode = segmentEditorNode
            self.ui.SegmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = pulmonary_arteries_segmentor_moduleLogic()

        self.branch_tree = BranchTree()
        begin_tab = self.ui.tabWidget.widget(0)
        begin_tab.layout().insertWidget(6, self.branch_tree)

        self.graph_branches = GraphBranches(self.branch_tree, self.ui.showCenterlineButton ,self.ui.showContourPointsButton, self.ui.lockButton)
        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Sliders
        self.ui.centerlineTextSize.connect('valueChanged(double)', self.changeTextSize)

        # Buttons
        self.ui.placePointButton.connect('clicked(bool)', self.startPlacePointProcedure)

        self.ui.createBranch.connect('clicked(bool)', self.create_branch)
        self.ui.clearTree.connect('clicked(bool)',
                                  lambda: (self.graph_branches.clear_all(), self.updateSegmentationButtonState(), self._checkCanApply()))

        self.ui.exportTreeButton.connect('clicked(bool)', self.graph_branches.save_networkX)
        self.ui.loadTreeArchitectureButton.connect('clicked(bool)', self.onLoadTreeArchitecture)

        self.ui.paintButton.connect('clicked(bool)', self.onStartSegmentationButton)

        self.ui.lockButton.connect('clicked(bool)', self.onLockButton)
        self.ui.showCenterlineButton.connect('clicked(bool)', lambda: self.graph_branches.on_header_clicked(TreeColumnRole.VISIBILITY_CENTER))
        self.ui.showContourPointsButton.connect('clicked(bool)', lambda: self.graph_branches.on_header_clicked(TreeColumnRole.VISIBILITY_CONTOUR))

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
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

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
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[pulmonary_arteries_segmentor_moduleParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateSegmentationButtonState)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.checkCanPlacePoint)
            self._checkCanApply()

    def _getParametersRansac(self) -> list:
        return [self._parameterNode.inputVolume, self._parameterNode.startingPoint,
                self._parameterNode.directionPoint, self._parameterNode.percentInlierPoints,
                self._parameterNode.percentThreshold, self._parameterNode.startingRadius]

    def _checkCanApply(self, caller=None, event=None) -> None:
        starting_point = self._parameterNode.startingPoint
        direction_point = self._parameterNode.directionPoint

        if starting_point and not self.hasObserver(starting_point, vtkMRMLMarkupsNode.PointAddedEvent,
                                                   self._checkCanApply):
            self.addObserver(starting_point, vtkMRMLMarkupsNode.PointAddedEvent, self._checkCanApply)
            self.addObserver(starting_point, vtkMRMLMarkupsNode.PointRemovedEvent, self._checkCanApply)

        if direction_point and not self.hasObserver(direction_point, vtkMRMLMarkupsNode.PointAddedEvent,
                                                    self._checkCanApply):
            self.addObserver(direction_point, vtkMRMLMarkupsNode.PointAddedEvent, self._checkCanApply)
            self.addObserver(direction_point, vtkMRMLMarkupsNode.PointRemovedEvent, self._checkCanApply)

        if self._parameterNode and not self.isPlacingPoints and all(
                self._getParametersRansac()) and starting_point.GetNumberOfControlPoints() and direction_point.GetNumberOfControlPoints():
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
        self.ui.placePointButton.enabled = self._parameterNode.startingPoint and self._parameterNode.directionPoint

    def _addObserver(self, obj, event, fct):
        if not self.hasObserver(obj, event, fct):
            self.addObserver(obj, event, fct)

    def _removeObserver(self, obj, event, fct):
        if self.hasObserver(obj, event, fct):
            self.removeObserver(obj, event, fct)
    def startPlacePointProcedure(self):
        self.startingPointPlaced = False
        self.directionPointPlaced = False
        self.isPlacingPoints = True
        self.ui.createBranch.enabled = False

        starting_point = self._parameterNode.startingPoint
        starting_point.GetDisplayNode().SetSelectedColor(*direction_points_color)

        # Prepare the case where the user place the first point
        self._addObserver(starting_point, vtkMRMLMarkupsNode.PointPositionDefinedEvent , self.directionPointPlacement)

        # Prepare the case where the user cancel the point placement
        self._addObserver(starting_point, vtkMRMLMarkupsNode.PointRemovedEvent, self.resetPlacementState)

        # Start placing procedure
        slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(starting_point.GetID())
        slicer.modules.markups.logic().StartPlaceMode(1)

    def directionPointPlacement(self, *args):
        self.startingPointPlaced = True
        starting_point = self._parameterNode.startingPoint

        if not starting_point:
            return

        self._removeObserver(starting_point, vtkMRMLMarkupsNode.PointPositionDefinedEvent , self.directionPointPlacement)
        self._removeObserver(starting_point, vtkMRMLMarkupsNode.PointRemovedEvent, self.resetPlacementState)

        direction_point = self._parameterNode.directionPoint
        direction_point.GetDisplayNode().SetSelectedColor(*direction_points_color)

        self._addObserver(direction_point, vtkMRMLMarkupsNode.PointPositionDefinedEvent , self.validate_last_point)
        self._addObserver(direction_point, vtkMRMLMarkupsNode.PointRemovedEvent, self.resetPlacementState)

        # Place direction point
        slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(direction_point.GetID())
        slicer.modules.markups.logic().StartPlaceMode(1)
    def validate_last_point(self, *args):
        self.directionPointPlaced = True
        slicer.modules.markups.logic().StartPlaceMode(0)
        self.resetPlacementState()

    def resetPlacementState(self, *args):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()

        if interactionNode.GetPlaceModePersistence() == 1 and interactionNode.GetCurrentInteractionMode() == 1:
            # We do not go further because if those conditions are met, it means that the user moved cursor out of window
            return

        self.isPlacingPoints = False
        self._checkCanApply()

        starting_point = self._parameterNode.startingPoint
        self._removeObserver(starting_point, vtkMRMLMarkupsNode.PointPositionDefinedEvent , self.directionPointPlacement)
        self._removeObserver(starting_point, vtkMRMLMarkupsNode.PointRemovedEvent, self.resetPlacementState)

        direction_point = self._parameterNode.directionPoint
        self._removeObserver(direction_point, vtkMRMLMarkupsNode.PointPositionDefinedEvent , self.validate_last_point)
        self._removeObserver(direction_point, vtkMRMLMarkupsNode.PointRemovedEvent, self.resetPlacementState)

        if self.startingPointPlaced and not self.directionPointPlaced:
            starting_point.RemoveNthControlPoint(starting_point.GetNumberOfControlPoints() - 1)

        while starting_point.GetNumberOfControlPoints() >= 2:
            starting_point.RemoveNthControlPoint(0)

        while direction_point.GetNumberOfControlPoints() >= 2:
            direction_point.RemoveNthControlPoint(0)

        self.startingPointPlaced = False
        self.directionPointPlaced = False

    def recenter3dView(self) -> None:
        # Recenter the 3D view
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()

    def create_branch(self) -> None:
        with (slicer.util.tryWithErrorDisplay("Failed to compute segmentation.", waitCursor=True)):
            progress_dialog = CustomStatusDialog(windowTitle="Computing centerline...", text="Please wait", width=300, height=50)
            self.graph_branches = self.logic.processBranch(self._getParametersRansac(), self.graph_branches,
                                                           self.ui.createBranch.text == "Create New Branch", progress_dialog)

            self.recenter3dView()

            # Select the starting markup node to ease future node placement
            slicer.app.applicationLogic().GetSelectionNode().SetActivePlaceNodeID(self._parameterNode.startingPoint.GetID())

            self._checkCanApply()
            self.updateSegmentationButtonState()
            progress_dialog.close()

    def paintArteriesWithMarkup(self):
        progress_bar = make_custom_progress_bar(labelText="Please wait", windowTitle="Painting arteries...", width=250)

        segmentation = self.segmentationNode.GetSegmentation()

        for segment_id in self.arteriesSegmentIds:
            segmentation.RemoveSegment(segment_id)

        self.arteriesSegmentIds = [segmentation.AddEmptySegment("", name, colors_float[i % len(colors_float)]) for i, name in enumerate(self.graph_branches.names)]

        segmentEditorWidget = self.ui.SegmentEditorWidget
        segmentEditorNode = self.segmentEditorNode

        segmentEditorWidget.setSegmentationNode(self.segmentationNode)

        segmentEditorWidget.setActiveEffectByName("Logical operators")

        segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
        segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)

        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("BypassMasking", "1")
        effect.setParameter("Operation", "UNION")

        for center_line_idx in range(len(self.graph_branches.centers_lines)):
            segmentEditorNode.SetSelectedSegmentID(self.arteriesSegmentIds[center_line_idx])
            for point_idx in range(len(self.graph_branches.centers_lines[center_line_idx])):
                point_pos = self.graph_branches.centers_lines[center_line_idx][point_idx]
                radius = self.graph_branches.centers_line_radius[center_line_idx][point_idx]

                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(point_pos)
                sphere.SetRadius(radius)
                sphere.Update()

                tmp_segment_id = self.segmentationNode.AddSegmentFromClosedSurfaceRepresentation(sphere.GetOutput(),
                                                                                                 "tmp_segment", colors_float[center_line_idx % len(colors_float)])
                effect.setParameter("ModifierSegmentID", tmp_segment_id)
                effect.self().onApply()
                segmentation.RemoveSegment(tmp_segment_id)
            progress_bar.value = math.floor(((center_line_idx + 1) / len(self.graph_branches.centers_lines)) * 100)
            slicer.app.processEvents()

        segmentEditorWidget.setActiveEffectByName("No editing")

        # Hide and close progress bar
        progress_bar.close()

    def paintArteriesContours(self):
        progress_dialog = CustomStatusDialog(windowTitle="Painting contours...", text="Please wait", width=300, height=50)

        segmentation = self.segmentationNode.GetSegmentation()

        if self.contoursSegmentId is not None:
            segmentation.RemoveSegment(self.contoursSegmentId)

        progress_dialog.setText("Creating contour segment")
        slicer.app.processEvents()
        self.contoursSegmentId = segmentation.AddEmptySegment("", "Contours", [1., 215./255. ,0.])
        segmentationDisplayNode = self.segmentationNode.GetDisplayNode()
        segmentationDisplayNode.SetSegmentOpacity3D(self.contoursSegmentId, 0.1)

        binaryLabelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        binaryLabelmap.CreateDefaultDisplayNodes()

        segmentIdArg = vtk.vtkStringArray()
        for segment_id in self.arteriesSegmentIds:
            segmentIdArg.InsertNextValue(segment_id)

        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(self.segmentationNode, segmentIdArg,
                                                                          binaryLabelmap,
                                                                          self._parameterNode.inputVolume)

        numpy_labelmap_default = np.array(slicer.util.arrayFromVolume(binaryLabelmap) > 0, dtype=np.bool_)
        progress_dialog.setText("Computing the outer edge")
        slicer.app.processEvents()
        numpy_labelmap_ball_6 = skimage.morphology.binary_dilation(numpy_labelmap_default,
                                                                   skimage.morphology.ball(radius=6))
        progress_dialog.setText("Computing the inner edge")
        slicer.app.processEvents()
        numpy_labelmap_ball_6[
            skimage.morphology.binary_dilation(numpy_labelmap_default, skimage.morphology.ball(radius=4))] = False

        progress_dialog.setText("Updating segmentation")
        slicer.app.processEvents()
        slicer.util.updateVolumeFromArray(binaryLabelmap, numpy_labelmap_ball_6.astype(np.uint8))

        segmentIdArg = vtk.vtkStringArray()
        segmentIdArg.InsertNextValue(self.contoursSegmentId)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(binaryLabelmap, self.segmentationNode,
                                                                              segmentIdArg)
        progress_dialog.setText("Processing done !")
        slicer.app.processEvents()

        slicer.mrmlScene.RemoveNode(binaryLabelmap)

        # Hide and close progress bar
        progress_dialog.close()

    def changeTextSize(self, value):
        for markup in self.graph_branches.centers_line_markups:
            markup.GetDisplayNode().SetTextScale(value)

    def updateSegmentationButtonState(self, *args):
        paintButton: qt.QPushButton = self.ui.paintButton
        paintButton.enabled = len(self.graph_branches.centers_line_markups) != 0 and self._parameterNode.inputVolume

        if self.segmentationNode is None or self.segmentationNode.GetScene() is None:
            paintButton.text = "Create Segmentation from Branches"
            self.segmentationNode = None
            if self.nodeDeletionObserverTag is not None:
                slicer.mrmlScene.RemoveObserver(self.nodeDeletionObserverTag)
                self.nodeDeletionObserverTag = None
        else:
            paintButton.text = "Update Segmentation from Branches"

    def onStartSegmentationButton(self) -> None:
        with slicer.util.tryWithErrorDisplay("Failed to compute segmentation.", waitCursor=True):
            if self.segmentationNode is None or self.segmentationNode.GetScene() is None:
                self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                self.nodeDeletionObserverTag = slicer.mrmlScene.AddObserver(
                    slicer.vtkMRMLScene.NodeAboutToBeRemovedEvent, self.updateSegmentationButtonState)
                self.segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self._parameterNode.inputVolume)

                self.segmentationNode.SetName("Lung Segmentation")
                self.segmentationNode.CreateDefaultDisplayNodes()
                self.contoursSegmentId = None
                self.arteriesSegmentIds = []

            # We do pause the tracking of segmentation deletion
            slicer.mrmlScene.RemoveObserver(self.nodeDeletionObserverTag)
            self.nodeDeletionObserverTag = None

            # Paint the artery segment
            self.paintArteriesWithMarkup()
            # Paint the arteries Contours
            self.paintArteriesContours()

            # Make the segmentation visible
            if not self.segmentationNode.GetSegmentation().ContainsRepresentation("Closed surface"):
                self.segmentationNode.CreateClosedSurfaceRepresentation()
            self.segmentationNode.GetDisplayNode().SetVisibility3D(True)

            # Hide markup nodes
            for branch in self.graph_branches.centers_line_markups + self.graph_branches.contour_points_markups:
                branch.GetDisplayNode().SetVisibility(False)

            for branch in self.graph_branches.tree_widget._branchDict.values():
                branch.setIcon(TreeColumnRole.VISIBILITY_CENTER, Icons.visibleOff)
                branch.setIcon(TreeColumnRole.VISIBILITY_CONTOUR, Icons.visibleOff)
            self._parameterNode.startingPoint.GetDisplayNode().SetVisibility(False)
            self._parameterNode.directionPoint.GetDisplayNode().SetVisibility(False)

            self.updateSegmentationButtonState()

            # We track segmentation deletion
            self.nodeDeletionObserverTag = slicer.mrmlScene.AddObserver(
                slicer.vtkMRMLScene.NodeAboutToBeRemovedEvent, self.updateSegmentationButtonState)

    def onLockButton(self) -> None:
        button = self.ui.lockButton
        markups = self.graph_branches.centers_line_markups + self.graph_branches.contour_points_markups

        if button.checked:
            button.text = "Unlock Tree"
            for markup in markups:
                markup.LockedOn()
        else:
            button.text = "Lock Tree"
            for markup in markups:
                markup.LockedOff()

    def onLoadTreeArchitecture(self) -> None:
        dialog = qt.QFileDialog()
        file_path = dialog.getOpenFileName(None, "Choose a file", "", "JSON file (*.json)")

        # cancel any action if the user cancel / close the window / press escape
        if not file_path:
            return

        with (slicer.util.tryWithErrorDisplay("Failed to load tree architecture.", waitCursor=False)):
            with open(file_path) as f:
                js_graph = json.load(f)
            graph : nx.Graph = json_graph.node_link_graph(js_graph)

            if not self.graph_branches.clear_all():
                return

        with (slicer.util.tryWithErrorDisplay("Failed to restore tree architecture.", waitCursor=True)):
            # Restoring lists
            edge_name_table = {0: None}
            for a, b in graph.edges:
                self.graph_branches.branch_list.append([cylinder(center=np.array(cp)) for cp in graph[a][b]["center_line"]])
                self.graph_branches.names.append(graph[a][b]["name"])
                self.graph_branches.centers_lines.append(np.array(graph[a][b]["center_line"]))
                self.graph_branches.contours_points.append(graph[a][b]["contour_points"])
                # Recompute radius
                self.graph_branches.centers_line_radius.append([np.linalg.norm(np.array(graph[a][b]["contour_points"][k]) - np.array(graph[a][b]["center_line"][k]), axis=1).min() for k in range(len(graph[a][b]["center_line"]))])
                self.graph_branches.edges.append((a, b))
                self.graph_branches.create_new_markups(graph[a][b]["name"], np.array(graph[a][b]["center_line"]), graph[a][b]["contour_points"])
                edge_name_table[b] = graph[a][b]["name"]


            for node in graph.nodes(data=True):
                self.graph_branches.nodes.append(node[1]["pos"])

            for a, b in nx.edge_dfs(graph):
                current_edge_name = graph[a][b]["name"]
                parent_edge_name = edge_name_table[a]
                self.graph_branches.tree_widget.insertAfterNode(nodeId=current_edge_name, parentNodeId=parent_edge_name)
            self._checkCanApply()
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
        return pulmonary_arteries_segmentor_moduleParameterNode(super().getParameterNode())

    def processBranch(self, params: list, graph_branches: GraphBranches, isNewBranch: bool, progress_dialog: CustomStatusDialog) -> None:
        """
        def run_ransac(vol, input_centers_curve_path, output_centers_curve_path, input_contour_point_path,
         output_contour_point_path, starting_point, direction_point, starting_radius, pct_inlier_points, threshold):
        """

        vol = slicer.util.array(params[0].GetID())
        vol = vol.swapaxes(0, 2)

        ijk_to_ras = vtk.vtkMatrix4x4()
        params[0].GetIJKToRASMatrix(ijk_to_ras)
        np_ijk_to_ras = np.zeros(shape=(4, 4))
        ijk_to_ras.DeepCopy(np_ijk_to_ras.ravel(), ijk_to_ras)

        vol = volume(vol, np_ijk_to_ras)

        starting_point = np.array([0, 0, 0])
        params[1].GetNthControlPointPosition(params[1].GetNumberOfControlPoints()-1, starting_point)

        direction_point = np.array([0, 0, 0])
        params[2].GetNthControlPointPosition(params[2].GetNumberOfControlPoints()-1, direction_point)

        graph_branches = run_ransac(vol, starting_point, direction_point, params[5],
                                    params[3], params[4], graph_branches, isNewBranch, progress_dialog)

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
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_pulmonary_arteries_segmentor_module()

    def test_pulmonary_arteries_segmentor_module(self):
        """ Ideally you should have several levels of tests.  At the lowest level
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
        self.delayDisplay('Test passed')
