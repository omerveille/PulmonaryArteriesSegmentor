import importlib
import os
import sys
from typing import Annotated, Optional

import numpy as np
import slicer
import SimpleITK as sitk
import vtk

from slicer.ScriptedLoadableModule import (ScriptedLoadableModuleWidget, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModule, ScriptedLoadableModuleTest)
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer.util import VTKObservationMixin
from vtkSlicerMarkupsModuleMRMLPython import vtkMRMLMarkupsNode
from slicer import (vtkMRMLScalarVolumeNode, vtkMRMLMarkupsFiducialNode)

try:
    to_reload = [key for key in sys.modules.keys() if "ransac_slicer." in key]
    for file_to_reload in to_reload:
        sys.modules[file_to_reload] = importlib.reload(sys.modules[file_to_reload])
except Exception as e:
    print(f"Exception occurred while reloading\n{e}")

from ransac_slicer.ransac import run_ransac
from ransac_slicer.graph_branches import GraphBranches
from ransac_slicer.branch_tree import BranchTree, TreeColumnRole, Icons
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
        self.parent.title = "Pulmonary Arteries Segmentor"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "Segmentation"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Gabriel Jacquinot",
                                    "Azéline Aillet"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#pulmonary_arteries_segmentor_module">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # pulmonary_arteries_segmentor_module1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='pulmonary_arteries_segmentor_module',
        sampleName='pulmonary_arteries_segmentor_module1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'pulmonary_arteries_segmentor_module1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='pulmonary_arteries_segmentor_module1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='pulmonary_arteries_segmentor_module1'
    )

    # pulmonary_arteries_segmentor_module2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='pulmonary_arteries_segmentor_module',
        sampleName='pulmonary_arteries_segmentor_module2',
        thumbnailFileName=os.path.join(iconsPath, 'pulmonary_arteries_segmentor_module2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='pulmonary_arteries_segmentor_module2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='pulmonary_arteries_segmentor_module2'
    )


#
# pulmonary_arteries_segmentor_moduleParameterNode
#

@parameterNodeWrapper
class pulmonary_arteries_segmentor_moduleParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
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
        begin_tab.layout().insertWidget(5, self.branch_tree)
        # self.branch_tree.addTopLevelItem(Branch_tree_item("test"))

        self.graph_branches = GraphBranches(self.branch_tree)
        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.createBranch.connect('clicked(bool)', self.create_branch)
        self.ui.clearTree.connect('clicked(bool)', self.graph_branches.clear_all)
        self.ui.clearTree.connect('clicked(bool)', self._checkCanApply)
        self.ui.saveTree.connect('clicked(bool)', self.graph_branches.save_networkX)
        self.ui.paintButton.connect('clicked(bool)', self.onStartSegmentationButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

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
            self._checkCanApply()

    def _getParametersBegin(self) -> list:
        return [self._parameterNode.inputVolume, self._parameterNode.startingPoint,
                self._parameterNode.directionPoint, self._parameterNode.percentInlierPoints,
                self._parameterNode.percentThreshold, self._parameterNode.startingRadius]


    def _getParametersSegmentation(self) -> list:
        return [self._parameterNode.valueInflation, self._parameterNode.valueCurvature,
                self._parameterNode.valueAttractionGradient, self._parameterNode.valueIterations,
                self._parameterNode.segmentationStrategy, self._parameterNode.segmentationInitialization,
                self._parameterNode.segmentationMethod]

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

        if self._parameterNode and all(
                self._getParametersBegin()) and starting_point.GetNumberOfControlPoints() and direction_point.GetNumberOfControlPoints():
            self.ui.createBranch.enabled = True
            if len(self.graph_branches.names) == 0:
                self.ui.createBranch.text = "Create root"
                self.ui.createBranch.toolTip = "Create root"
            else:
                self.ui.createBranch.text = "Create new branch"
                self.ui.createBranch.toolTip = "Create new branch"
        else:
            self.ui.createBranch.toolTip = "Select all input before creating branch"
            self.ui.createBranch.enabled = False

        if len(self.graph_branches.names) != 0:
            self.ui.clearTree.toolTip = "Clear all tree"
            self.ui.clearTree.enabled = True
            self.ui.saveTree.toolTip = "Save graph tree"
            self.ui.saveTree.enabled = True
        else:
            self.ui.clearTree.toolTip = "Tree is already empty"
            self.ui.clearTree.enabled = False
            self.ui.saveTree.toolTip = "There is nothing to save"
            self.ui.saveTree.enabled = False

    def create_branch(self) -> None:
        with slicer.util.tryWithErrorDisplay("Failed to compute segmentation.", waitCursor=True):
            # Compute output
            progress_bar = slicer.util.createProgressDialog(parent=slicer.util.mainWindow(), autoClose=False,
                                                            labelText="Please wait", windowTitle="Running RANSAC...",
                                                            value=0)
            progress_bar.setCancelButton(None)
            slicer.app.processEvents()
            self.old_graph_branches = self.graph_branches
            self.graph_branches = self.logic.processBranch(self._getParametersBegin(), self.graph_branches, self.ui.createBranch.text == "Create new branch")
            progress_bar.hide()
            progress_bar.close()
            self._checkCanApply()


    def paintArteriesWithMarkup(self):
        # Create a progress bar
        progress_bar = slicer.util.createProgressDialog(parent=slicer.util.mainWindow(), autoClose=False,
                                                        labelText="Please wait",
                                                        windowTitle="Painting arteries with markup...",
                                                        value=0)
        progress_bar.setCancelButton(None)
        slicer.app.processEvents()

        segmentation = self.segmentationNode.GetSegmentation()

        segment = segmentation.GetSegment(self.arteriesSegmentId)
        name = segment.GetName()
        color = segment.GetColor()

        segmentation.RemoveSegment(self.arteriesSegmentId)
        self.arteriesSegmentId = segmentation.AddEmptySegment("", name, color)
        segment = segmentation.GetSegment(self.arteriesSegmentId)

        segmentEditorWidget = self.ui.SegmentEditorWidget
        segmentEditorNode = self.segmentEditorNode

        segmentEditorWidget.setSegmentationNode(self.segmentationNode)
        segmentEditorNode.SetSelectedSegmentID(self.arteriesSegmentId)

        segmentEditorWidget.setActiveEffectByName("Logical operators")

        segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
        segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)

        for markup_node_idx in range(len(self.graph_branches.centers_line_markups)):

            markupsNode = self.graph_branches.centers_line_markups[markup_node_idx]
            for point_idx in range(markupsNode.GetNumberOfControlPoints()):
                point_pos = [0, 0, 0]
                markupsNode.GetNthControlPointPosition(point_idx, point_pos)
                radius = self.graph_branches.centers_line_radius[markup_node_idx][point_idx]

                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(point_pos)
                sphere.SetRadius(radius)
                sphere.Update()

                tmp_segment_id = self.segmentationNode.AddSegmentFromClosedSurfaceRepresentation(sphere.GetOutput(),
                                                                                                 "tmp_segment", color)
                effect = segmentEditorWidget.activeEffect()
                effect.setParameter("BypassMasking", "1")
                effect.setParameter("ModifierSegmentID", tmp_segment_id)
                effect.setParameter("Operation", "UNION")
                effect.self().onApply()
                segmentation.RemoveSegment(tmp_segment_id)

        slicer.modules.segmentations.logic().SetSegmentStatus(segment, 0)
        segmentEditorWidget.setActiveEffectByName("No editing")
        segmentEditorNode.SetSelectedSegmentID(self.otherSegmentId)

        # Hide and close progress bar
        progress_bar.hide()
        progress_bar.close()

    def paintLungs(self):
        # Create a progress bar
        progress_bar = slicer.util.createProgressDialog(parent=slicer.util.mainWindow(), autoClose=False,
                                                        labelText="Please wait",
                                                        windowTitle="Painting lungs...",
                                                        value=0)
        progress_bar.setCancelButton(None)
        slicer.app.processEvents()

        def GetSlicerITKReadWriteAddress(myNode):
            myNodeSceneAddress = myNode.GetScene().GetAddressAsString("").replace('Addr=', '')
            myNodeSceneID = myNode.GetID()
            myNodeFullITKAddress = 'slicer:' + myNodeSceneAddress + '#' + myNodeSceneID
            return myNodeFullITKAddress

        otsu_filter = sitk.OtsuMultipleThresholdsImageFilter()

        otsu_filter.SetNumberOfHistogramBins(500)
        otsu_filter.SetNumberOfThresholds(1)
        otsu_filter.SetValleyEmphasis(False)

        img = sitk.ReadImage(GetSlicerITKReadWriteAddress(self._parameterNode.inputVolume))
        labelmap = otsu_filter.Execute(img)

        erode_filter = sitk.BinaryDilateImageFilter()
        erode_filter.SetKernelType(sitk.sitkBall)
        erode_filter.SetKernelRadius(3)
        erode_filter.SetForegroundValue(1)
        erode_filter.SetBackgroundValue(0)

        labelmap = erode_filter.Execute(labelmap)
        labelmap = sitk.BinaryNot(labelmap)

        size = labelmap.GetSize()

        # Get corner points
        corners = [(0, 0, 0), (0, 0, size[2] - 1), (0, size[1] - 1, 0), (0, size[1] - 1, size[2] - 1),
                   (size[0] - 1, 0, 0), (size[0] - 1, 0, size[2] - 1), (size[0] - 1, size[1] - 1, 0),
                   (size[0] - 1, size[1] - 1, size[2] - 1)]

        cc_filter = sitk.ConnectedComponentImageFilter()
        labelmap = cc_filter.Execute(labelmap)

        # 0 is the background so we do not care about removing it
        connected_components_to_remove = {labelmap[corner] for corner in corners if labelmap[corner] != 0}

        for cc_to_remove in connected_components_to_remove:
            to_remove = sitk.BinaryThreshold(labelmap, cc_to_remove, cc_to_remove, cc_to_remove, 0)
            to_remove = sitk.Cast(to_remove, labelmap.GetPixelID())
            labelmap = sitk.SubtractImageFilter().Execute(labelmap, to_remove)

        labelmap = cc_filter.Execute(labelmap)
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(labelmap)
        cc_sizes = [label_stats.GetNumberOfPixels(label) for label in range(1, label_stats.GetNumberOfLabels() + 1)]
        labels_of_largest_cc = (np.argsort(cc_sizes)[::-1][:2] + 1).astype(float)
        labelmap = sitk.AddImageFilter().Execute(sitk.BinaryThreshold(labelmap, labels_of_largest_cc[0], labels_of_largest_cc[0], 1, 0),
                                       sitk.BinaryThreshold(labelmap, labels_of_largest_cc[1], labels_of_largest_cc[1], 1, 0))


        labelMapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
        labelMapNode.CreateDefaultDisplayNodes()

        sitk.WriteImage(labelmap, GetSlicerITKReadWriteAddress(labelMapNode))

        lungSegmentId = vtk.vtkStringArray()
        lungSegmentId.InsertNextValue(self.lungsSegmentId)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelMapNode, self.segmentationNode, lungSegmentId)
        slicer.mrmlScene.RemoveNode(labelMapNode)


        # Hide and close progress bar
        progress_bar.hide()
        progress_bar.close()



    def onStartSegmentationButton(self) -> None:
        with slicer.util.tryWithErrorDisplay("Failed to compute segmentation.", waitCursor=True):
            if self.segmentationNode is None:
                self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                self.segmentationNode.SetName("Lung Segmentation")
                self.segmentationNode.CreateDefaultDisplayNodes()
                self.lungsSegmentId = self.segmentationNode.GetSegmentation().AddEmptySegment("", "Lungs",
                                                                                              [128. / 255., 174. / 255,
                                                                                               128. / 255.])
                self.arteriesSegmentId = self.segmentationNode.GetSegmentation().AddEmptySegment("", "Arteries",
                                                                                                 [216. / 255., 101. / 255,
                                                                                                  79. / 255.])
                self.otherSegmentId = self.segmentationNode.GetSegmentation().AddEmptySegment("", "Other",
                                                                                              [230. / 255., 220. / 255,
                                                                                               70. / 255.])
                segmentationDisplayNode = self.segmentationNode.GetDisplayNode()
                segmentationDisplayNode.SetSegmentOpacity3D(self.lungsSegmentId, 0.1)
                segmentationDisplayNode.SetSegmentOpacity3D(self.otherSegmentId, 0.1)

            if self._parameterNode.inputVolume:
                self.segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self._parameterNode.inputVolume)

            if self.graph_branches and len(self.graph_branches.centers_line_markups):

                # Paint the artery segment
                self.paintArteriesWithMarkup()
                # Paint the lungs segment
                self.paintLungs()

                # Make the segmentation visible
                if not self.segmentationNode.GetSegmentation().ContainsRepresentation("Closed surface"):
                    self.segmentationNode.CreateClosedSurfaceRepresentation()
                self.segmentationNode.GetDisplayNode().SetVisibility3D(True)

                # Hide markup nodes
                for markup in [self.graph_branches.centers_line_markups, self.graph_branches.contour_points_markups]:
                    for branch in markup:
                        branch.GetDisplayNode().SetVisibility(False)

                for icon in self.graph_branches.tree_widget._branchDict.values():
                    icon.setIcon(TreeColumnRole.VISIBILITY_CENTER, Icons.visibleOff)
                    icon.setIcon(TreeColumnRole.VISIBILITY_CONTOUR, Icons.visibleOff)
                self._parameterNode.startingPoint.GetDisplayNode().SetVisibility(False)
                self._parameterNode.directionPoint.GetDisplayNode().SetVisibility(False)

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

    def processBranch(self, params: list, graph_branches: GraphBranches, isNewBranch: bool) -> None:
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
        params[1].GetNthControlPointPosition(0, starting_point)

        direction_point = np.array([0, 0, 0])
        params[2].GetNthControlPointPosition(0, direction_point)

        graph_branches = run_ransac(vol, starting_point, direction_point, params[5],
                                    params[3], params[4], graph_branches, isNewBranch)

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
        self.test_pulmonary_arteries_segmentor_module1()

    def test_pulmonary_arteries_segmentor_module1(self):
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

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('pulmonary_arteries_segmentor_module1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = pulmonary_arteries_segmentor_moduleLogic()

        # Test algorithm with non-inverted threshold
        logic.processBranch(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.processBranch(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
