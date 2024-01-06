import json
import os
from typing import Annotated, Optional

import numpy as np
import vtk

import qt
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModuleWidget, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModule, ScriptedLoadableModuleTest)
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer.parameterNodeWrapper import *

from slicer import (vtkMRMLScalarVolumeNode, vtkMRMLMarkupsCurveNode, vtkMRMLMarkupsFiducialNode, qMRMLSubjectHierarchyTreeView)
import tempfile

import networkx as nx
from ransac_slicer.ransac import run_ransac
from ransac_slicer.graph_branches import Graph_branches
from ransac_slicer.branch_tree import Branch_tree

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
        self.parent.contributors = ["Gabriel Jacquinot", "Azéline Aillet"]  # TODO: replace with "Firstname Lastname (Organization)"
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

    # Segmentation tab
    valueInflation: Annotated[float, WithinRange(-100, 100)] = 0
    valueCurvature: Annotated[float, WithinRange(-100, 100)] = 70
    valueAttractionGradient: Annotated[float, WithinRange(-100, 100)] = 50
    valueIterations: float = 10

    segmentationStrategy: Annotated[str, Choice(["One vessel per branch", "One vessel per parent child", "One vessel per parent and sub child", "One vessel for whole tree"])] = "One vessel per branch"
    segmentationInitialization: Annotated[str, Choice(["Colliding Fronts", "Fast Marching"])] = "Colliding Fronts"
    segmentationMethod: Annotated[str, Choice(["Geodesic", "Curves"])] = "Geodesic"


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

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = pulmonary_arteries_segmentor_moduleLogic()

        self.branch_tree = Branch_tree()
        begin_tab = self.ui.tabWidget.widget(0)
        begin_tab.layout().insertWidget(6, self.branch_tree)
        # self.branch_tree.addTopLevelItem(Branch_tree_item("test"))

        self.graph_branches = Graph_branches(self.branch_tree)
        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.createRoot.connect('clicked(bool)', self.createRoot)
        self.ui.createNewBranch.connect('clicked(bool)', self.createNewBranch)
        self.ui.clearTree.connect('clicked(bool)', self.graph_branches.clearAll)
        self.ui.saveTree.connect('clicked(bool)', self.graph_branches.saveNetworkX)

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
        if self._parameterNode and all(self._getParametersBegin()) and len(self.graph_branches.names) == 0:
            self.ui.createRoot.toolTip = "Create root"
            self.ui.createRoot.enabled = True
        else:
            self.ui.createRoot.toolTip = "Select all input before creating root"
            self.ui.createRoot.enabled = False

        if self._parameterNode and all(self._getParametersBegin()) and len(self.graph_branches.names) != 0:
            self.ui.createNewBranch.toolTip = "Create new branch"
            self.ui.createNewBranch.enabled = True
        else:
            self.ui.createNewBranch.toolTip = "Create root before creating other branches"
            self.ui.createNewBranch.enabled = False

        if len(self.graph_branches.names) != 0:
            self.ui.clearTree.toolTip = "Compute new branch"
            self.ui.clearTree.enabled = True
            self.ui.saveTree.toolTip = "Compute new branch"
            self.ui.saveTree.enabled = True
        else:
            self.ui.clearTree.toolTip = "Tree is already empty"
            self.ui.clearTree.enabled = False
            self.ui.saveTree.toolTip = "There is nothing to save"
            self.ui.saveTree.enabled = False

        # TODO FOR tab branches

    def createRoot(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            # Compute output
            self.graph_branches = self.logic.processBranch(self._getParametersBegin(), self.graph_branches, False)
            self._checkCanApply()
    
    def createNewBranch(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            # Compute output
            self.graph_branches = self.logic.processBranch(self._getParametersBegin(), self.graph_branches, True)


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

    def processBranch(self, params: list, graph_branches: Graph_branches, isNewBranch: bool) -> None:
        # [self._parameterNode.inputVolume, self._parameterNode.startingPoint, self._parameterNode.directionPoint, self._parameterNode.percentInlierPoints, self._parameterNode.percentThreshold, self._parameterNode.startingRadius]
        """
        def run_ransac(input_volume_path, input_centers_curve_path, output_centers_curve_path, input_contour_point_path,
         output_contour_point_path, starting_point, direction_point, starting_radius, pct_inlier_points, threshold):
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            _, input_volume_path = tempfile.mkstemp(prefix="input_volume_", suffix=".nrrd", dir=tmpdirname)
            slicer.util.exportNode(params[0], input_volume_path)

            starting_point = np.array([0, 0, 0])
            params[1].GetNthControlPointPosition(0, starting_point)

            direction_point = np.array([0, 0, 0])
            params[2].GetNthControlPointPosition(0, direction_point)
            
            graph_branches = run_ransac(input_volume_path, starting_point, direction_point, params[5],
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
