# PulmonaryArteriesSegmentor, a pulmonary artery 3D Slicer segmentation and annotation plugin
<!-- TOC -->
* [PulmonaryArteriesSegmentor, a pulmonary artery 3D Slicer segmentation and annotation plugin](#pulmonaryarteriessegmentor-a-pulmonary-artery-3d-slicer-segmentation-and-annotation-plugin)
  * [Introduction 📜](#introduction-)
  * [Installation 📂](#installation-)
  * [Usage of the plugin 💡](#usage-of-the-plugin-)
    * [Overview 🌐](#overview-)
    * [The centerline tab ➰](#the-centerline-tab-)
    * [The segmentation tab 🧩](#the-segmentation-tab-)
  * [Algorithms involved](#algorithms-involved)
    * [Centerline ➰](#centerline-)
    * [Segmentation 🧩](#segmentation-)
  * [For developers 👩‍💻👨‍💻](#for-developers-)
    * [Setup pre-commit 🏗️](#setup-pre-commit-)
    * [Enforce pre-commit to run 🏃](#enforce-pre-commit-to-run-)
<!-- TOC -->
## Introduction 📜
PulmonaryArteriesSegmentor is a 3D Slicer plugin that aims to ease the segmentation and annotation of the pulmonary arteries for angiography images.
The segmentation and annotation process is divided into three steps:
1) Place points to indicate the direction of the vessel you would like to segment.
2) Start a centerline segmentation from points, and repeat steps 1) and 2) until you have all the vessels you desire.
3) Once you have all the vessels you want, paint the segmentation to generate an initialization of region-growing.

This plugin is an end-of-study project, made by Azéline Aillet (Student at EPITA) and Gabriel Jacquinot (Student at EPITA), under the direction of Odyssée Merveille (CREATIS) and Morgane Des Ligneris (CREATIS).

The RANSAC code is based on the previous work of Jack CARBONERO (CReSTIC), Guillaume DOLLE (LMR) and Nicolas PASSAT (CReSTIC) on the plugin vestract.

The hierarchy code is based on the work of Lucie Macron (Kitware SAS), Thibault Pelletier (Kitware SAS), Camille Huet (Kitware SAS), Leo Sanchez (Kitware SAS) from the RVesselX plugin.

## Installation 📂
To install the plugin, you have to do the following steps:

1) Download the plugin as a zip file or clone this repository. Extract it if you downloaded a zip, you can clone / extract it wherever you want.
2) Start 3D Slicer, in the `module` drop-down menu, click on `Developer Tools -> Extension Wizard`.
3) Click on `Select Extension`, a folder tab will appear, select the repository folder / the zip folder you extracted, and click on `Choose`.
4) A popup will show, select `Yes`.⚠️ Make sure you are connected to the internet. When loading the module, it will check if you have all the dependencies and attempt to download the missing ones. ⚠️
5) After downloading the missing dependencies, the plugin can be found in the module drop-down menu, under the `Segmentation` category.

Tips: If you struggle to find any of those modules, you can use the magnifying glass to search modules by name.

## Usage of the plugin 💡
### Overview 🌐
The plugin is divided into two tabs:
- **Centerline** which is used to create a tree from the vessels' centerline.
- **Segmentation** which creates a segmentation and fills it according to the centerline tree.

### The centerline tab ➰
In order to make a segmentation, you first have to create a vessel centerline tree.
To do so, you have to go inside the centerline tab. Inside it, you must select a volume, a starting point, and a direction point list.

To select a branch centerline, you must, once you have met the previous requirements, place two points using the `Place a new starting and direction point` button.
Those points will indicate in which direction the algorithm will travel in order to detect a centerline.

Select a radius for the vessel, and then click on the `Create root` / `Create new branch` button in order to start finding a centerline.

Once the processing is done, you can add new branches to the tree by adding two new starting and direction points by doing the same process.

You can delete a branch from the hierarchy by clicking on the bin icon, or clear the whole tree by clicking on the `Clear tree` button.
If the centerline is correct until a certain point where it is not, you can delete the irrelevant part by selecting the last correct point and right-clicking the concerned artery in the hierarchy, and selecting `Remove end of the branch`.

If you want, you can also save the hierarchy of all the vessel points into a networkX graph, which is exported in `.json` and `.pickle` files.
### The segmentation tab 🧩
The segmentation tab is pretty straightforward, you have a button `Create segmentation from branches` / `Update segmentation from branches`, and when you click on it, it creates a new segmentation, which is an initialization for the 3D slicer's region-growing.
You can always add segments / edit the segmentation generated as you want in the segmentation widget.
To have a finalize segmentation, you just have to run the region-growing of slicer which can be found in the plugin directly.

To finalize segmentation, you just have to run the region-growing of slicer which can be found in the plugin directly.

## Algorithms involved
### Centerline ➰
In this part, the main algorithm used is RANSAC. Basically, we take a set of points around the starting and direction points and try to fit a cylinder. If the set of points is a good candidate for a cylinder, then we keep this cylinder and iterate to find a new one with a certain width and height proportional to the last cylinder in a predefined range of angles. The algorithm stops eventually when there are no more fitting cylinders to add.

When adding a new branch, we simply rerun a RANSAC, find the closest point to the starting point, and reorder the branches so that they form a tree.

### Segmentation 🧩
In order to draw each segment starting seed for the segmentation, we simply iterate through each point of the centerline, and for each point, we draw a sphere of the radius of the vessel. We underestimate the radius so that the segmentation can find the accurate radius. The radius of a vessel is the distance between the closest contour point and the centerline.

For the stopping edge, we use the labelmap representation of the segmentation. We take all the segments and apply two morphological dilations with a sphere of sizes 4 and 6. After that, we subtract the result of the dilation of size 4 from the result of size 6. At the end, we obtain an edge that surrounds our vessels and acts as a stopper point for a the 3D slicer's region-growing algorithm.

## For developers 👩‍💻👨‍💻
### Setup pre-commit 🏗️

```shell
pip install pre-commit
pre-commit install
```

### Enforce pre-commit to run 🏃

```shell
pre-commit run --all-files
```