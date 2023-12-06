cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(PulmonaryArteriesSegmentor)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/PulmonaryArteriesSegmentor")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Gabriel Jacquinot")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/PulmonaryArteriesSegmentor.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/PulmonaryArteriesSegmentor/Screenshots/1.png")
set(EXTENSION_DEPENDS "SlicerVMTK") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(pulmonary_arteries_segmentor_module)
add_subdirectory(DefaultModule)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})