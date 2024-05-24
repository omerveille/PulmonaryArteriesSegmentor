from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec

from .popup_utils import make_custom_progress_bar
import slicer
import math


def install_missing_module(modules: list[str | tuple[str, str]]) -> None:
    """
    Check that the module is installed, if not install it
    :param modules: list of str or tuple[str, str], modules to install
    """
    progress_bar = make_custom_progress_bar(
        labelText="Installing dependency...",
        windowTitle="Installing dependencies...",
        width=300,
    )
    print("Installing missing dependencies...")
    for i, module in enumerate(modules):
        module_name = module[1] if isinstance(module, tuple) else module
        install_text = f"Installing {module_name}..."
        print(install_text)
        progress_bar.labelText = install_text
        slicer.app.processEvents()

        subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
        install_text = f"{module_name} installed !"
        print(install_text)
        progress_bar.labelText = install_text
        progress_bar.value = math.floor(((i + 1) / len(modules)) * 100)
        slicer.app.processEvents()

    progress_bar.close()


missing_modules = [
    module
    for module in ["numpy", "scipy", "trimesh", ("skimage", "scikit-image"), "networkx"]
    if find_spec(module[0] if isinstance(module, tuple) else module) is None
]

if missing_modules:
    with slicer.util.tryWithErrorDisplay(
        "Failed to install dependencies.", waitCursor=True
    ):
        install_missing_module(missing_modules)
