from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec


import slicer
import time
from datetime import timedelta
import qt
import math

def make_custom_progress_bar(labelText="labelText", windowTitle="windowTitle", width=None, height=None):
    progress_bar = slicer.util.createProgressDialog(parent=slicer.util.mainWindow(), autoClose=False,
                                                    labelText=labelText,
                                                    windowTitle=windowTitle,
                                                    value=0)
    shape = (progress_bar.width if width is None else width, progress_bar.height if height is None else height)
    # Ensure the dialog is deleted when closed
    progress_bar.setAttribute(qt.Qt.WA_DeleteOnClose)
    progress_bar.setCancelButton(None)
    progress_bar.resize(*shape)
    progress_bar.show()
    slicer.app.processEvents()
    return progress_bar

class ProgressBarTimer:
    def __init__(self, total):
        self.total = total
        self.count = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        pass

    @classmethod
    def format_time(cls, seconds):
        return str(timedelta(seconds=int(seconds)))

    def update(self) -> tuple[float, float]:
        self.count += 1
        elapsed_time = time.time() - self.start_time
        remaining_time = (self.total - self.count) * (elapsed_time / self.count) if self.count > 0 else 0
        percent_done = math.floor(((self.count + 1) / self.total) * 100)
        return  elapsed_time, remaining_time, percent_done

class CustomStatusDialog:
    def __init__(self, windowTitle="windowTitle", text="text", width=None, height=None):
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle(windowTitle)
        # Ensure the dialog is deleted when closed
        dialog.setAttribute(qt.Qt.WA_DeleteOnClose)
        shape = (dialog.width if width is None else width, dialog.height if height is None else height)
        dialog.resize(*shape)
        layout = qt.QVBoxLayout()
        label = qt.QLabel(text)
        font = label.font
        font.setPointSize(14)
        label.setFont(font)
        # Center the text
        label.setAlignment(qt.Qt.AlignCenter)
        layout.addWidget(label)
        dialog.setLayout(layout)

        dialog.show()
        slicer.app.processEvents()

        self.label = label
        self.dialog = dialog

    def setText(self, text : str):
        self.label.setText(text)
        slicer.app.processEvents()

    def close(self):
        self.dialog.close()

def install_missing_module(modules: list[str | tuple[str, str]]) -> None:
    """
    Check that the module is installed, if not install it
    :param modules: list of str or tuple[str, str], modules to install
    """
    progress_bar = make_custom_progress_bar(labelText="Installing dependency...", windowTitle="Installing dependencies...", width=300)
    print("Installing missing dependencies...")
    for i, module in enumerate(modules):
        module_name = module[1] if isinstance(module, tuple) else module
        install_text = f"Installing {module_name}..."
        print(install_text)
        progress_bar.labelText = install_text
        slicer.app.processEvents()

        subprocess.check_call([sys.executable, '-m', 'pip', 'install', module_name])
        install_text = f"{module_name} installed !"
        print(install_text)
        progress_bar.labelText = install_text
        progress_bar.value = math.floor(((i + 1) / len(modules)) * 100)
        slicer.app.processEvents()

    progress_bar.close()

missing_modules = [module for module in ["numpy", "scipy", "trimesh", ("skimage", "scikit-image"), "networkx"] if find_spec(module[0] if isinstance(module, tuple) else module) is None]

if missing_modules:
    with slicer.util.tryWithErrorDisplay("Failed to install dependencies.", waitCursor=True):
        install_missing_module(missing_modules)