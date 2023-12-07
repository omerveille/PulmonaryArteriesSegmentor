from __future__ import annotations
import subprocess
import sys
from importlib.util import find_spec

def check_installed(module: str | tuple[str, str]) -> None:
    """
    Check that the module is installed, if not install it
    :param module: str or tuple: module name as string or tuple of strings, depending if install name match module name
    """
    is_tuple = isinstance(module, tuple)
    if find_spec(module[0] if is_tuple else module) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', module[1] if is_tuple else module])

required_modules = ["numpy", "scipy", "trimesh", "pandas", ("nrrd", "pynrrd"), "nibabel"]

for module in required_modules:
    check_installed(module)
