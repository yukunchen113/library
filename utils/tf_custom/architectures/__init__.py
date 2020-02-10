import os
import glob
modules = glob.glob(os.path.join(os.path.dirname(__file__), "*"))
modules = filter(lambda filename: (
	os.path.isdir(filename) and not filename.endswith("__pycache__")) or (
	filename.endswith(".py") and os.path.isfile(filename) and not 
	filename.endswith("__init__.py")), modules)
modules = list(map(lambda f: os.path.splitext(os.path.basename(f))[0], modules))
__all__ = modules
from . import *