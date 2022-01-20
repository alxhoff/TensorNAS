from setuptools import find_packages
from pkgutil import iter_modules
import os
from enum import Enum


def find_block_architectures():
    from TensorNAS.Blocks import BlockArchitectures as BT

    modules = []
    BT_dir = os.path.dirname(BT.__file__)
    for mod in iter_modules([BT_dir]):
        if not mod.ispkg:
            mod_name = mod.name
            modules.append(mod_name)

    for pkg in find_packages(BT_dir):
        pkg_path = BT_dir + "/" + pkg.replace(".", "/")
        for mod in iter_modules([pkg_path]):
            if not mod.ispkg:
                mod_name = mod.name
                modules.append(mod_name)

    return modules


def find_blocks():
    from TensorNAS.Blocks import SubBlocks as BT

    modules = []
    BT_dir = os.path.dirname(BT.__file__)
    for mod in iter_modules([BT_dir]):
        if not mod.ispkg:
            mod_name = mod.name
            modules.append(mod_name)

    for pkg in find_packages(BT_dir):
        pkg_path = BT_dir + "/" + pkg.replace(".", "/")
        for mod in iter_modules([pkg_path]):
            if not mod.ispkg:
                mod_name = mod.name
                modules.append(mod_name)

    return modules


ArchitectureModules = find_block_architectures()
ArchitectureModules = find_blocks()

SupportedBlocks = Enum(
    "SupportedBlocks", {str.upper(i): i for i in ArchitectureModules}
)
SupportedArchitectureBlocks = Enum(
    "SupportedArchitectureBlocks", {str.upper(i): i for i in ArchitectureModules}
)
