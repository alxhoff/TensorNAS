from setuptools import find_packages
from pkgutil import iter_modules
import os
from enum import Enum


def find_block_architectures():
    import tensornas.blocktemplates.blockarchitectures as BT

    modules = []
    BT_pkg = BT.__name__
    BT_dir = os.path.dirname(BT.__file__)
    # modules.append(find_modules(BT_pkg, BT_dir))
    for mod in iter_modules([BT_dir]):
        if not mod.ispkg:
            mod_name = mod.name
            modules.append(mod_name)

    for pkg in find_packages(BT_dir):
        pkg_path = BT_dir + "/" + pkg.replace(".", "/")
        # modules.append(find_modules(BT_pkg + "." + pkg, pkg_path))
        for mod in iter_modules([pkg_path]):
            if not mod.ispkg:
                mod_name = mod.name
                modules.append(mod_name)

    return modules


def find_blocks():
    import tensornas.blocktemplates.subblocks as BT

    modules = []
    BT_pkg = BT.__name__
    BT_dir = os.path.dirname(BT.__file__)
    # modules.append(find_modules(BT_pkg, BT_dir))
    for mod in iter_modules([BT_dir]):
        if not mod.ispkg:
            mod_name = mod.name
            modules.append(mod_name)

    for pkg in find_packages(BT_dir):
        pkg_path = BT_dir + "/" + pkg.replace(".", "/")
        # modules.append(find_modules(BT_pkg + "." + pkg, pkg_path))
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
