#!/usr/bin/env python
from enum import Enum

from setuptools import find_packages

from TensorNAS.Tools import find_modules


def find_layer_modules():
    from TensorNAS import Layers as Layers
    import itertools
    import os

    modules = []
    layer_pkg = Layers.__name__
    layer_dir = os.path.dirname(Layers.__file__)
    modules.append(find_modules(layer_pkg, layer_dir))

    for pkg in find_packages(layer_dir):
        pkg_path = layer_dir + "/" + pkg.replace(".", "/")
        modules.append(find_modules(layer_pkg + "." + pkg, pkg_path))

    return list(itertools.chain(*modules))


LayerModules = find_layer_modules()
LayerNames = [(lambda i: i.__name__.split(".")[-1])(i) for i in LayerModules]
Layers = Enum(
    "Layers", {str.upper(layer): mod for layer, mod in zip(LayerNames, LayerModules)}
)
SupportedLayers = Enum("SupportedLayers", {str.upper(i): i for i in LayerNames})
