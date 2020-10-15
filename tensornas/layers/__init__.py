#!/usr/bin/env python
import os, pkgutil
from enum import Enum

__all__ = list(
    module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)])
)

SupportedLayers = Enum("SupportedLayers", {str.upper(i): i for i in __all__})
