# -*- coding: utf-8 -*-
"""
The "distortpy" package is intended for providing functions for distortion of images.

@author: Sergei Klykov
@licence: MIT, @year: 2023
"""
if __name__ == "__main__":
    # use absolute imports for importing as module
    __all__ = ['...']  # for specifying from distortpy import * if package imported from some script
elif __name__ == "distortpy":
    pass  # do not add module "..." to __all__ attribute, because it demands to construct explicit path

# Automatically bring the main class and some methods to the name space when one of import command is used commands:
# 1) from distortpy import ... ; 2) from distortpy import *
if __name__ != "__main__" and __name__ != "__mp_main__":
    # from .distortMain import ...  # main class auto export on the import call of the package
    # functions auto export - when everything imported from the module
    from .distortMain import function1, function2
