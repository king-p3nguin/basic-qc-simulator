"""
Module for utility functions.
"""

import importlib.util


def check_module_installed(module_name: str):
    """Check if a module is installed

    Args:
        module_name (str): name of the module

    """
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ModuleNotFoundError(
            f"You need to install the '{module_name}' module to use this function."
        )
