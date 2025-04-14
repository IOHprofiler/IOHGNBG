import os

def get_static_package_path():
    """
    Returns the absolute path to the 'static' package directory.
    Assumes the 'static' directory is located relative to this file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_path = os.path.join(current_dir, 'static')
    return os.path.abspath(static_path) 