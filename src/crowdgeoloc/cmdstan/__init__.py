from importlib.resources import path as irpath


with irpath(__package__, "__init__.py") as path_:
    RESOURCE_DIR = __name__.split('.')[-1]  # directory name for the *.stan files
    RESOURCES_PATH = path_.parents[0] / RESOURCE_DIR


def resource_filename(filename):
    return RESOURCES_PATH / filename
