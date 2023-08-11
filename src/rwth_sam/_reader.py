import json
import numpy as np
# TODO: check json structure before returning data
def napari_get_reader(path):
    if not path.endswith(".json"):
        return None
    
    return json_reader_function


def json_reader_function(path):
    """Take a path to a JSON file, read the metadata, and return a Points layer with zero points and the extracted metadata.

    Parameters
    ----------
    path : str
        Path to JSON file.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple contains (data, metadata, layer_type). 
    """
    # Load and parse the JSON file
    with open(path, 'r') as file:
        metadata = json.load(file)
    # Create an empty Points layer data to hold the metadata
    data = np.empty((0, 2))  # Zero points in a 2D space
    add_kwargs = {"metadata": metadata}
    #layer_type = "points"
    layer_type = "shapes"

    return [(data, add_kwargs, layer_type)]