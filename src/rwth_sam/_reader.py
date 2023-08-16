import json
import numpy as np

def napari_get_reader(path):
    if not path.endswith(".json"):
        return None

    # Before returning the json_reader_function, validate the JSON content
    with open(path, 'r') as file:
        data = json.load(file)

    if not validate_json_structure(data):
        #raise InvalidFormatError("JSON file does not match the expected format.")
        return None
    
    return json_reader_function

def json_reader_function(path):
    """Take a path to a JSON file, read the metadata, and return a shapes layer with zero points and the extracted metadata."""
    # Load and parse the JSON file
    with open(path, 'r') as file:
        metadata = json.load(file)

    # Create an empty Points layer data to hold the metadata
    data = np.empty((0, 2))  # Zero points in a 2D space
    add_kwargs = {"metadata": metadata}
    layer_type = "shapes" # Use an empty shapes layer to store segmentation profile as metadata

    return [(data, add_kwargs, layer_type)]

def validate_json_structure(data):
    """ Validate if the JSON structure has the expected segmentation profile format. """
    # Check if 'classes' key exists and is a list
    if 'classes' not in data or not isinstance(data['classes'], list):
        raise Exception("JSON file is missing 'classes' key or it is not a list.")
    ...

    # Check if all elements of 'classes' are strings
    for cls in data['classes']:
        if not isinstance(cls, str):
            raise Exception(f"Invalid entry in 'classes'. Expected strings but found {type(cls)}.")

    return True

class InvalidFormatError(Exception):
    """Custom exception for invalid format."""
    pass