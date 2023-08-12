import json
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_single_image(path: str, data: Any, meta: dict) -> List[str]:
    """Writes a single image layer."""
    
    # Check if the layer has 'classes' in the nested 'metadata'
    if 'metadata' in meta and 'classes' in meta['metadata']:
        classes_data = {
            "classes": meta['metadata']['classes']
        }
        
        with open(path, 'w') as file:
            json.dump(classes_data, file)

        return [path]
    
    # Other saving logic can be added here...
    return [path]
