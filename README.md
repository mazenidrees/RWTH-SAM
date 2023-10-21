# RWTH-SAM
A plugin for segmenting images using the Segment Anything model with focus on 3d CT-images of batteries


This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template. It is inspired by [the Napari-sam plugin](https://github.com/MIC-DKFZ/napari-sam) developed by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the [German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).

It is adapted within the frame of a bachelor thesis to fit the needs of segmenting x-ray tomography images of lithium ion batteries at the Institute for Power Electronics and Electrical Drives (ISEA) at the RWTH university.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

1. Create a virtual environment `conda create -n rwth-sam python=3.10 -y` and activate it `conda activate rwth-sam`
2. Install [`pytorch>=1.7`](https://pytorch.org/get-started/locally/)
3. run the following commands:
```bash
pip install napari[all]
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install segment-anything-hq
pip install git+https://github.om/ChaoningZhang/MobileSAM.git
pip install git+https://github.com/mazenidrees/RWTH-SAM.git
pip install timm
```

You can also install all the needed packages using the rwth_sam_conda.yml file using the the following command
```bash
conda env create -f rwth_sam_conda.yml
```

## Usage
Activate the conda environment:
```bash
conda activate rwth-sam
```
Open Napari
```bash
Napari
```

### Finding The Plugin
the plugin is accessible from the plugin menu in Napari.

<img width="540" alt="image" src="https://github.com/mazenidrees/RWTH-SAM/assets/130779425/9f99f48f-c410-4a13-b8d9-519a58a56126">

Two widgets are provided:

### The Main Widget
The RWTH-SAM widget is designed for choosing the desired settings and for performing the segmentation.

#### Widget Opening and Model Selection

https://github.com/mazenidrees/RWTH-SAM/assets/130779425/450418bf-ed17-41c3-b2bb-35ba6b63d360

#### Input Image and Output Label Layers Selection
After using the native Napari tools for opening images and creating label layers, the opened/created layers become available for selection in their respective menus.

https://github.com/mazenidrees/RWTH-SAM/assets/130779425/bb77b4e6-ddc3-4fd3-99b4-e4f4c17d022c

Label layers must be created after opening the desired input image in Napari to make sure that the label layer has the right dimensions corresponding to the input image.

#### Segmentation Profile Selection
Segmentation profiles are layers in which the classes present in a dataset are stored. For more details, please refer to [The Segmentation Profile Creation Widget](https://github.com/mazenidrees/RWTH-SAM/edit/main/README.md#the-segmentation-profile-creation-widget).
After creating a segmentation profile with the segmentation profile creation widget, or opening a corresponding stored file, the segmentation profile will appear in the designated menu for selection.

https://github.com/mazenidrees/RWTH-SAM/assets/130779425/a26eb823-4f91-4b3b-a59c-3edd48ff8a1d

After selection, classes are listed in the list below it.

#### Annotation Mode
In the main widget, there is an option to choose between semi-automatic annotation (click) or automatic mask generation. Selecting automatic mask generation reveals additional settings in line with SAM's configurations. For more details, refer to [TODO].

<img width="278" alt="image" src="https://github.com/mazenidrees/RWTH-SAM/assets/130779425/35e44665-bb55-4337-b1df-56bbc105ad2b">

### The Segmentation Profile Creation Widget
As the name suggests, this widget is responsible for creating a segmentation profile containing the classes present in a dataset.

https://github.com/mazenidrees/RWTH-SAM/assets/130779425/930f0f15-0562-4816-a12c-1ed3f1adc341

Segmentation profiles can be stored as JSON files and accessed later. While JSON files are not native to Napari, the plugin will automatically manage their opening.

### Performing the segmentation in semi-automatic mode (click)
Using a combination of mouse clicks and modifier keys (control, shift), users can guide SAM with positive points, negative points, and bounding boxes. A temporary green mask will be displayed upon that. This mask can be further refined with more prompts. Once satisfied, the mask must be submitted to one of the classes available in the right menu.

#### Positive Points
Positive points indicate to SAM where to search for an object. 
They can be added using a middle mouse click.

https://github.com/mazenidrees/RWTH-SAM/assets/130779425/ec194dc5-7e3e-47b2-85f8-534ff8a037e4

#### Negative Points
Conversely, negative points direct SAM to omit a region from a generated mask, particularly if SAM's initial mask is incorrect.
They can be added using Ctrl + middle mouse click.

https://github.com/mazenidrees/RWTH-SAM/assets/130779425/b6b9ea31-aac3-445d-a588-25afb2e59c10

#### Bounding Boxes
Bounding boxes instruct SAM to search for an object within its confines.
They can be created by using Shift + sustained middle mouse click + drag + release.

https://github.com/mazenidrees/RWTH-SAM/assets/130779425/86dbd14f-1f9b-492f-afc4-70a00a80b87d

#### Deleting Prompts
Points can be selected with a left mouse click and deleted by pressing Ctrl + K.
To delete all points, Ctrl + Shift + K can be used.
To change a bounding box, simply redraw a new one.

#### Manual Modifications to Masks
Masks can be modified using the native Napari tools for annotation (eraser, painting brush, etc.).
After selecting the label layer from the left layer list, the tools will appear in the layer controls in the top left corner.
Masks can only be modified after being submitted to a class.
By picking the color linked to the desired class, you can paint or erase on a mask.
Currently, clicking "submit to class" is required to save changes. This will be addressed in the subsequent update.

### Settings for Automatic Mask Generation:

Referenced from [SAM's official documentation](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py):

- **Points per side:** 
  The number of points to be sampled along one side of the image. The total number of points equates to `points_per_side^2`. If set to `None`, the 'point_grids' must provide explicit point sampling.

- **Points per batch:** 
  Determines the number of points processed simultaneously by the model. Using higher numbers can lead to quicker results but might consume more GPU memory.

- **Prediction IoU threshold:** 
  Utilizes a filtering threshold range of [0,1] based on the model's predicted mask quality.

- **Stability score threshold:** 
  A filtering threshold in [0,1], which relates to the stability of the mask when changes are made to the cutoff. This cutoff is used to binarize the model's mask predictions.

- **Stability score offset:** 
  Specifies the shift amount for the cutoff when determining the stability score.

- **Box NMS threshold:** 
  Defines the box IoU cutoff utilized by the non-maximal suppression to filter out duplicate masks.

- **Crop N layers:** 
  If set to >0, the mask prediction is rerun on image crops. This setting also defines the number of layers to execute, where each layer runs `2^i_layer` number of image crops.

- **Crop NMS threshold:** 
  The box IoU cutoff leveraged by non-maximal suppression to filter out duplicate masks across different image crops.

- **Crop overlap ratio:** 
  Determines the extent of crop overlap. In the initial crop layer, the crops will overlap by this fraction of the total image length. Subsequent layers with increased crops will have a reduced overlap.

- **Crop N points downscale factor:** 
  The points-per-side sampled in layer 'n' are scaled down by `crop_n_points_downscale_factor^n`.

- **Min mask region area:** 
  If set to >0, post-processing is employed to eradicate disconnected regions and mask holes smaller than `min_mask_region_area`. This feature requires OpenCV.





## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"RWTH-SAM" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/mazenidrees/RWTH-SAM/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
