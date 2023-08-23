import copy
import inspect
import warnings
from collections import Counter
from enum import Enum
from os.path import join
from pathlib import Path
import urllib.request

import numpy as np
import torch
from pympler import asizeof # FIXME: remove for release
import napari
from qtpy.QtWidgets import (
    QWidget,
    QApplication,
)
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QSpinBox, 
    QPushButton, QMessageBox
)
from superqt.utils import qdebounced
from tqdm import tqdm
from vispy.util.keys import CONTROL

from rwth_sam._ui_elements import UiElements, AnnotatorMode
from rwth_sam.slicer import slicer
from rwth_sam.utils import normalize

from segment_anything import (
    SamPredictor,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
)
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

from mobile_sam import (
    SamPredictor as SamPredictorMobile,
    build_sam_vit_t
)
from mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorMobile

from segment_anything_hq import (
    SamPredictor as SamPredictorHQ,
    build_sam_vit_h as build_sam_vit_h_hq,
    build_sam_vit_l as build_sam_vit_l_hq,
    build_sam_vit_b as build_sam_vit_b_hq,
)
from segment_anything_hq.automatic_mask_generator import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorHQ


class SegmentationMode(Enum):
    SEMANTIC = 0
    INSTANCE = 1

class BboxState(Enum):
    CLICK = 0
    DRAG = 1
    RELEASE = 2

SAM_MODELS = {
    "default": {"filename": "sam_vit_h_4b8939.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model": build_sam_vit_h, "predctor": SamPredictor, "automatic_mask_generator": SamAutomaticMaskGenerator},
    "vit_h": {"filename": "sam_vit_h_4b8939.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model": build_sam_vit_h, "predctor": SamPredictor, "automatic_mask_generator": SamAutomaticMaskGenerator},
    "vit_l": {"filename": "sam_vit_l_0b3195.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "model": build_sam_vit_l, "predctor": SamPredictor, "automatic_mask_generator": SamAutomaticMaskGenerator},
    "vit_b": {"filename": "sam_vit_b_01ec64.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "model": build_sam_vit_b, "predctor": SamPredictor, "automatic_mask_generator": SamAutomaticMaskGenerator},
    "vit_h_hq": {"filename": "sam_hq_vit_h.pth", "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth", "model": build_sam_vit_h_hq, "predctor": SamPredictorHQ, "automatic_mask_generator": SamAutomaticMaskGeneratorHQ},
    "vit_l_hq": {"filename": "sam_hq_vit_l.pth", "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth", "model": build_sam_vit_l_hq, "predctor": SamPredictorHQ, "automatic_mask_generator": SamAutomaticMaskGeneratorHQ},
    "vit_b_hq": {"filename": "sam_hq_vit_b.pth", "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth", "model": build_sam_vit_b_hq, "predctor": SamPredictorHQ, "automatic_mask_generator": SamAutomaticMaskGeneratorHQ},
    "MedSAM": {"filename": "sam_vit_b_01ec64_medsam.pth", "url": "https://syncandshare.desy.de/index.php/s/yLfdFbpfEGSHJWY/download/medsam_20230423_vit_b_0.0.1.pth", "model": build_sam_vit_b, "predctor": SamPredictor, "automatic_mask_generator": SamAutomaticMaskGenerator},
    "MobileSAM" : {"filename": "mobile_sam.pt", "url": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt?raw=true", "model": build_sam_vit_t, "predctor": SamPredictorMobile, "automatic_mask_generator": SamAutomaticMaskGeneratorMobile}
}
# BUG: when switching images from 3D to 2D an error is thrown


class SAMWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        # TODO: make sure to reset everything when switching images
        self.viewer = napari_viewer
        self.ui_elements = UiElements(self.viewer)
    
        # a list of all variables
        ['_history_limit',
        'check_prev_mask',
        'image_layer',
        'image_name',
        'label_color_mapping',
        'label_layer',
        'point_size',
        'points',
        'points_labels',
        'points_layer',
        'sam_anything_predictor',
        'sam_features',
        'sam_logits',
        'sam_model',
        'sam_model_type',
        'sam_predictor',
        'temp_class_id',
        'temp_label_layer']

        # a list of variables that should be taken into consideration when deactivating/activating
        ['check_prev_mask',
        'sam_features']


        self.debounced_on_contrast_limits_change = qdebounced(self.on_contrast_limits_change, timeout=1000)
        self.sam_model = None
        self.sam_predictor = None
        self.temp_class_id = 95 # corresponds to green color
        
        self.setLayout(self.ui_elements.main_layout)

        #### setting up ui callbacks ####
        self.ui_elements.set_external_handler_btn_load_model(self.load_model)
        self.ui_elements.set_external_handler_btn_activate(self.activate, self.deactivate)
        self.ui_elements.set_external_handler_btn_submit_to_class(self.submit_to_class)
    ################################ model loading ################################
    def load_model(self, model_type):
        if not torch.cuda.is_available():
            if not torch.backends.mps.is_available():
                device = "cpu"
            else:
                device = "mps"
        else:
            device = "cuda"
            torch.cuda.empty_cache()
        device = "cpu" #FIXME: Remove #DEBUG 
        self.sam_model = SAM_MODELS[model_type]["model"](self.get_weights_path(model_type))

        self.sam_model.to(device)
        self.sam_predictor = SAM_MODELS[model_type]["predctor"](self.sam_model)
        self.sam_model_type = model_type

    def get_weights_path(self, model_type):
        weight_url = SAM_MODELS[model_type]["url"]

        cache_dir = Path.home() / ".cache/napari-segment-anything"
        cache_dir.mkdir(parents=True, exist_ok=True)

        weight_path = cache_dir / SAM_MODELS[model_type]["filename"]

        if not weight_path.exists():
            print("Downloading {} to {} ...".format(weight_url, weight_path))
            self.download_with_progress(weight_url, weight_path)

        return weight_path

    def download_with_progress(self, url, output_file):
        # Open the URL and get the content length
        req = urllib.request.urlopen(url)
        content_length = int(req.headers.get('Content-Length'))

        self.ui_elements.create_progress_bar(int(content_length / 1024), "Downloading model:")

        # Set up the tqdm progress bar
        progress_bar_tqdm = tqdm(total=content_length, unit='B', unit_scale=True, desc="Downloading model")

        # Download the file and update the progress bar
        with open(output_file, 'wb') as f:
            downloaded_bytes = 0
            while True:
                buffer = req.read(8192)
                if not buffer:
                    break
                downloaded_bytes += len(buffer)
                f.write(buffer)
                progress_bar_tqdm.update(len(buffer))

                # Update the progress bar using UiElements method
                self.ui_elements.update_progress_bar(int(downloaded_bytes / 1024))

        self.ui_elements.delete_progress_bar()

        progress_bar_tqdm.close()
        req.close()

    ################################ activating sam ################################
    def activate(self, annotator_mode):
        self.set_layers()
        self.set_point_size()
        self.check_image_dimension()
        self.adjust_image_layer_shape()
        self.set_sam_logits()

        self.points_layer = None
        self.submit_to_class(self.temp_class_id) # init

        if annotator_mode == AnnotatorMode.AUTO:
            self.activate_annotation_mode_auto()

        elif annotator_mode == AnnotatorMode.CLICK:
            self.activate_annotation_mode_click()

    #### preparing
    def set_layers(self):
        self.image_name = self.ui_elements.cb_input_image_selctor.currentText()
        self.image_layer = self.viewer.layers[self.ui_elements.cb_input_image_selctor.currentText()]
        self.label_layer = self.viewer.layers[self.ui_elements.cb_output_label_selctor.currentText()]
        
        
        #FIXME: remove for release
        size_image_layer = asizeof.asizeof(self.image_layer)
        size_label_layer = asizeof.asizeof(self.label_layer)
        print(f"Memory usage of image_layer: {size_image_layer} bytes") 
        print(f"Memory usage of label_layer: {size_label_layer} bytes")
        # TODO: History
        # self.label_layer_changes = None

    def set_point_size(self):
        if self.image_layer.ndim == 2:
            self.point_size = max(int(np.min(self.image_layer.data.shape[:2]) / 100), 1) # 2D Shape is (Height, Width, Channels)
        else:
            self.point_size = max(int(np.min(self.image_layer.data.shape[-2:]) / 100), 1) # 3D Shape is (Layers, Height, Width)

    def check_image_dimension(self):
        if self.image_layer.ndim not in [2, 3]:
            raise RuntimeError("Only 2D and 3D images are supported.")
        
        if self.image_layer.ndim != self.label_layer.ndim:
            self.ui_elements._internal_handler_btn_activate()
            raise RuntimeError("Image and label layer must have the same number of dimensions. Please choose another image or label layer.")

        if self.image_layer.ndim == 2:
            if self.image_layer.data.shape[:2] != self.label_layer.data.shape:
                self.ui_elements._internal_handler_btn_activate()
                raise RuntimeError("Image and label layer must have the same shape. Please choose another image or label layer.")
            
        elif self.image_layer.ndim == 3:
            if len(self.image_layer.data.shape) == 4:
                self.ui_elements._internal_handler_btn_activate()
                raise RuntimeError("Multichannel 3D images are not supported. Please choose another image layer.")
            
            if self.image_layer.data.shape != self.label_layer.data.shape:
                self.ui_elements._internal_handler_btn_activate()
                raise RuntimeError("Image and label layer must have the same shape. Please choose another image or label layer.")




    def adjust_image_layer_shape(self):
        if self.image_layer.ndim == 3:
            # Fixes shape adjustment by napari
            self.image_layer_affine_scale = self.image_layer.affine.scale
            self.image_layer_scale = self.image_layer.scale
            self.image_layer_scale_factor = self.image_layer.scale_factor
            self.label_layer_affine_scale = self.label_layer.affine.scale
            self.label_layer_scale = self.label_layer.scale
            self.label_layer_scale_factor = self.label_layer.scale_factor
            self.image_layer.affine.scale = np.array([1, 1, 1])
            self.image_layer.scale = np.array([1, 1, 1])
            self.image_layer.scale_factor = 1
            self.label_layer.affine.scale = np.array([1, 1, 1])
            self.label_layer.scale = np.array([1, 1, 1])
            self.label_layer.scale_factor = 1
            pos = self.viewer.dims.point
            self.viewer.dims.set_point(0, 0)
            self.viewer.dims.set_point(0, pos[0])
            self.viewer.reset_view()

    def set_sam_logits(self):
        if self.image_layer.ndim == 2:
            self.sam_logits = None
        else:
            self.sam_logits = [None] * self.image_layer.data.shape[0]

    #### auto
    def activate_annotation_mode_auto(self):
        args = {
            'points_per_side': int(self.ui_elements.le_points_per_side.text()),
            'points_per_batch': int(self.ui_elements.le_points_per_batch.text()),
            'pred_iou_thresh': float(self.ui_elements.le_prediction_iou_threshold.text()),
            'stability_score_thresh': float(self.ui_elements.le_stability_score_threshold.text()),
            'stability_score_offset': float(self.ui_elements.le_stability_score_offset.text()),
            'box_nms_thresh': float(self.ui_elements.le_box_nms_threshold.text()),
            'crop_n_layers': int(self.ui_elements.le_crop_n_layers.text()),
            'crop_nms_thresh': float(self.ui_elements.le_crop_nms_threshold.text()),
            'crop_overlap_ratio': float(self.ui_elements.le_crop_overlap_ratio.text()),
            'crop_n_points_downscale_factor': int(self.ui_elements.le_crop_n_points_downscale_factor.text()),
            'min_mask_region_area': int(self.ui_elements.le_minimum_mask_region_area.text()),
        }

        self.sam_anything_predictor = SAM_MODELS[self.sam_model_type]["automatic_mask_generator"](self.sam_model, **args)

        prediction = self.predict_everything()
        self.label_layer.data = prediction

    def predict_everything(self):
        contrast_limits = self.image_layer.contrast_limits
        if self.image_layer.ndim == 2:
            image = np.asarray(self.image_layer.data)
            prediction = self.predict_everything_2D(image, contrast_limits)
        elif self.image_layer.ndim == 3:
            self.ui_elements.create_progress_bar(self.image_layer.data.shape[0], "Predicting everything:")
            prediction = []
            for index in tqdm(range(self.image_layer.data.shape[0]), desc="Predicting everything"):
                image_slice = np.asarray(self.image_layer.data[index, ...])
                prediction_slice = self.predict_everything_2D(image_slice, contrast_limits)
                prediction.append(prediction_slice)
                self.ui_elements.update_progress_bar(index+1)
                QApplication.processEvents()
            self.ui_elements.delete_progress_bar()
            prediction = np.asarray(prediction)
            prediction = self.merge_classes_over_slices(prediction)
        else:
            raise RuntimeError("Only 2D and 3D images are supported.")
        return prediction

    def predict_everything_2D(self, image, contrast_limits):
        image = self.prepare_image_for_sam(image, contrast_limits)
        records = self.sam_anything_predictor.generate(image)#
        masks = np.asarray([record["segmentation"] for record in records])#
        prediction = np.argmax(masks, axis=0)
        return prediction 

    def merge_classes_over_slices(self, prediction, threshold=0.5):  # Currently only computes overlap from next_slice to current_slice but not vice versa
        for i in range(prediction.shape[0] - 1):
            current_slice = prediction[i]
            next_slice = prediction[i+1]
            next_labels, next_label_counts = np.unique(next_slice, return_counts=True)
            next_label_counts = next_label_counts[next_labels != 0]
            next_labels = next_labels[next_labels != 0]
            new_next_slice = np.zeros_like(next_slice)
            if len(next_labels) > 0:
                for next_label, next_label_count in zip(next_labels, next_label_counts):
                    current_roi_labels = current_slice[next_slice == next_label]
                    current_roi_labels, current_roi_label_counts = np.unique(current_roi_labels, return_counts=True)
                    current_roi_label_counts = current_roi_label_counts[current_roi_labels != 0]
                    current_roi_labels = current_roi_labels[current_roi_labels != 0]
                    if len(current_roi_labels) > 0:
                        current_max_count = np.max(current_roi_label_counts)
                        current_max_count_label = current_roi_labels[np.argmax(current_roi_label_counts)]
                        overlap = current_max_count / next_label_count
                        if overlap >= threshold:
                            new_next_slice[next_slice == next_label] = current_max_count_label
                        else:
                            new_next_slice[next_slice == next_label] = next_label
                    else:
                        new_next_slice[next_slice == next_label] = next_label
                prediction[i+1] = new_next_slice
        return prediction
    
    # TODO: control correctness
    """ chatgpt created 
    def merge_classes_over_slices(self, prediction, threshold=0.5):
        for i in range(prediction.shape[0] - 1):
            current_slice, next_slice = prediction[i], prediction[i+1]
            unique_next_labels = np.unique(next_slice)
            for label in unique_next_labels:
                if label != 0:  # ignore the background
                    overlapping_labels = current_slice[next_slice == label]
                    label_counter = Counter(overlapping_labels[overlapping_labels != 0])
                    if label_counter:  # if not empty
                        most_common_label, most_common_count = label_counter.most_common(1)[0]
                        overlap = most_common_count / np.sum(overlapping_labels == label)
                        if overlap >= threshold:
                            next_slice[next_slice == label] = most_common_label
        return prediction
     """
    
    #### click
    def activate_annotation_mode_click(self):
            #TODO: add again
            #self.create_label_color_mapping() 

            # TODO: History
            """ 
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self._history_limit = self.label_layer._history_limit
            self._reset_history()
             """

            self.image_layer.events.contrast_limits.connect(self.debounced_on_contrast_limits_change)
            self.viewer.mouse_drag_callbacks.append(self.callback_click)

            self.set_image()

            self.viewer.keymap['Delete'] = self.on_delete
            self.viewer.keymap['Control-K'] = self.on_delete_all
            # TODO: History: keymap
            """ 
            
            self.label_layer.keymap['Control-Z'] = self.on_undo
            self.label_layer.keymap['Control-Shift-Z'] = self.on_redo
             """

    def create_label_color_mapping(self, num_labels=1000):
        self.label_color_mapping = {"label_mapping": {}, "color_mapping": {}}
        for label in range(num_labels):
            color = self.label_layer.get_color(label)
            self.label_color_mapping["label_mapping"][label] = color
            self.label_color_mapping["color_mapping"][str(color)] = label

    def set_image(self):
        self.sam_features = self.extract_feature_embeddings(self.image_layer.data, self.image_layer.contrast_limits)

    def extract_feature_embeddings(self, image, contrast_limits):
        if self.image_layer.ndim == 2:
            return self.extract_feature_embeddings_2D(image, contrast_limits)
        elif self.image_layer.ndim == 3:
            return self.extract_feature_embeddings_3D(image, contrast_limits)

    def extract_feature_embeddings_2D(self, image, contrast_limits):
        image = self.prepare_image_for_sam(image, contrast_limits)
        self.sam_predictor.set_image(image)
        return self.sam_predictor.features

    def extract_feature_embeddings_3D(self, data, contrast_limits):
        total_slices = data.shape[0]
        self.ui_elements.create_progress_bar(total_slices, "Creating SAM image embedding:")
        sam_features = []

        for index, image_slice in enumerate(data):
            image_slice = np.asarray(image_slice)
            sam_features.append(self.extract_feature_embeddings_2D(image_slice, contrast_limits))
            self.ui_elements.update_progress_bar(index + 1)

        self.ui_elements.delete_progress_bar()
        return sam_features

    def prepare_image_for_sam(self, image, contrast_limits):
        if not self.image_layer.rgb:
            image = np.stack((image,) * 3, axis=-1)  # Expand to 3-channel image
        image = image[..., :3]  # Remove a potential alpha channel
        image = normalize(image, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)#
        return image

    def on_contrast_limits_change(self):
        self.set_image()

    def callback_click(self, layer, event):
        """ decides what to do when a click is performed on the image and calls the corresponding function """
        data_coordinates = self.image_layer.world_to_data(event.position)
        coords = np.round(data_coordinates).astype(int)
        print(f"the selected point is: {coords}")
        if (not CONTROL in event.modifiers) and event.button == 3:  # Positive middle click
            self.do_point_click(coords, 1)
            yield
        elif CONTROL in event.modifiers and event.button == 3:  # Negative middle click
            self.do_point_click(coords, 0)
            yield
        elif (not CONTROL in event.modifiers) and event.button == 1 and self.points_layer is not None and len(self.points_layer.data) > 0:
            # Find the closest point to the mouse click
            distances = np.linalg.norm(self.points_layer.data - coords, axis=1)
            closest_point_idx = np.argmin(distances)
            closest_point_distance = distances[closest_point_idx]

            # Select the closest point if it's within self.point_size pixels of the click
            if closest_point_distance <= self.point_size:
                self.points_layer.selected_data = {closest_point_idx}
            else:
                self.points_layer.selected_data = set()
            yield
        elif (CONTROL in event.modifiers) and event.button == 1:
            picked_label = self.label_layer.data[slicer(self.label_layer.data, coords)]
            self.label_layer.selected_label = picked_label
            yield

    def do_point_click(self, coords, is_positive):
        """ checks for repeated points, adds the point to the points list and calls the prediction function"""
        # Check if there is already a point at these coordinates
        for point in self.points:
            if np.array_equal(coords, point):
                warnings.warn("There is already a point in this location. This click will be ignored.")
                return

        self.points_labels.append(is_positive)
        self.points.append(coords)

        x_coord = coords[0]
        
        prediction = self.predict_sam(points=self.points, 
                                    labels=self.points_labels, 
                                    x_coord=x_coord)

        self.update_points_layer()
        self.update_label_layer(prediction, self.temp_class_id, x_coord)

    def predict_sam(self, points, labels, x_coord=None):
        points = copy.deepcopy(points)
        labels = copy.deepcopy(labels)
        x_coord = copy.deepcopy(x_coord)


        if self.image_layer.ndim == 2:
            points = np.array(points)
            points = np.flip(points, axis=-1)
            labels = np.array(labels)

            # TODO: check prev mask
            """ 
            logits = self.sam_logits
            if not self.check_prev_mask.isChecked():
                logits = None
            """
            logits = None
            self.sam_predictor.features = self.sam_features
            prediction, _, self.sam_logits = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                mask_input=logits,
                multimask_output=False,
            )
            prediction =prediction.squeeze(axis=0) # was (1,h,w) because multimask_output=False
        # TODO: add 3D support
        elif self.image_layer.ndim == 3:
            prediction = np.zeros_like(self.label_layer.data)

            points = np.array(points)
            x_coords = np.unique(points[:, 0])

            # Group points if they are on the same image slice
            groups = {x_coord: list(points[points[:, 0] == x_coord]) for x_coord in x_coords}

            # Extract group points and labels
            group_points = groups[x_coord]
            group_labels = [labels[np.argwhere(np.all(points == point, axis=1)).flatten()[0]] for point in group_points]

            # removing x-coordinate (depth)
            group_points = [point[1:] for point in group_points]

            # Flip because of sam coordinates system
            points = np.flip(group_points, axis=-1)
            labels = np.asarray(group_labels)

            self.sam_predictor.features = self.sam_features[x_coord]

            # TODO: use when check_prev_mask is implemented
            #logits = self.sam_logits[x_coord] if not self.check_prev_mask.isChecked() else None

            logits = None
            prediction_yz, _, self.sam_logits[x_coord] = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                mask_input=logits,
                multimask_output=False,
            )
            
            # Adjust shape of prediction_yz and update prediction array
            prediction_yz = prediction_yz.squeeze(axis=0) # was (1,h,w) because multimask_output=False
            prediction[x_coord, :, :] = prediction_yz

        else:
            raise RuntimeError("Only 2D and 3D images are supported.")
        return prediction

    def update_points_layer(self):
        selected_layer = None
        color_list = ["red" if i==0 else "blue" for i in self.points_labels]
        print(f"points_labels: {self.points_labels}")
        print(f"color_list: {color_list}")
        #save selected layer
        if self.viewer.layers.selection.active != self.points_layer:
            selected_layer = self.viewer.layers.selection.active


        if self.points_layer is not None:
            self.viewer.layers.remove(self.points_layer)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.points_layer = self.viewer.add_points(name="ignore this layer", data=np.asarray(self.points), face_color=color_list, edge_color="white", size=self.point_size)
        self.points_layer.editable = False
        self.points_layer.refresh()

        #reselect selected layer
        if selected_layer is not None:
            self.viewer.layers.selection.active = selected_layer

    def update_label_layer(self,prediction, point_label, x_coord):

        # x_coord selects everything
        if self.image_layer.ndim == 2:
            x_coord = slice(None, None)

        label_layer = np.asarray(self.label_layer.data)
        
        # Create masks for better readability
        is_current_class = label_layer[x_coord] == point_label
        indecies_prediction_true = prediction[x_coord] == 1
        indecies_prediction_false = ~indecies_prediction_true

        print(f"shape of is_current_class: {is_current_class.shape}")
        print(f"shape of indecies_prediction_true: {indecies_prediction_true.shape}")
        print(f"shape of indecies_prediction_false: {indecies_prediction_false.shape}")
        print(f"shape of label_layer[x_coord]: {label_layer[x_coord].shape}")
        # Reset label_layer for the current class
        label_layer[x_coord][is_current_class] = 0

        # Update the label layer for the current class
        label_layer[x_coord][indecies_prediction_true] = point_label
        label_layer[x_coord][indecies_prediction_false] = self.temp_label_layer[x_coord][indecies_prediction_false]

        # Update the label layer data
        self.label_layer.data = label_layer

    def submit_to_class(self, class_id):
        self.points = []
        self.points_labels = []
        self.update_points_layer()
        print(class_id)

        label_layer = np.asarray(self.label_layer.data)
        print(f"shape of label_layer: {label_layer.shape}")
        label_layer[label_layer == self.temp_class_id] = class_id
        label_layer[label_layer == self.temp_class_id] = 0 

        self.label_layer.data = label_layer
        self.temp_label_layer = np.copy(label_layer)

    def deactivate(self):
        self.remove_all_widget_callbacks(self.viewer)
        if self.label_layer is not None:
            self.remove_all_widget_callbacks(self.label_layer)
        if self.points_layer is not None and self.points_layer in self.viewer.layers:
            self.viewer.layers.remove(self.points_layer)

        self.image_layer.events.contrast_limits.disconnect(self.debounced_on_contrast_limits_change)

    def remove_all_widget_callbacks(self, layer):
        callback_types = ['mouse_double_click_callbacks', 'mouse_drag_callbacks', 'mouse_move_callbacks',
                          'mouse_wheel_callbacks', 'keymap']
        for callback_type in callback_types:
            callbacks = getattr(layer, callback_type)
            if isinstance(callbacks, list):
                for callback in callbacks:
                    if inspect.ismethod(callback) and callback.__self__ == self:
                        callbacks.remove(callback)
            elif isinstance(callbacks, dict):
                for key in list(callbacks.keys()):
                    if inspect.ismethod(callbacks[key]) and callbacks[key].__self__ == self:
                        del callbacks[key]
            else:
                raise RuntimeError("Could not determine callbacks type.")
    
    def on_delete(self, layer):
        if self.points_layer is None or len(self.points_layer.selected_data) == 0:
            raise RuntimeError("No point selected.")

        selected_point = list(self.points_layer.selected_data)[0]
        print(selected_point)
        print(type(selected_point))
        x_coord = None
        if self.image_layer.ndim == 3:
            x_coord = self.points[selected_point][0]
        
        self.points.pop(selected_point)
        self.points_labels.pop(selected_point)
        
        if len(self.points) == 0:
            prediction = np.zeros_like(self.label_layer.data)

        elif self.image_layer.ndim == 3:
            points = copy.deepcopy(self.points)
            points = np.array(points)
            x_coords = np.unique(points[:, 0])
            groups = {x_coord: list(points[points[:, 0] == x_coord]) for x_coord in x_coords}

        if len(self.points) > 0:
            # are there points in the current slice left?
            if self.image_layer.ndim == 3 and x_coord not in groups.keys():
                prediction = np.zeros_like(self.label_layer.data)

            else:
                prediction = self.predict_sam(points=self.points, 
                                            labels=self.points_labels, 
                                            x_coord=x_coord)
    
        self.update_points_layer()
        self.update_label_layer(prediction, self.temp_class_id, x_coord)
    
    def on_delete_all(self, layer):
        x_coord = None
        prediction = np.zeros_like(self.label_layer.data)

        if self.image_layer.ndim == 2:
            self.update_label_layer(prediction, self.temp_class_id, x_coord)
        else:
            x_coords = [point[0] for point in self.points]
            for x_coord in x_coords:
                self.update_label_layer(prediction, self.temp_class_id, x_coord) 

        self.points.clear()
        self.points_labels.clear()
        self.update_points_layer()



class SegmentationProfileQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.class_names = []

        layout = QVBoxLayout()

        
        layout.addWidget(QLabel("segmanation profile:"))
        # Layer name input
        self.layer_name_edit = QLineEdit()
        self.layer_name_edit.setPlaceholderText("Segmanation Profile Name")
        layout.addWidget(self.layer_name_edit)

        # Class count input
        self.class_count_spinbox = QSpinBox()
        self.class_count_spinbox.setValue(4)
        self.class_count_spinbox.setMinimum(1)
        layout.addWidget(QLabel("Number of classes:"))
        layout.addWidget(self.class_count_spinbox)
        self.class_count_spinbox.valueChanged.connect(self._update_class_fields)

        # Dynamic class name fields
        self.class_inputs_layout = QVBoxLayout()
        layout.addLayout(self.class_inputs_layout)
        self._update_class_fields()

        # Save button
        save_btn = QPushButton("Save Profile")
        save_btn.clicked.connect(self._save_profile)
        layout.addWidget(save_btn)

        self.setLayout(layout)

    def _update_class_fields(self):
        count = self.class_count_spinbox.value()
        while self.class_inputs_layout.count():
            widget = self.class_inputs_layout.takeAt(0).widget()
            widget.deleteLater()

        self.class_name_edits = []
        for i in range(1, count + 1):
            edit = QLineEdit(self)
            edit.setPlaceholderText(f"Class {i} Name")
            self.class_inputs_layout.addWidget(edit)
            self.class_name_edits.append(edit)

    def _save_profile(self):
        layer_name = self.layer_name_edit.text()
        self.class_names = [edit.text().strip() for edit in self.class_name_edits]

        # Check if any field is empty
        if not layer_name or any(not class_name for class_name in self.class_names):
            QMessageBox.warning(self, "Input Error", "Please fill all the fields before saving.")
            return

        # Check if class names are unique
        if len(self.class_names) != len(set(self.class_names)):
            QMessageBox.warning(self, "Input Error", "Class names must be unique. Please correct before saving.")
            return

        profile = {
            "classes": self.class_names
        }

        # Add a new shapes layer in Napari
        if self.viewer is not None:
            data = np.empty((0, 2))
            self.viewer.add_shapes(data, name=layer_name, metadata=profile)
