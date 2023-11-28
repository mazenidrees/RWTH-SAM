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
from vispy.util.keys import CONTROL, SHIFT

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
    DELETE = 3

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
# BUG: (possible) when switching images from 3D to 2D an error is thrown 
# BUG: when deactivating and choosing another model without loading it the activate button is still enabled
# BUG: when removing image, label and/or profile layer, you can no longer deactivate

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
        total_steps = 10  # Assume there are 10 major steps in the activation process
        self.ui_elements.create_progress_bar(total_steps, "Activating:")
        try:
            self.points_layer = None
            self.bbox_layer = None
            self.ui_elements.update_progress_bar(1)

            self.set_layers()
            self.ui_elements.update_progress_bar(2)

            self.set_point_and_bbox_size()
            self.ui_elements.update_progress_bar(3)

            self.check_image_dimension()
            self.ui_elements.update_progress_bar(4)

            self.adjust_image_layer_shape()
            self.ui_elements.update_progress_bar(5)

            self.set_sam_logits()
            self.ui_elements.update_progress_bar(6)

            self.create_label_color_mapping()
            self.ui_elements.update_progress_bar(7)

            self.ui_elements.cs_class_selector.set_colors(self.label_color_mapping["label_mapping"])
            self.ui_elements.update_progress_bar(8)

            self._submit_to_class(self.temp_class_id)  # init
            self.ui_elements.update_progress_bar(9)

            if annotator_mode == AnnotatorMode.AUTO:
                self.activate_annotation_mode_auto()
            elif annotator_mode == AnnotatorMode.CLICK:
                self.activate_annotation_mode_click()
            self.ui_elements.update_progress_bar(10)

        except Exception as e:
            # Here, you can handle the error or print it, if desired
            print(f"Error encountered: {e}")
            self.ui_elements.delete_progress_bar()  # This will stop the progress bar
            raise e  # Re-raise the exception for further handling, if needed

        self.ui_elements.delete_progress_bar()

    #### preparing
    def set_layers(self):
        self.image_name = self.ui_elements.cb_input_image_selctor.currentText()
        self.label_name = self.ui_elements.cb_output_label_selctor.currentText()

        self.image_layer = self.viewer.layers[self.image_name]
        self.label_layer = self.viewer.layers[self.label_name]

        self.label_layer.name = f"{self.image_name}_label"

        if self.image_layer.ndim == 2:
            self.image_shape = self.label_layer.data.shape # using label_layer because image_layer might be a 2D image with 3 channels
        elif self.image_layer.ndim == 3:
            self.image_shape = self.label_layer.data.shape[1:]



    def set_point_and_bbox_size(self):
        """ Sets the size of the points and bounding boxes based on the geometric mean of the image dims. """
        geometric_mean = np.sqrt(self.image_shape[0]*self.image_shape[1])
        point_factor = 100 
        bbox_factor = 800

        self.point_size = max(int(geometric_mean / point_factor), 1)
        self.bbox_edge_width =max(int(geometric_mean / bbox_factor), 1)


 
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

    ########## auto
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
    

    ########## click mode (manual)
    def activate_annotation_mode_click(self):


            self.image_layer.events.contrast_limits.connect(self.debounced_on_contrast_limits_change)
            self.viewer.mouse_drag_callbacks.append(self.decide_callback_for_clicks)

            self.set_image()

            self.viewer.keymap['Control-K'] = self.delete_selected_point
            self.viewer.keymap['Control-Shift-K'] = self.on_delete_all


    

    #### preparing after activation
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


    #### deactivating
    def deactivate(self):
        self._remove_all_widget_callbacks(self.viewer)
        if self.label_layer is not None:
            self._remove_all_widget_callbacks(self.label_layer)
        if self.points_layer is not None and self.points_layer in self.viewer.layers:
            self.viewer.layers.remove(self.points_layer)
        if self.bbox_layer is not None and self.bbox_layer in self.viewer.layers:
            self.viewer.layers.remove(self.bbox_layer)

        self.image_layer.events.contrast_limits.disconnect(self.debounced_on_contrast_limits_change)

    def _remove_all_widget_callbacks(self, layer):
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

    #### callbacks
    def decide_callback_for_clicks(self, layer, event):
        coords = self._get_coords(event)
  
        if (not CONTROL in event.modifiers) and (not SHIFT in event.modifiers) and event.button == 3:  # Positive middle click
            self.catch_new_prompt()
            self.add_point(coords, 1)
            yield

        elif (CONTROL in event.modifiers) and event.button == 3:  # Negative middle click
            self.catch_new_prompt()
            self.add_point(coords, 0)
            yield

        elif (not CONTROL in event.modifiers) and event.button == 1 and self.points_layer is not None and len(self.points_layer.data) > 0:
            self.select_point(coords)
            yield

        elif (CONTROL in event.modifiers) and event.button == 1:
            self.select_label(coords)
            yield

        elif (SHIFT in event.modifiers) and event.button == 3:  # Positive middle click
            self.catch_new_prompt()
            yield from self.handle_bbox_click(event, coords)

    def handle_bbox_click(self, event, initial_coords):
        self.add_bbox(initial_coords, BboxState.CLICK)
        yield
        while event.type == 'mouse_move' or event.type == 'mouse_release':
            coords = self._get_coords(event)
            if event.type == 'mouse_move':
                self.add_bbox(coords, BboxState.DRAG)
            else:  # event.type == 'mouse_release':
                self.add_bbox(coords, BboxState.RELEASE)
            yield

    def catch_new_prompt(self):
        label_layer = np.asarray(self.label_layer.data)
        if not np.any(label_layer == self.temp_class_id):
            self._submit_to_class(0)


    def select_label(self, coords):
            picked_label = self.label_layer.data[slicer(self.label_layer.data, coords)]
            self.label_layer.selected_label = picked_label

    def select_point(self, coords):
        # Find the closest point to the mouse click
        distances = np.linalg.norm(self.points_layer.data - coords, axis=1)
        closest_point_idx = np.argmin(distances)
        closest_point_distance = distances[closest_point_idx]

        # Select the closest point if it's within self.point_size pixels of the click
        if closest_point_distance <= self.point_size:
            self.points_layer.selected_data = {closest_point_idx}
        else:
            self.points_layer.selected_data = set()

    def add_bbox(self, coords, bbox_state):
        if bbox_state == BboxState.CLICK:
            self.bbox_first_coords = coords
            return

        if self.image_layer.ndim == 2:
            self.bbox = np.asarray([self.bbox_first_coords, (self.bbox_first_coords[0], coords[1]), coords, (coords[0], self.bbox_first_coords[1])])
        elif self.image_layer.ndim == 3:
            self.bbox = np.asarray([self.bbox_first_coords, (self.bbox_first_coords[0], self.bbox_first_coords[1], coords[2]), coords, (self.bbox_first_coords[0], coords[1], self.bbox_first_coords[2])])

        self.bbox = np.rint(self.bbox).astype(np.int32)

        x_coord = self._get_x_coord_for_given_coords(coords)
        self.bboxes[str(x_coord)] = self.bbox # using str because x_coord can be slice(None, None) for 2D images



        if bbox_state == BboxState.RELEASE:
            self._predict_and_update_label_layer(coords)

        self._update_bbox_layer(bbox_state)

    def add_point(self, coords, is_positive):
        # Check if there is already a point at these coordinates
        for point in self.points:
            if np.array_equal(coords, point):
                warnings.warn("There is already a point in this location. This click will be ignored.")
                return

        self.points_labels.append(is_positive)
        self.points.append(coords)

        self._predict_and_update_label_layer(coords)

        self._update_points_layer()

    def submit_to_class(self, class_id):
        if class_id == 0:
            warnings.warn("Please choose a class.")
            return
        self._submit_to_class(class_id)


    def _submit_to_class(self, class_id):
        self.points = []
        self.points_labels = []
        self.bboxes = {}
        self.bbox = None
        self._update_bbox_layer(BboxState.DELETE)
        self._update_points_layer()

        label_layer = np.asarray(self.label_layer.data)
        label_layer[label_layer == self.temp_class_id] = class_id
        label_layer[label_layer == self.temp_class_id] = 0 

        self.label_layer.data = label_layer
        self.temp_label_layer = np.copy(label_layer)

    def delete_selected_point(self, layer):

        if self.points_layer is None or len(self.points_layer.selected_data) == 0:
            warnings.warn("No points to delete.")
            return
        
        deleted_point_index = list(self.points_layer.selected_data)[0]
        deleted_point_coords = self.points[deleted_point_index]

        self.points.pop(deleted_point_index)
        self.points_labels.pop(deleted_point_index)

        self._predict_and_update_label_layer(deleted_point_coords)

        self._update_points_layer()

    def on_delete_all(self, layer=None):
        if self.points_layer is None:
            warnings.warn("No points layer")
            return
        
        coords_list = self._get_dummy_coords_for_delete_all()

        self.points.clear()
        self.points_labels.clear()
        self.bboxes = {}
        self.bbox = None

        for coords in coords_list:
            self._predict_and_update_label_layer(coords)


        self._update_points_layer()
        self._update_bbox_layer(BboxState.DELETE)

    def on_contrast_limits_change(self):
        self.set_image()

#TODO: optimize: submit_to_class, on_delete_all

    #### backend
    def _get_coords(self, event):
        data_coordinates = self.image_layer.world_to_data(event.position)
        coords = np.round(data_coordinates).astype(int)
        return coords

    def _predict_and_update_label_layer(self, coords):
        points, labels, bbox, x_coord = self._get_formatted_sam_prompt(points=self.points, labels=self.points_labels, bbox=self.bbox, coords=coords)
        prediction = self._predict_with_sam(points=points, labels=labels, bbox=bbox, x_coord=x_coord)
        self._update_label_layer(prediction, self.temp_class_id, x_coord)

    def _get_formatted_sam_prompt(self, points, labels, bbox, coords):

        x_coord = self._get_x_coord_for_given_coords(coords)

        points, labels = self._group_points_by_slice(points, labels, x_coord)

        points, labels, x_coord, bbox = self._deepcopy(points, labels, x_coord, bbox)
        points = np.array(points)
        labels = np.array(labels)
        # Flip because of sam coordinates system
        points = np.flip(points, axis=-1)
        labels = np.asarray(labels)
        

        bbox = self.bboxes.get(str(x_coord), None)

        if bbox is not None:
            if self.image_layer.ndim == 3:
                bbox = [item[1:] for item in bbox]

            top_left_coord, bottom_right_coord = self._find_corners(bbox)
            bbox = [np.flip(top_left_coord), np.flip(bottom_right_coord)]
            bbox = np.asarray(bbox).flatten()
        return points, labels, bbox, x_coord
    
    def _get_x_coord_for_given_coords(self, coords):
        if self.image_layer.ndim == 3:
            x_coord = coords[0]
        else: # 2D = everything
            x_coord = slice(None, None)
        return x_coord

    def _group_points_by_slice(self, points, labels, x_coord):

        if x_coord == slice(None, None):
            return points, labels

        # Check if there points left on the current image slice if at all
        if not points:
            return [], []

        points = np.array(points)
        x_coords = np.unique(points[:, 0])

        if x_coord not in x_coords:
            return [], []

        # Group points if they are on the same image slice
        groups = {x: list(points[points[:, 0] == x]) for x in x_coords}

        # Extract group points and labels
        group_points = groups[x_coord]
        group_labels = [labels[np.argwhere(np.all(points == point, axis=1)).flatten()[0]] for point in group_points]

        # Removing x-coordinate (depth)
        group_points = [point[1:] for point in group_points]

        return group_points, group_labels

    def _predict_with_sam(self, points, labels, bbox, x_coord, use_prev_mask=False):

        if points.size == 0 and bbox is None:
            return np.zeros(self.image_shape, dtype=np.int32)
        
        elif points.size == 0 and bbox is not None:
            points = None
            labels = None

        # TODO: use when check_prev_mask is implemented
        if use_prev_mask:
            logits = self.sam_logits[x_coord]
        else:
            logits = None

        self.sam_predictor.features = self.sam_features[x_coord]
        prediction, _, _ = self.sam_predictor.predict(# TODO: assign to self.sam_logits[x_coord] when check_prev_mask is implemented
            point_coords=points,
            point_labels=labels,
            box=bbox,
            mask_input=logits,
            multimask_output=False,
        )
        # Adjust shape of prediction_yz and update prediction array
        return prediction.squeeze(axis=0) # was (1,h,w) because multimask_output=False

    def _update_points_layer(self):
        selected_layer = None
        color_list = ["red" if i==0 else "blue" for i in self.points_labels]
        #save selected layer
        if self.viewer.layers.selection.active != self.points_layer:
            selected_layer = self.viewer.layers.selection.active


        if self.points_layer is not None:
            self.viewer.layers.remove(self.points_layer)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.points_layer = self.viewer.add_points(name="Points - DO NOT CHANGE", data=np.asarray(self.points), face_color=color_list, edge_color="white", size=self.point_size)
        self.points_layer.editable = False
        self.points_layer.refresh()

        #reselect selected layer
        if selected_layer is not None:
            self.viewer.layers.selection.active = selected_layer
        self.points_layer.selected_data = set()
        self.ui_elements.cs_class_selector.update_colors() # for some reason, napari is reseting them!

    def _get_dummy_coords_for_delete_all(self):
        #TODO: add coords of bboxes
        coords_list = []
        coords_list1 = []
        coords_list2 = []
        bboxes_corners_list = [bbox[0] for bbox in list(self.bboxes.values())]
 
        if self.image_layer.ndim == 3:
            points = np.array(self.points)
            if len(points) > 0:
                x_coords = np.unique(points[:, 0])
                coords_list1 = np.zeros((len(x_coords), 3), dtype=int)
                coords_list1[:, 0] = x_coords
                coords_list1 = coords_list1.tolist()
            
            bboxes_corners_list = np.array(bboxes_corners_list)
            if len(bboxes_corners_list) > 0:
                x_coords = np.unique(bboxes_corners_list[:, 0])
                coords_list2 = np.zeros((len(x_coords), 3), dtype=int)
                coords_list2[:, 0] = x_coords
                coords_list2 = coords_list2.tolist()
            
            coords_list = coords_list1 + coords_list2
        elif self.image_layer.ndim == 2:
            coords_list = [[0,0]]

        return coords_list

    def _update_label_layer(self,prediction, point_label, x_coord):

        label_layer = np.asarray(self.label_layer.data)
        
        # Create masks for better readability
        is_current_class = label_layer[x_coord] == point_label
        indecies_prediction_true = prediction == 1
        indecies_prediction_false = ~indecies_prediction_true

        # Reset label_layer for the current class
        label_layer[x_coord][is_current_class] = 0

        # Update the label layer for the current class
        label_layer[x_coord][indecies_prediction_true] = point_label
        label_layer[x_coord][indecies_prediction_false] = self.temp_label_layer[x_coord][indecies_prediction_false]

        # Update the label layer data
        self.label_layer.data = label_layer

    def _update_bbox_layer(self, bbox_state):
        # Save selected layer
        selected_layer = None
        if self.viewer.layers.selection.active != self.bbox_layer:
            selected_layer = self.viewer.layers.selection.active


        bboxes = list(self.bboxes.values())
        edge_width = [self.bbox_edge_width] * len(bboxes)
        face_color = [(0, 0, 0, 0)] * len(bboxes)

        if bbox_state == BboxState.DRAG:
            edge_colors = ['skyblue'] * len(bboxes)
        elif bbox_state == BboxState.RELEASE:
            edge_colors = ['steelblue'] * len(bboxes)
        elif bbox_state == BboxState.DELETE:
            edge_colors = ['red'] * len(bboxes) # dummy color

        if self.bbox_layer is None:
            # Create the layer if it doesn't exist
            self.bbox_layer = self.viewer.add_shapes(name="Bounding Boxes - DO NOT CHANGE", 
                                                    data=bboxes, 
                                                    edge_width=edge_width, 
                                                    edge_color=edge_colors, 
                                                    face_color=face_color, 
                                                    opacity=1)
            self.bbox_layer.editable = False

        elif self.bbox_layer is not None:
            # Update the data and colors if the layer already exists
            self.bbox_layer.data = bboxes
            self.bbox_layer.edge_width = edge_width if edge_width else self.bbox_layer.edge_width
            self.bbox_layer.edge_color = edge_colors if edge_colors else self.bbox_layer.edge_color
            self.bbox_layer.face_color = face_color if face_color else self.bbox_layer.face_color
            self.bbox_layer.refresh()

        # Reselect selected layer
        if selected_layer is not None:
            self.viewer.layers.selection.active = selected_layer

    #### utils and misc
    def _deepcopy(self, *args):
        return tuple(copy.deepcopy(arg) for arg in args)

    def _find_corners(self, coords):
        # convert the coordinates to numpy arrays
        coords = np.array(coords)

        # find the indices of the leftmost, rightmost, topmost, and bottommost coordinates
        left_idx = np.min(coords[:, 0])
        right_idx = np.max(coords[:, 0])
        top_idx = np.min(coords[:, 1])
        bottom_idx = np.max(coords[:, 1])

        # determine the top left and bottom right coordinates
        # top_left_coord = coords[top_idx, :] if left_idx != top_idx else coords[right_idx, :]
        # bottom_right_coord = coords[bottom_idx, :] if right_idx != bottom_idx else coords[left_idx, :]

        top_left_coord = [left_idx, top_idx]
        bottom_right_coord = [right_idx, bottom_idx]

        return top_left_coord, bottom_right_coord





#BUG: when closing the widget and reopening it, the old model is lost and it can't be deactivated anymore.
#FIXME: when adding more classes the names of the old ones are lost.
class SegmentationProfileQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.class_name_edits = []

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Segmentation profile:"))
        # Layer name input
        self.layer_name_edit = QLineEdit()
        self.layer_name_edit.setPlaceholderText("Segmentation Profile Name")
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
        save_btn = QPushButton("Create Profile")
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

    def _update_class_fields(self):
        count = self.class_count_spinbox.value()

        # Remove extra class name edits if count decreases
        while len(self.class_name_edits) > count:
            edit = self.class_name_edits.pop()
            edit.deleteLater()
        
        # Update existing class name edits
        for i, edit in enumerate(self.class_name_edits):
            edit.setPlaceholderText(f"Class {i + 1} Name")

        # Add new class name edits if needed
        for i in range(len(self.class_name_edits), count):
            edit = QLineEdit(self)
            edit.setPlaceholderText(f"Class {i + 1} Name")
            self.class_inputs_layout.addWidget(edit)
            self.class_name_edits.append(edit)

    def _save_profile(self):
        layer_name = self.layer_name_edit.text()
        class_names = [edit.text().strip() for edit in self.class_name_edits]

        # Check if any field is empty
        if not layer_name or any(not class_name for class_name in class_names):
            QMessageBox.warning(self, "Input Error", "Please fill all the fields before saving.")
            return

        # Check if class names are unique
        if len(class_names) != len(set(class_names)):
            QMessageBox.warning(self, "Input Error", "Class names must be unique. Please correct before saving.")
            return

        profile = {
            "classes": class_names
        }

        # Add a new shapes layer in Napari
        if self.viewer is not None:
            data = np.empty((0, 2))
            self.viewer.add_shapes(data, name=layer_name, metadata=profile)
