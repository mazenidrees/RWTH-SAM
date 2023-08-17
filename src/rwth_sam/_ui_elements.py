# from napari_sam.QCollapsibleBox import QCollapsibleBox
import os
from collections import deque, defaultdict
from enum import Enum
from pathlib import Path
from os.path import join

from vispy.util.keys import CONTROL

import napari
import qtpy
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator, QDoubleValidator
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


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

class AnnotatorMode(Enum):
    NONE = 0
    CLICK = 1
    AUTO = 2

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

class ClassSelector(QListWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.update_classes()
        # connect a method to the itemSelectionChanged signal
        self.itemSelectionChanged.connect(self.item_selected)

    def update_classes(self, classes=None):
        self.clear()
        if classes:
            for class_name in classes:
                self.addItem(class_name)

        # Adjust the height of the QListWidget based on the content
        self.adjust_height()

    def item_selected(self):
        # get the text of the currently selected item
        selected_index = self.currentRow()
        print(f'Selected index: {selected_index}')

    def adjust_height(self):
        # The minimum height of QListWidget
        min_height = 100  # Set this to your desired minimum height

        # Calculate the total height of all items
        total_height = sum([self.sizeHintForRow(i) for i in range(self.count())])

        # Add the spacing (margin) around items in the list
        total_height += self.count() * self.spacing()

        # Ensure the QListWidget height is not less than the minimum height
        self.setMaximumHeight(max(total_height, min_height))
        
class UiElements:
    def __init__(self, viewer):
        self.layer_types = {"image": napari.layers.image.image.Image, "labels": napari.layers.labels.labels.Labels}
        self.viewer = viewer
        self.annotator_mode = AnnotatorMode.NONE

        #### overview of interactive UI elements ####
        self.cached_models = None
        self.loaded_model = None

        self.cb_model_selctor = None
        self.btn_load_model = None
        self.cb_input_image_selctor = None
        self.rb_click_mode = None
        self.rb_annotation_mode_automatic = None
        self.cs_class_selector = None
        self.btn_activate = None
        self.btn_submit_to_class = None

        self.le_points_per_side = None
        self.le_points_per_batch = None
        self.le_prediction_iou_threshold = None
        self.le_stability_score_threshold = None
        self.le_stability_score_offset = None
        self.le_box_nms_threshold = None
        self.le_crop_n_layers = None
        self.le_crop_nms_threshold = None
        self.le_crop_overlap_ratio = None
        self.le_crop_n_points_downscale_factor = None
        self.le_minimum_mask_region_area = None
        

        self._init_main_layout()
        self._init_UI_signals()
        self._update_UI()

    ################################ UI elements ################################

    def _init_main_layout(self):
        self.main_layout = QVBoxLayout()
        self._init_model_selection()
        self._init_input_image_selection()
        self._init_output_label_selection()
        self._init_segmentation_profile_selection()
        self._init_annotation_mode()
        self._init_activation_button()
        self._init_class_selection()
        self._init_submit_to_class_button()
        self._init_tooltip()
        self.init_auto_mode_settings()

    def _init_model_selection(self):
        l_model_type = QLabel("Select model type:")
        self.main_layout.addWidget(l_model_type)

        self.cb_model_selctor = QComboBox()
        self.main_layout.addWidget(self.cb_model_selctor)
        
        self.btn_load_model = QPushButton("Load model") #### Callback function is defined externally through a setter function below
        self.main_layout.addWidget(self.btn_load_model)

        self._update_model_selection_combobox_and_button()

    def _init_input_image_selection(self):
        l_input_image = QLabel("Select input image:")
        self.main_layout.addWidget(l_input_image)

        self.cb_input_image_selctor = QComboBox()
        self._update_layer_selection_combobox(self.cb_input_image_selctor, napari.layers.Image)
        self.main_layout.addWidget(self.cb_input_image_selctor)
    
    def _init_output_label_selection(self):
        l_output_label = QLabel("Select output label:")
        self.main_layout.addWidget(l_output_label)

        self.cb_output_label_selctor = QComboBox()
        self._update_layer_selection_combobox(self.cb_output_label_selctor, napari.layers.Labels)
        self.main_layout.addWidget(self.cb_output_label_selctor)

    def _init_segmentation_profile_selection(self):
        l_segmentation_profile = QLabel("Select segmentation profile:")
        self.main_layout.addWidget(l_segmentation_profile)

        self.cb_segmentation_profile_selctor = QComboBox()
        self._update_layer_selection_combobox(self.cb_segmentation_profile_selctor, napari.layers.Points)
        self.main_layout.addWidget(self.cb_segmentation_profile_selctor)

    def _init_annotation_mode(self):
        self.g_annotation = QGroupBox("Annotation mode")
        self.l_annotation = QVBoxLayout()

        self.rb_annotation_mode_click = QRadioButton("Click")
        self.rb_annotation_mode_click.setChecked(True)
        self.rb_annotation_mode_click.setToolTip(
            "Positive Click: Middle Mouse Button\n \n"
            "Negative Click: Control + Middle Mouse Button \n \n"
            "Undo: Control + Z \n \n"
            "Select Point: Left Click \n \n"
            "Delete Selected Point: Delete"
        )
        self.l_annotation.addWidget(self.rb_annotation_mode_click)

        self.rb_annotation_mode_automatic = QRadioButton("Automatic mask generation")
        self.rb_annotation_mode_automatic.setToolTip(
            "Creates automatically an instance segmentation \n"
            "of the entire image.\n"
            "No user interaction possible."
        )
        self.l_annotation.addWidget(self.rb_annotation_mode_automatic)

        self.g_annotation.setLayout(self.l_annotation)
        self.main_layout.addWidget(self.g_annotation)

    def _init_class_selection(self):
        l_class_selector = QLabel("Select class:")
        self.main_layout.addWidget(l_class_selector)

        self.cs_class_selector = ClassSelector(self.viewer)  # No callback function
        self.cs_class_selector.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.main_layout.addWidget(self.cs_class_selector)

    def _init_activation_button(self):
        # Callback function is defined externally through a setter function below
        self.btn_activate = QPushButton("Activate")
        self.btn_activate.setEnabled(False)
        self.main_layout.addWidget(self.btn_activate)

    def _init_submit_to_class_button(self):
        # Callback function is defined externally through a setter function below
        self.btn_submit_to_class = QPushButton("submit to class")
        self.btn_submit_to_class.setEnabled(False)
        self.main_layout.addWidget(self.btn_submit_to_class)

    def _init_tooltip(self):
        container_widget_info = QWidget()
        container_layout_info = QVBoxLayout(container_widget_info)


        self.g_info_tooltip = QGroupBox("Tooltip Information")
        self.l_info_tooltip = QVBoxLayout()
        self.label_info_tooltip = QLabel("Every mode shows further information when hovered over.")
        self.label_info_tooltip.setWordWrap(True)
        self.l_info_tooltip.addWidget(self.label_info_tooltip)
        self.g_info_tooltip.setLayout(self.l_info_tooltip)
        container_layout_info.addWidget(self.g_info_tooltip)

        """ 
        self.g_info_contrast = QGroupBox("Contrast Limits")
        self.l_info_contrast = QVBoxLayout()
        self.label_info_contrast = QLabel("SAM computes its image embedding based on the current image contrast.\n"
                                          "Image contrast can be adjusted with the contrast slider of the image layer.")
        self.label_info_contrast.setWordWrap(True)
        self.l_info_contrast.addWidget(self.label_info_contrast)
        self.g_info_contrast.setLayout(self.l_info_contrast)
        container_layout_info.addWidget(self.g_info_contrast)
         """
        
        self.g_info_click = QGroupBox("Click Mode")
        self.l_info_click = QVBoxLayout()
        self.label_info_click = QLabel("\nPositive Click: Middle Mouse Button\n \n"
                                 "Negative Click: Control + Middle Mouse Button\n \n"
                                 "Undo: Control + Z\n \n"
                                 "Select Point: Left Click\n \n"
                                 "Delete Selected Point: Delete\n \n"
                                 "Pick Label: Control + Left Click\n \n"
                                 "Increment Label: M\n \n")
        self.label_info_click.setWordWrap(True)
        self.l_info_click.addWidget(self.label_info_click)
        self.g_info_click.setLayout(self.l_info_click)
        container_layout_info.addWidget(self.g_info_click)

        self.scroll_area_click = QScrollArea()
        self.scroll_area_click.setWidget(container_widget_info)
        self.scroll_area_click.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.main_layout.addWidget(self.scroll_area_click)

    def init_auto_mode_settings(self):
        container_widget_auto = QWidget()
        container_layout_auto = QVBoxLayout(container_widget_auto)

        # self.g_auto_mode_settings = QCollapsibleBox("Everything Mode Settings")
        self.g_auto_mode_settings = QGroupBox("Everything Mode Settings")
        self.l_auto_mode_settings = QVBoxLayout()

        l_points_per_side = QLabel("Points per side:")
        self.l_auto_mode_settings.addWidget(l_points_per_side)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_points_per_side = QLineEdit()
        self.le_points_per_side.setText("32")
        self.le_points_per_side.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_points_per_side)

        l_points_per_batch = QLabel("Points per batch:")
        self.l_auto_mode_settings.addWidget(l_points_per_batch)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_points_per_batch = QLineEdit()
        self.le_points_per_batch.setText("64")
        self.le_points_per_batch.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_points_per_batch)

        l_pred_iou_thresh = QLabel("Prediction IoU threshold:")
        self.l_auto_mode_settings.addWidget(l_pred_iou_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_prediction_iou_threshold = QLineEdit()
        self.le_prediction_iou_threshold.setText("0.88")
        self.le_prediction_iou_threshold.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_prediction_iou_threshold)

        l_stability_score_thresh = QLabel("Stability score threshold:")
        self.l_auto_mode_settings.addWidget(l_stability_score_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_stability_score_threshold = QLineEdit()
        self.le_stability_score_threshold.setText("0.95")
        self.le_stability_score_threshold.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_stability_score_threshold)

        l_stability_score_offset = QLabel("Stability score offset:")
        self.l_auto_mode_settings.addWidget(l_stability_score_offset)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_stability_score_offset = QLineEdit()
        self.le_stability_score_offset.setText("1.0")
        self.le_stability_score_offset.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_stability_score_offset)

        l_box_nms_thresh = QLabel("Box NMS threshold:")
        self.l_auto_mode_settings.addWidget(l_box_nms_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_box_nms_threshold = QLineEdit()
        self.le_box_nms_threshold.setText("0.7")
        self.le_box_nms_threshold.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_box_nms_threshold)

        l_crop_n_layers = QLabel("Crop N layers")
        self.l_auto_mode_settings.addWidget(l_crop_n_layers)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_crop_n_layers = QLineEdit()
        self.le_crop_n_layers.setText("0")
        self.le_crop_n_layers.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_n_layers)

        l_crop_nms_thresh = QLabel("Crop NMS threshold:")
        self.l_auto_mode_settings.addWidget(l_crop_nms_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_crop_nms_threshold = QLineEdit()
        self.le_crop_nms_threshold.setText("0.7")
        self.le_crop_nms_threshold.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_nms_threshold)

        l_crop_overlap_ratio = QLabel("Crop overlap ratio:")
        self.l_auto_mode_settings.addWidget(l_crop_overlap_ratio)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_crop_overlap_ratio = QLineEdit()
        self.le_crop_overlap_ratio.setText("0.3413")
        self.le_crop_overlap_ratio.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_overlap_ratio)

        l_crop_n_points_downscale_factor = QLabel("Crop N points downscale factor")
        self.l_auto_mode_settings.addWidget(l_crop_n_points_downscale_factor)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_crop_n_points_downscale_factor = QLineEdit()
        self.le_crop_n_points_downscale_factor.setText("1")
        self.le_crop_n_points_downscale_factor.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_n_points_downscale_factor)

        l_min_mask_region_area = QLabel("Min mask region area")
        self.l_auto_mode_settings.addWidget(l_min_mask_region_area)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_minimum_mask_region_area = QLineEdit()
        self.le_minimum_mask_region_area.setText("0")
        self.le_minimum_mask_region_area.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_minimum_mask_region_area)

        # self.g_auto_mode_settings.setContentLayout(self.l_auto_mode_settings)
        self.g_auto_mode_settings.setLayout(self.l_auto_mode_settings)
        # main_layout.addWidget(self.g_auto_mode_settings)
        container_layout_auto.addWidget(self.g_auto_mode_settings)

        self.scroll_area_auto = QScrollArea()
        # scroll_area_info.setWidgetResizable(True)
        self.scroll_area_auto.setWidget(container_widget_auto)
        # Set the scrollbar policies for the scroll area
        self.scroll_area_auto.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # scroll_area_info.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area_auto.hide()

        self.main_layout.addWidget(self.scroll_area_auto)
   
    def _init_UI_signals(self):
        """ connecting signals for pure UI elements interactions """
        self.cb_model_selctor.currentTextChanged.connect(self._update_model_selection_combobox_and_button)
        self.btn_load_model.clicked.connect(self._internal_handler_btn_load_model) 
        self.btn_activate.clicked.connect(self._internal_handler_btn_activate)
        self.btn_submit_to_class.clicked.connect(self._internal_handler_btn_submit_to_class)
        
        self.viewer.layers.events.inserted.connect(self._update_UI) # TODO make spacial cases instead of updating everything
        self.viewer.layers.events.removed.connect(self._update_UI)
        
        self.cb_input_image_selctor.currentTextChanged.connect(self._check_activate_btn)
        self.cb_output_label_selctor.currentTextChanged.connect(self._check_activate_btn)
        self.cb_segmentation_profile_selctor.currentTextChanged.connect(self._check_activate_btn)

        self.cb_segmentation_profile_selctor.currentTextChanged.connect(self._update_class_selector)

        self.rb_annotation_mode_click.clicked.connect(self._update_tooltip)
        self.rb_annotation_mode_automatic.clicked.connect(self._update_tooltip)

    def _update_UI(self):
        self._update_layer_selection_combobox(self.cb_input_image_selctor, napari.layers.Image)
        self._update_layer_selection_combobox(self.cb_output_label_selctor, napari.layers.Labels)
        self._update_layer_selection_combobox(self.cb_segmentation_profile_selctor, napari.layers.Shapes)

        self._update_class_selector()

        self._check_activate_btn()
  

    ################################ internal signals ################################

    def _update_layer_selection_combobox(self, layer_selection_combobox, layers_type):
        layer_selection_combobox.clear()

        for layer in self.viewer.layers:
            if isinstance(layer, layers_type):
                if layers_type == napari.layers.Shapes and 'classes' not in layer.metadata: # spacial case for segmentation profile
                    continue
                layer_selection_combobox.addItem(layer.name)
                
    def _update_class_selector(self):
        if self.cb_segmentation_profile_selctor.currentText() != "":
            selected_layer = self.viewer.layers[self.cb_segmentation_profile_selctor.currentText()]
            self.cs_class_selector.update_classes(selected_layer.metadata['classes'])

        else:
                self.cs_class_selector.clear()

    def _update_model_selection_combobox_and_button(self):
        """Updates the model selection combobox and load model button based on the cached models."""

        # Disconnect the signal if it's connected to avoid triggering the signal when clearing the combobox
        if self.cb_model_selctor.receivers(self.cb_model_selctor.currentTextChanged) > 0: 
            self.cb_model_selctor.currentTextChanged.disconnect()

        self.cached_models, combobox_models = get_cached_models(SAM_MODELS, self.loaded_model)

        current_selection_index = self.cb_model_selctor.currentIndex()

        self.cb_model_selctor.clear()
        self.cb_model_selctor.addItems(combobox_models)

        # Reselect the previously selected item
        self.cb_model_selctor.setCurrentIndex(current_selection_index)

        if self.cached_models[list(self.cached_models.keys())[self.cb_model_selctor.currentIndex()]]:
            self.btn_load_model.setText("Load model")
        else:
            self.btn_load_model.setText("Download and load model")

        # Reconnect the signal
        self.cb_model_selctor.currentTextChanged.connect(self._update_model_selection_combobox_and_button)

    def _internal_handler_btn_load_model(self):
        self.cb_model_selctor.setEnabled(False)
        self.btn_load_model.setEnabled(False)

        model_types = list(SAM_MODELS.keys())
        model_type = model_types[self.cb_model_selctor.currentIndex()]

        self.external_handler_btn_load_model(model_type)

        self.loaded_model = model_type
        self._update_model_selection_combobox_and_button()
        self.cb_model_selctor.setEnabled(True)
        self.btn_load_model.setEnabled(True)
        self._check_activate_btn()

    def _internal_handler_btn_activate(self):
        self.btn_activate.setEnabled(False)
        if self.btn_activate.text() == "Activate":
            self.cb_model_selctor.setEnabled(False)
            self.btn_load_model.setEnabled(False)
            self.cb_input_image_selctor.setEnabled(False)
            self.cb_output_label_selctor.setEnabled(False)
            self.cb_segmentation_profile_selctor.setEnabled(False)
            self.btn_submit_to_class.setEnabled(True)

            if self.rb_annotation_mode_click.isChecked():
                self.annotator_mode = AnnotatorMode.CLICK
                self.rb_annotation_mode_automatic.setEnabled(False)
                self.rb_annotation_mode_automatic.setStyleSheet("color: gray")
            
            elif self.rb_annotation_mode_automatic.isChecked():
                self.annotator_mode = AnnotatorMode.AUTO
                self.rb_annotation_mode_click.setEnabled(False)
                self.rb_annotation_mode_click.setStyleSheet("color: gray")
        
            self.btn_activate.setText("Deactivate")
            self.external_handler_btn_activate(self.annotator_mode)
        
        else:
            self.external_handler_btn_deactivate()

            self.cb_model_selctor.setEnabled(True)
            self.btn_load_model.setEnabled(True)
            self.cb_input_image_selctor.setEnabled(True)
            self.cb_output_label_selctor.setEnabled(True)
            self.cb_segmentation_profile_selctor.setEnabled(True)
            
            self.rb_annotation_mode_click.setEnabled(True)
            self.rb_annotation_mode_click.setStyleSheet("")
            self.rb_annotation_mode_automatic.setEnabled(True)
            self.rb_annotation_mode_automatic.setStyleSheet("")

            self.btn_submit_to_class.setEnabled(False)

            self.btn_activate.setText("Activate")

        self.btn_activate.setEnabled(True)

    def _internal_handler_btn_submit_to_class(self):
        self.external_handler_btn_submit_to_class(self.cs_class_selector.currentRow()+1)

    def _check_activate_btn(self):
        if (
            self.loaded_model is not None and
            self.cb_input_image_selctor.currentText() != "" and
            self.cb_output_label_selctor.currentText() != "" and
            self.cb_segmentation_profile_selctor.currentText() != ""
        ):
            self.btn_activate.setEnabled(True)
        else:
            self.btn_activate.setEnabled(False)

    def _update_tooltip(self):
        if self.rb_annotation_mode_click.isChecked():
            self.scroll_area_click.show()
        else:
            self.scroll_area_click.hide()
        if self.rb_annotation_mode_automatic.isChecked():
            self.scroll_area_auto.show()
        else:
            self.scroll_area_auto.hide()


    ################################ set external signals ################################

    def set_external_handler_btn_load_model(self, handler):
        self.external_handler_btn_load_model = handler

    def set_external_handler_btn_activate(self, activate_handler, deactivate_handler):
        self.external_handler_btn_activate = activate_handler
        self.external_handler_btn_deactivate = deactivate_handler

    def set_external_handler_btn_submit_to_class(self, handler):
        self.external_handler_btn_submit_to_class = handler


    ################################ externally activated UI elements ################################

    def create_progress_bar(self, max_value, text):
        self.l_creating_features = QLabel(text)
        self.main_layout.addWidget(self.l_creating_features)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(max_value)
        self.progress_bar.setValue(0)
        self.main_layout.addWidget(self.progress_bar)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        QApplication.processEvents()

    def delete_progress_bar(self):
        self.progress_bar.deleteLater()
        self.l_creating_features.deleteLater()


    ################################ utilities ################################

def get_cached_models(SAM_MODELS: dict, loaded_model: str) -> tuple:
    """Check if the weights of the SAM models are cached locally."""
    model_types = list(SAM_MODELS.keys())
    cached_models: dict = {}
    cache_dir: str = str(Path.home() / ".cache/napari-segment-anything")

    for model_type in model_types:
        filename = os.path.basename(SAM_MODELS[model_type]["filename"])
        if os.path.isfile(os.path.join(cache_dir, filename)):
            cached_models[model_type] = True
        else:
            cached_models[model_type] = False

    """ creates a list of strings for the combobox """
    entries: list = []
    for name, is_cached in cached_models.items():
        if name == loaded_model:
            entries.append(f"{name} (Loaded)")
        elif is_cached:
            entries.append(f"{name} (Cached)")
        else:
            entries.append(f"{name} (Auto-Download)")
    return cached_models, entries
