import os
import random
import cv2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

# ---------- 1️⃣ Register your dataset ----------
register_coco_instances(
    "my_dataset",
    {},
    "dataset/annotations/annotations.json",
    "dataset/images"
)

metadata = MetadataCatalog.get("my_dataset")
dataset_dicts = DatasetCatalog.get("my_dataset")

# ---------- 2️⃣ Config ----------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = os.path.join("./output", "model_final.pth")  # <-- resume here
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00005  # smaller for fine-tuning
cfg.SOLVER.MAX_ITER = 1500      # continue training longer
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ---------- 3️⃣ Train ----------
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)  # <-- resume
trainer.train()
