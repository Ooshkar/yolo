import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

# Register your COCO dataset
register_coco_instances(
    "f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1",
    {},
    "dataset/annotations/annotations.json",
    "dataset/images"
)

# Load an image
img_path = "dataset/images/Screenshot_20260115_195301.png"
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Image not found or cannot be read: {img_path}")

# 1️⃣ Setup cfg like in training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./output/model_final.pth"  # path to your trained model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # match your dataset
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold
cfg.DATASETS.TEST = ("f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1",)

# 2️⃣ Create predictor
predictor = DefaultPredictor(cfg)

# 3️⃣ Make predictions
outputs = predictor(img)

# 4️⃣ Visualize predictions
metadata = MetadataCatalog.get("f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1")
v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# 5️⃣ Show image
cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
