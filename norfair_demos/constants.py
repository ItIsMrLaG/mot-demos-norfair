import os
from pathlib import Path

from __init__ import BASE_DIR

STATIC_FILES = BASE_DIR / "cfg_files"

DETECTRON2_CFG = STATIC_FILES / "detectron2_config.yaml"
DETECTRON2_MODEL_W = STATIC_FILES / "model_final_f10217.pkl"
