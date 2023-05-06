import numpy as np
import sys
from typing import Any, Dict, List, Optional, Tuple
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class Predictor():
    def __init__(self, model_path="model/sam_vit_h_4b8939.pth", device="cpu", model_type="vit_h"):
        model_type = "vit_h"
        device = "cpu"
        self._sam = sam_model_registry[model_type](checkpoint=model_path)
        self._sam.to(device=device)

        # 加载Segment Anything模型
        self._sam_predictor = SamPredictor(self._sam)
        self._auto_predictor = SamAutomaticMaskGenerator(self._sam, output_mode="uncompressed_rle")

    def generate(self, image: np.ndarray
                     , input_points: Optional[np.ndarray] = None
                     , input_labels : Optional[np.ndarray] = None
                     , input_boxs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        self._sam_predictor.set_image(image)
        masks, scores, _ = self._sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_boxs,
            multimask_output=True,
        )
        return masks,scores

    def generate_auto(self, image: np.ndarray) -> List[Dict[str, Any]]:
        return self._auto_predictor.generate(image)

    def batch_generate(self, image: np.ndarray):
        pass