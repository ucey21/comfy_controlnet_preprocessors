from .util import common_annotator_call, img_np_to_tensor
from ..v1 import uniformer
from ..v11 import tile
from .. import mp_face_mesh, color


class Uniformer_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", )
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "preprocessors/semseg"

    def semantic_segmentate(self, image):
        #Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_seg2image.py
        np_detected_map = common_annotator_call(uniformer.UniformerDetector(), image)
        return (img_np_to_tensor(np_detected_map),)

class Media_Pipe_Face_Mesh_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", ),
                              "max_faces": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}), #Which image has more than 50 detectable faces?
                              "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.1})
                            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"

    CATEGORY = "preprocessors/face_mesh"

    def detect(self, image, max_faces, min_confidence):
        np_detected_map = common_annotator_call(mp_face_mesh.generate_annotation, image, max_faces, min_confidence)
        return (img_np_to_tensor(np_detected_map),)



class Color_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",) }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_processed_pallete"

    CATEGORY = "preprocessors/color_style"

    def get_processed_pallete(self, image):
        np_detected_map = common_annotator_call(color.apply_color, image)
        return (img_np_to_tensor(np_detected_map),)



class Tile_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "pyrUp_iters": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}) }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"

    CATEGORY = "preprocessors/tile"

    def preprocess(self, image, pyrUp_iters):
        np_detected_map = common_annotator_call(tile.preprocess, image, pyrUp_iters)
        return (img_np_to_tensor(np_detected_map),)

NODE_CLASS_MAPPINGS = {
    "SemSegPreprocessor": Uniformer_Preprocessor,
    "MediaPipe-FaceMeshPreprocessor": Media_Pipe_Face_Mesh_Preprocessor,
    "ColorPreprocessor": Color_Preprocessor
}
