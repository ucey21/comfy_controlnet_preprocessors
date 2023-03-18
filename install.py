import os
from time import sleep
from importlib.util import spec_from_file_location, module_from_spec
import sys

module_name = "comfy_controlnet_preprocessors"
EXT_PATH = os.path.dirname(os.path.realpath(__file__))
def add_global_shortcut_module():
    #Naming things is hard
    module_spec = spec_from_file_location(module_name, os.path.join(EXT_PATH, "__init__.py"))
    module = module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)

def download_models():
    canny.CannyDetector()
    hed.HEDdetector()
    midas.MidasDetector()
    mlsd.MLSDdetector()
    openpose.OpenposeDetector()
    uniformer.UniformerDetector()

print("Installing requirements...")
sleep(2)
os.system(f"pip install -r {EXT_PATH}/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117 --no-warn-script-location")

add_global_shortcut_module()
from comfy_controlnet_preprocessors import canny, hed, midas, mlsd, openpose, uniformer

print("Download models...")
sleep(2)
download_models()
print("Done!")