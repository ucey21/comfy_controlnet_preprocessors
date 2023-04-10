import os
from time import sleep
from importlib.util import spec_from_file_location, module_from_spec
import sys
import argparse
import subprocess

this_module_name = "comfy_controlnet_preprocessors"
EXT_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--no_download_ckpts', action="store_true", help="Don't download any model")

args = parser.parse_args()

def add_global_shortcut_module(module_name, module_path):
    #Naming things is hard
    module_spec = spec_from_file_location(module_name, module_path)
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
    leres.download_model_if_not_existed()
    pidinet.download_if_not_existed()

command = [sys.executable, '-m','pip', 'install', '-r', f'{EXT_PATH}/requirements.txt', '--extra-index-url', 'https://download.pytorch.org/whl/cu117', '--no-warn-script-location']
print("Installing requirements...")
sleep(2)
proc = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)
for line in iter(proc.stdout.readline, ''):
    print(line, end='')
proc.wait()

if args.no_download_ckpts: exit()

add_global_shortcut_module("cli_args", os.path.join(EXT_PATH, "../../comfy/cli_args.py"))
add_global_shortcut_module("model_management", os.path.join(EXT_PATH, "../../comfy/model_management.py"))
add_global_shortcut_module(this_module_name, os.path.join(EXT_PATH, "__init__.py"))
from comfy_controlnet_preprocessors import canny, hed, midas, mlsd, openpose, uniformer, leres, pidinet
print("Download models...")
sleep(2)
download_models()
print("Done!")
