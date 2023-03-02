import os
from time import sleep
from . import canny, hed, midas, mlsd, openpose, uniformer
print("Installing requirements...")
sleep(2)
os.system("pip install -r requirements.txt")

def download_models():
    canny.CannyDetector()
    hed.HEDdetector()
    midas.MidasDetector()
    mlsd.MLSDdetector()
    openpose.OpenposeDetector()
    uniformer.UniformerDetector()
print("Download models...")
sleep(2)
download_models()
print("Done!")