import os
from . import canny, hed, midas, mlsd, openpose, uniformer 
os.system("pip install -r requirements.txt")

def download_models():
    canny.CannyDetector()
    hed.HEDdetector()
    midas.MidasDetector()
    mlsd.MLSDdetector()
    openpose.OpenposeDetector()
    uniformer.UniformerDetector()

download_models()