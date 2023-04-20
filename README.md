# ControlNet Preprocessors for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
### NOTE: IF YOU USE COMFYUI STANDALONE BUILD, YOU HAVE TO USE LOCAL PYTHON FROM THAT ONE (`python_embeded`) TO EXECUTE `install.py`, NOT PYTHON FROM THE SYSTEM. CHECK INSTALL SECTION FOR MORE DETAILS. 
Moved from https://github.com/comfyanonymous/ComfyUI/pull/13 <br>
Original repo: https://github.com/lllyasviel/ControlNet <br>
List of my comfyUI node repos: https://github.com/Fannovel16/FN16-ComfyUI-nodes <br>

## Change log:
### 2023-04-01
* Rename MediaPipePreprocessor to MediaPipe-PoseHandPreprocessor to avoid confusion
* Add MediaPipe-FaceMeshPreprocessor for [ControlNet Face Model](https://www.reddit.com/r/StableDiffusion/comments/1281iva/new_controlnet_face_model/)
### 2023-04-02
* Fixed https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/20
* Fixed typo at ##Nodes
### 2023-04-10
* Fixed https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/18, https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/25: https://github.com/Fannovel16/comfy_controlnet_preprocessors/commit/b8a108a0f8ae37b9302b32d7c236cfa3dde97920, https://github.com/Fannovel16/comfy_controlnet_preprocessors/commit/01fbab5cdfc7b013256d4aec4e5ad77edb80a039
### 2023-04-20
* Add HED-v11-Preprocessor, PiDiNet-v11-Preprocessor, Zoe-DepthMapPreprocessor and BAE-NormalMapPreprocessor
* If you installed this repo, please run `install.py` after `git pull` if you want to download all files for new preprocessors for offline run
* Remove the needs of BasicSR (https://github.com/Fannovel16/comfy_controlnet_preprocessors/commit/6b073101f75d6ab1e53c231dab8118990fec96ed) since ComfyUI's custom Python build can't install it.

## Usage
All preprocessor nodes take an image, usually came from LoadImage node and output a map image (aka hint image):
* The input image can have any kind of resolution, not need to be multiple of 64. They will be resized to fit the nearest multiple-of-64 resolution behind the scene.
* The hint image is a black canvas with a/some subject(s) like Openpose stickman(s), depth map, etc

## Nodes
TL;DR: You should only use these preprocessor nodes when using ControlNet 1.1:
* CannyEdgePreprocessor
* M-LSDPreprocessor
* HED-v11-Preprocessor (Use HEDPreprocesor if you use ControlNet v1)
* PiDiNet-v11-Preprocessor (Use PiDiNetPreprocesor if you use ControlNet v1)
* ScribblePreprocessor
* FakeScribblePreprocessor
* BinaryPreprocessor
* MiDaS-DepthMapPreprocessor
* LeReS-DepthMapPreprocessor
* BAE-NormalMapPreprocessor (MiDaS-NormalMapPreprocessor doesn't work well with control_v11p_sd15_normalbae as this model made specifically for BAE's normal map)
* Openpose-v11-Preprocessor (Comming Soon)
* SemSeg-Preprocessor
* Zoe-DepthMapPreprocessor
A simple rule to remember is that you should use preprocess with "v11" in name instead of one not having it if it exists. <br>
You shouldn't use v11-only preprocessors with ControlNet v1.

TODO: Refactor this section and update
### preprocessors/edge_line
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| CannyEdgePreprocessor       | canny                                                 | control_canny/control_v11p_sd15_canny <br> t2iadapter_canny | preprocessors/edge_line |
| M-LSDPreprocessor           | mlsd                                                  | control_mlsd/control_v11p_sd21_canny      | preprocessors/edge_line          |

### Full
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| HEDPreprocessor             | hed                                                   | control_hed                               | preprocessors/edge_line          |
| PiDiNetPreprocessor         | pidinet                                               | t2iadapter_sketch <br> control_scribble   | preprocessors/edge_line          |
| ScribblePreprocessor        | scribble                                              | control_scribble                          | preprocessors/edge_line          |
| FakeScribblePreprocessor    | fake_scribble                                         | control_scribble                          | preprocessors/edge_line          |
| BinaryPreprocessor          | binary                                                | control_scribble                          | preprocessors/edge_line          |
| MiDaS-DepthMapPreprocessor  | (normal) depth                                        | control_depth <br> t2iadapter_depth       | preprocessors/normal_depth_map   |
| MiDaS-NormalMapPreprocessor | normal_map                                            | control_normal                            | preprocessors/normal_depth_map   |
| LeReS-DepthMapPreprocessor  | depth_leres                                           | control_depth <br> t2iadapter_depth       | preprocessors/normal_depth_map   |
| OpenposePreprocessor        | openpose (or openpose_hand if detect_hand is enabled) | control_openpose <br> t2iadapter_openpose | preprocessors/pose               |
|MediaPipe-PoseHandPreprocessor| https://natakaro.gumroad.com/l/oprmi                 | https://civitai.com/models/16409         | preprocessors/pose                |
| ColorPreprocessor           | color                                                 | t2iadapter_color                          | preprocessors/color_style        |
| SemSegPreprocessor          | segmentation                                          | control_seg <br> t2iadapter_seg           | preprocessors/semseg             |
|MediaPipe-FaceMeshPreprocessor| mediapipe_face                                       | controlnet_sd21_laion_face_v2             | preprocessors/face_mesh          |

## Install
Firstly, [install comfyui's dependencies](https://github.com/comfyanonymous/ComfyUI#installing) if you didn't.
Then run:
```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfy_controlnet_preprocessors
cd comfy_controlnet_preprocessors
```
Next, run install.py. It will download all models by default. <br>
Note that you have to check if ComfyUI you are using is portable standalone build or not. If you use the wrong command, requirements won't be installed in the right place. 
For directly-cloned ComfyUI repo, run:
```
python install.py
```
For ComfyUI portable standalone build:
```
/path/to/ComfyUI/python_embeded/python.exe install.py
```
Add `--no_download_ckpts` to not download any model. <br>
When a preprocessor node runs, if it can't find the models it need, that models will be downloaded automatically.
## Model Dependencies 
The total disk's free space needed if all models are downloaded is ~1.58 GB. <br>
All models will be downloaded to `comfy_controlnet_preprocessors/ckpts`
* network-bsds500.pth (hed): 56.1 MB
* res101.pth (leres): 506 MB
* dpt_hybrid-midas-501f0c75.pt (midas): 470 MB
* mlsd_large_512_fp32.pth (mlsd): 6 MB
* body_pose_model.pth (for both openpose v1 and v1.1): 200 MB
* hand_pose_model.pth (for both openpose v1 and v1.1): 141 MB
* facenet.pth (openpose v1.1): 154 MB
* upernet_global_small.pth (uniformer aka SemSeg): 197 MB
* table5_pidinet.pth (for both PiDiNet v1 and v1.1): 2.87 MB
* ControlNetHED.pth (New HED 1.1): 29.4 MB
* scannet.pt (NormalBAE): 291 MB

## Limits
* There may be bugs since I don't have time ~~(lazy)~~ to test
* ~~You must have CUDA device because I just put `.cuda()` everywhere.~~ It is fixed

## Citation
### Original ControlNet repo
    @misc{zhang2023adding,
      title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
      author={Lvmin Zhang and Maneesh Agrawala},
      year={2023},
      eprint={2302.05543},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

[Arxiv Link](https://arxiv.org/abs/2302.05543)

### Mikubill/sd-webui-controlnet
https://github.com/Mikubill/sd-webui-controlnet
### Others:
* https://natakaro.gumroad.com/l/oprmi
