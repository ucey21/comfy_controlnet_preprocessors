# ControlNet Preprocessors for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
Moved from https://github.com/comfyanonymous/ComfyUI/pull/13 <br>
Original repo: https://github.com/lllyasviel/ControlNet <br>
List of my comfyUI node repos: https://github.com/Fannovel16/FN16-ComfyUI-nodes <br>
Require free space of 1070.15MB to download all ckpts needed for this repo. <br>
The input images can have any kind of resolution, not need to be multiple of 64. They will be resized to fit the nearest multiple-of-64 resolution behind the scene.
## Install
Firstly, [install comfyui's dependencies](https://github.com/comfyanonymous/ComfyUI#installing) if you didn't.
Then run:
```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfy_controlnet_preprocessors
cd comfy_controlnet_preprocessors
```
Next, run instal.py.
For directly-cloned ComfyUI repo:
```
python install.py
```
For ComfyUI portable standalone build:
```
#You may need to replace "..\..\..\python_embeded\python.exe" depends your python_embeded location
..\..\..\python_embeded\python.exe install.py
```
## Citation

    @misc{zhang2023adding,
      title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
      author={Lvmin Zhang and Maneesh Agrawala},
      year={2023},
      eprint={2302.05543},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

[Arxiv Link](https://arxiv.org/abs/2302.05543)
