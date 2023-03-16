# ControlNet Preprocessors for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
Moved from https://github.com/comfyanonymous/ComfyUI/pull/13 <br>
Original repo: https://github.com/lllyasviel/ControlNet <br>
List of my comfyUI node repos: https://github.com/Fannovel16/FN16-ComfyUI-nodes <br>
Require free space of 1070.15MB to download all ckpts needed for this repo.
## Install
Firstly, run:
```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfy_controlnet_preprocessors
cd comfy_controlnet_preprocessors
```
For directly-cloned ComfyUI repo:
```
python install.py
```
For ComfyUI portable standalone build:
```
#You may need to replace "..\..\..\python_embeded\python.exe" depends your python_embeded location
#Ref: https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/2#issuecomment-1471005717
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117 --no-warn-script-location
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
