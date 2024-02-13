# Dream2Real

[ICRA 2024] **Dream2Real: Zero-Shot 3D Object Rearrangement with Vision-Language Models**<br/>Ivan Kapelyukh\*, Yifei Ren\*, Ignacio Alzugaray, Edward Johns <br/>
(\* Joint first authorship) </br>
[[Webpage]](https://www.robot-learning.uk/dream2real) [[Paper PDF]](https://arxiv.org/pdf/2312.04533.pdf)

<p align="center">
    <img src="teaser/video.gif">
</p>

## Summary

Dream2Real imagines 3D goal states for robotic rearrangement tasks. It works zero-shot, without requiring any example arrangements, using NeRF and VLMs.

![Figure 1](teaser/figure_1.png)

## Abstract

We introduce Dream2Real, a robotics framework which integrates vision-language models (VLMs) trained on 2D data into a 3D object rearrangement pipeline. This is achieved by the robot autonomously constructing a 3D representation of the scene, where objects can be rearranged virtually and an image of the resulting arrangement rendered. These renders are evaluated by a VLM, so that the arrangement which best satisfies the user instruction is selected and recreated in the real world with pick-and-place. This enables language-conditioned rearrangement to be performed zero-shot, without needing to collect a training dataset of example arrangements. Results on a series of real-world tasks show that this framework is robust to distractors, controllable by language, capable of understanding complex multi-object relations, and readily applicable to both tabletop and 6-DoF rearrangement tasks.

## Installation

### Requirements

1. Ubuntu 20.04 (this code may also work with other versions but has not been tested).

2. Hardware: we have tested on a 24 GB GPU (RTX 4090). Only a single GPU is required. It is possible to run Dream2Real on GPUs with less memory by decreasing the batch sizes used, e.g. when batch-computing CLIP feature vectors for rendered images (at the cost of slower runtime). We also recommend leaving plenty of free space on your SSD/HDD before running, e.g. 60 GB if you wish to run every method/baseline on every dataset. This is because some intermediate output from the method takes a lot of space to store, e.g. video segmentation. However, you may delete this intermediate output after running Dream2Real, if you only care about the final goal pose.

3. CUDA 11.7 (this code may also work with other versions such as CUDA 12 but this has not been thoroughly tested).

4. Miniconda (tested with conda 22.11.1).

5. An OpenAI API key, with access to the [GPT-4 API](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4), since we use this to process the user's language instruction. You will be prompted to enter this key during Dream2Real installation. It will be saved in a file `openai_key.sh` in the root directory of this repository. You may later move this file somewhere else. Remember to keep this key secret and do not publish it on GitHub. You can easily edit the code to use GPT-3.5 if you prefer.

7. The dependencies for Instant-NGP should already be installed before installing Dream2Real. If you have previously used Instant-NGP on your machine, then you have probably already satisfied this requirement. If you have not run Instant-NGP before, then you can follow the instructions [here](https://github.com/NVlabs/instant-ngp) to build and run it. Once you are satisfied that the Instant-NGP dependencies have been installed successfully, you may proceed. Our installation script will clone its own version of Instant-NGP automatically into the correct directory.

### One-Shot Installation

To make Dream2Real easier to use, we have automated the installation steps so that all you need to do is clone the repo and run one script. Please follow these steps:

**Step 1**: Check that the requirements above have been satisfied.

**Step 2**: Clone this repository *(including submodules)* and enter the `dream2real` directory:
```
git clone --recurse-submodules git@github.com:FlyCole/dream2real.git
cd dream2real
```

**Step 3**: Run the install script. This should automatically install Dream2Real:
```
bash install.sh
```

This script might take some time to run (e.g. > 15 minutes), as it needs to set up a conda environment, build Instant-NGP, download datasets, etc. However, you can leave it to run automatically and get a coffee in the meantime!

## Demo

When running Dream2Real, remember to first activate the conda environment and load the OpenAI API key as an environment variable, like so:
```commandline
conda activate dream2real
source openai_key.sh
```

The demo script can be run using the following format:
```commandline
python demo.py DATA_DIR OUT_DIR CFG_PATH USER_INSTR
```
DATA_DIR: Path to the data directory containing raw scanning data. \
OUT_DIR: Path to directory containing intermediate output files e.g. segmentations. \
CONFIG_FILE: Path to config file. Pass path to different config to run different variants/ablations. \
USER_INSTR: The command to be executed by the robot.

The demo script takes as input a scan of the scene performed by the robot, as well as a language command from the user. The output is a predicted goal pose for the object which the robot should move. This can be used as the goal pose for your robotic pick-and-place system. Note that the demo script uses cached intermediate outputs from the system (e.g. already trained NeRFs) so that you can quickly see the results. To run this without cache, set the "use cache" flags to false in the .json config file which you are using.

Example commands are shown below.

#### Shopping
```
python demo.py dataset/shopping method_out/shopping configs/shopping_demo.json "put the apple inside the blue and white bowl"
```
#### Pool Triangle
```
python demo.py dataset/pool_triangle method_out/pool_triangle configs/pool_triangle_demo.json "move the black 8 pool ball so that there is a triangle made of balls on a pool table"
```
#### Pool X Shape
```
python demo.py dataset/pool_X method_out/pool_X configs/pool_X_demo.json "move the black 8 ball so that there are balls in an X shape"
```
#### Shelf
```
python demo.py dataset/shelf method_out/shelf configs/shelf_demo.json "move the strawberry milkshake bottle to make three milkshake bottles standing upright in a neat row"
```

#### Custom Instructions

To use your own instructions, you can modify the commands above. You should also change the .json config file (the path is in the command) to set the "use cache" flags to false. For example, if your cache is for the task of moving an apple and you want to move the orange instead, then you can no longer use the cached NeRF visual models etc. Therefore, set all the "use cache" flags from `use_cache_phys` onwards to false.

## License

This code is available under the CC BY-NC-SA 4.0 license as described here:
https://creativecommons.org/licenses/by-nc-sa/4.0/

Feel free to use it as described in the license, and you can cite this work as shown below to meet the attribution requirement. If you would like to use it for some purpose not currently permitted under the license (e.g. commercial), please get in touch to discuss further.

## Citation

If you found this paper/code useful in your research, you can cite it using this BibTex entry:

```
@inproceedings{dream2real,
  title={{Dream2Real}: Zero-Shot {3D} Object Rearrangement with Vision-Language Models},
  author={Kapelyukh, Ivan and Ren, Yifei and Alzugaray, Ignacio and Johns, Edward},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

## Contact

Feel free to reach out to discuss anything about Dream2Real. Please email ik517@ic.ac.uk and yr820@ic.ac.uk. For help with errors, please first check out the GitHub Issues page, since somebody may have already solved a similar issue in the past.

## FAQ

*Q: Do I really need to re-run all the steps like segmentation every time I want to try a change to the method?*

A: In the .json config you can set the "use cache" flags to true, which will use cached intermediate outputs of the method, for example segmentation masks. This of course makes the method run faster on the same input, which allows you to quickly test tweaks e.g. to the prompt.

*Q: I am getting errors about missing intermediate outputs from the method, e.g. segmentation masks.*

A: This is likely caused by missing or corrupted cache. In the .json config file, set the "use cache" flags to false, in order to regenerate these intermediate outputs from scratch.

*Q: The object which the method selects to be moved does not match my language command.*

A: You may need to regenerate the cache. For example, if your cache is for the task of moving an apple and you want to move the orange instead, then you can no longer use the cached NeRF visual models etc. You will need to set the "use cache" flags to false in the .json config file. However, you can still use some of the cache, e.g. segmentations.