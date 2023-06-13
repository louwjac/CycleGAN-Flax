<div align="left">
<h1>
  CycleGAN in <a href="https://jax.readthedocs.io/">Jax</a> + <a href="https://flax.readthedocs.io/">Flax</a>
</h1>
<div>
  <a target="_blank" href="https://colab.research.google.com/github/louwjac/CycleGAN-Flax/blob/main/notebooks/inference.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://huggingface.co/spaces/louwjac/CycleGan-Flax">
    <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces">
  </a>  
</div>
</div>

<img src='video/olsen.gif' height=300 align="right" style="padding:10px;"/>

This implementation is intended to serve as an example of coding an image-to-image translation model in Flax. The codebase makes use of [Gin] for configuration and includes support for TensorBoard logging of both training metrics and images with Google's [Common Loop Utils] library.

[Common Loop Utils]: https://github.com/google/CommonLoopUtils

CycleGAN was published in 2017 by Jun-Yan Zhu, Taesung Park, Phillip Isola and Alexei A. Efros. Since then, the model has been covered thoroughly in online tutorials and blog posts. You will have no trouble finding resources that explain how it works. Please start by looking at the [original project page] and reading the [paper] if you are not already familiar with it.

<br/><br/>

_*Thanks to [Barbara Olsen](https://www.pexels.com/video/a-horse-playing-with-a-salt-block-7881859) for providing the video that I used to make the animation to the right._

[original project page]: https://junyanz.github.io/CycleGAN/
[paper]: https://arxiv.org/pdf/1703.10593.pdf


## Trained Models
### Weights
Weights for models trained on the horse2zebra and monet2photos datasets are published to HuggingFace at the following link: https://huggingface.co/louwjac/CycleGAN-Flax

### Colab Notebook
The inference demo in the "notebooks" folder provides a detailed example of how to instantiate a model and load the trained weights from HuggingFace. It also includes a Gradio demo that makes it easy to submit external images to the model.

<a target="_blank" href="https://colab.research.google.com/github/louwjac/CycleGAN-Flax/blob/main/notebooks/inference.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### TensorBoard logs
The TensorBoard logs for the training runs can be viewed at the links below. The logged images are unfortunately not available due to file size constraints.
1. [horse2zebra](https://tensorboard.dev/experiment/5tJZAfBLQkOJ66BUrBVBkQ/)
2. [monet2photo](https://tensorboard.dev/experiment/GZZKiDoBTl6dsgsqRo7rZw/)

## Training Data
This repo uses the TensorFlow Datasets version of the [CycleGAN datasets]. See [./cyclegan/data.py](./cyclegan/data.py) for the details on the input pipeline.

[CycleGAN datasets]:(https://www.tensorflow.org/datasets/catalog/cycle_gan)

## Running locally

#### Prerequisites
- Linux
- Local Python environment
- Nvidia GPU with the latest drivers
- Git

#### Installation
  ```
    git clone https://github.com/louwjac/CycleGAN-Flax
    cd CycleGAN-Flax
    pip install -r requirements.txt
  ```

#### Train a model
  ```bash

  python main.py config/horse2zebra_original.gin
  ```


## Notes
### Gin
[Gin] is an extremely convenient configuration library that can replace other approaches to configuring parameters such as [ConfigDict] or [Abseil flags]. See the discussion on using Gin with Flax here: [#1226], and keep it in mind when you use Gin for your own projects. Configuration files for several of the CycleGAN datasets are provided in the "config" folder of this repo.

[Gin]: https://github.com/google/gin-config
[ConfigDict]: https://github.com/google/ml_collections
[Abseil flags]: https://abseil.io/docs/cpp/guides/flags
[#1226]: https://github.com/google/flax/discussions/1226


### Common Loop Utils
[Common Loop Utils] (or CLU) is a Python library that provides tools to make it easier to write training loops. For the purposes of this project, that consists of utilities to capture and aggregate training metrics in addition to logging to both files and TensorBoard. CLU is a very young project and documentation is still scarce as of June, 2023. You'll need to read CLU's source code and look at code examples to learn how to use it. Please feel free to let me know if you have a question about the way that I used it.


