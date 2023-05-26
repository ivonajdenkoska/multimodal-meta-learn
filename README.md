# Meta Learning to Bridge Vision and Language Models for Multimodal Few-Shot Learning
This is the official code repository for "Meta Learning to Bridge Vision and Language Models for Multimodal Few-Shot Learning", published at ICLR 2023.

[[arXiv]](https://arxiv.org/pdf/2302.14794.pdf) [[OpenReview]](https://openreview.net/forum?id=3oWo92cQyxL)

### Intro
Multimodal few-shot learning is challenging due to the large domain gap between vision and language modalities. 
Existing methods are trying to communicate visual concepts as prompts to frozen language models, but rely on hand-engineered task induction to reduce the hypothesis space.
To address these limitations and enable a learnable process, we propose a **multimodal meta-learning approach**.

![overview](https://github.com/ivonajdenkoska/multimodal-meta-learn/blob/main/meta_learning_overview.png)

### Approach Overview
Our approach breaks down the model training into observing a collection of multimodal few-shot tasks. 
We introduce a meta-mapper network, which serves as a meta-learner, effectively bridging the gap between frozen large-scale vision and language models and leveraging 
their pre-existing learned capacity. By updating only the learnable parameters of the meta-mapper, it learns to accumulate shared meta-knowledge across these tasks.

![model](https://github.com/ivonajdenkoska/multimodal-meta-learn/blob/main/model.png)

### Getting Started
First clone the project, create the environment and install dependencies: 

```
git clone https://github.com/ivonajdenkoska/multimodal-meta-learn.git
conda env create -f environment.yml
conda activate multimodal_meta_learn
```

Download the multimodal few-shot datasets from [here](https://fh295.github.io/frozen.html) and place them in your data folder 
which will be assigned to `--data_path`. Also, download the COCO image captioning dataset from [here](https://cocodataset.org/#download).

### Usage

To perform meta-training with COCO captioning dataset, first run `parse_coco.py` to obtain the 
preprocessed COCO pickle file.
To perform the training of the full model, run `python main.py`. You can choose the episodic method to perform 
the meta-training or the non_episodic one to perform standard mini-batched training from this script.
To perform inference with trained models, run `python main_inference.py`.

### Reference
If you find this code or the paper useful for your work, please cite:
```
@inproceedings{
    najdenkoska2023meta,
    title={Meta Learning to Bridge Vision and Language Models for Multimodal Few-Shot Learning},
    author={Ivona Najdenkoska and Xiantong Zhen and Marcel Worring},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=3oWo92cQyxL}
    }
}
```

### Acknowledgments
This repository uses [HuggingFace](https://huggingface.co/) and is based on [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) 
and [MAML](https://github.com/cbfinn/maml) code repositories. 