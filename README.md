<div align="center">
<h1>
  Facilitating NSFW Text Detection in Open-Domain Dialogue Systems via Knowledge Distillation
</h1>
</div>

<p align="center">
📄 <a href="https://arxiv.org/pdf/2309.09749v2.pdf" target="_blank">Paper</a> • 
🤗 <a href="" target="_blank">Dataset</a> • 
🛠️ <a href="" target="_blank">Model</a>
</p>

## Overview

_CensorChat_ is a dialogue monitoring dataset aimed at NSFW dialogue detection. Leveraging knowledge distillation techniques involving GPT-4 and ChatGPT, this dataset offers a cost-effective means of constructing NSFW content detectors. The process entails collecting real-life human-machine interaction data and breaking it down into single utterances and single-turn dialogues, with the chatbot delivering the final utterance. ChatGPT is employed to annotate unlabeled data, serving as a training set. Rationale validation and test sets are constructed using ChatGPT and GPT-4 as annotators, with a self-criticism strategy for resolving discrepancies in labeling. A BERT model is fine-tuned as a text classifier on pseudo-labeled data, and its performance is assessed.

<p align="center"> <img src="assets/proposed_methodology.png" style="width: 70%;" id="title-icon"></p>

## Data Collection

- NSFW text in dialogues refers to text-based communication that contains **sexually explicit language, violence, profanity, hate speech, or suggestive content** that is not suitable for beneficial and healthy dialogue platforms.

- We collect data from a popular social media platform for personal dialogue that allows people to engage in deep discussions about life, aspirations, and philosophy with renowned virtual figures.

- we propose extracting the dialogue into two data formats: utterance-level and context-level content. For utterance-level content, we split the dialogue into utterances, consisting of $\{u_i\}_1^n$, based on the speaker's perspective. For context-level content, we divide the dialogue into single-turn sessions, consisting of $\{u_i^\mathrm{U}, u_i^\mathrm{C}\}_1^n$, where users initiate the conversation and bots respond. $u$ denotes the utterance. $\mathrm{U}$ and $\mathrm{C}$ denote the user and chatbot, respectively.

## Algorithm

Text classification with BERT model via knowledge distillation is shown below:

<p align="center"> <img src="assets/algorithm.png" style="width: 70%;" id="title-icon"></p>

## Data Annotation

- NSFW: whether a response is NSFW or not (a binary label).

### Cohen's Kappa

Cohen's kappa for valid and test set is shown below:

<p align="center"> <img src="assets/kappa.png" style="width: 100%;" id="title-icon"></p>

### Data Statistics

Data statistics is shown below:

<p align="center"> <img src="assets/data_statistics.png" style="width: 85%;" id="title-icon"></p>

### Examples

We present some examples in our dataset as follows:

<p align="center"> <img src="assets/examples.png" style="width: 100%;" id="title-icon"></p>

## Model Performance

<p align="center"> <img src="assets/results.png" style="width: 80%;" id="title-icon"></p>

## Usage

**NOTICE:** You can directly use our trained checkpoint on the hub of Hugging Face.

## Citation

If our work is useful for your own, you can cite us with the following BibTex entry:

```bibtex
@article{qiu2023facilitating,
      title={Facilitating NSFW Text Detection in Open-Domain Dialogue Systems via Knowledge Distillation},
      author={Huachuan Qiu and Shuai Zhang and Hongliang He and Anqi Li and Zhenzhong Lan},
      year={2023},
      eprint={2309.09749},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2309.09749}
}
```
