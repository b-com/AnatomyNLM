# AnatomyNLM (Anatomy of Neural Language Models)
This repository contains the source codes linked to our tutorial [Anatomy of Neural Language Models](https://arxiv.org/abs/2401.03797).
The objective of the tutorial is to provide a self-contained comprehensive anatomy of neural language models in a strict mathematical framework accompanied with clear graphical illustrations. The framework covers Feedforward Neural Network (FFNN) LMs, Recurrent Neural Network (RNN) LMs and transformer LMs.
In order to validate our mathematical framework, we derived the formulas that give the total number of trainable parameters of the considered LMs as functions of their hyper parameters. These formulas have been implemented in Python and their results were compared to trainable parameters' counters of Tensorflow with examples on widely used models like BERT and GPT2. In addition, we provided a simple from-scratch implementation of transformer inference-pass equations with an example implementing GPT2. Our implementation was compared to the one of KerasNLP. The aforementioned comparisons, showing identical results, confirm that the anatomy of LMs explored in this tutorial is accurate and complete.

If you use this repository, please cite the following paper:
```
@article{Saleh2024,
  title={Anatomy of Neural Language Models},
  author={Majd Saleh and Stephane Paquelet},
  journal={arXiv preprint arXiv:2401.03797},
  year={2024}
}
```

## Setup instructions

1.	Clone the repository to your local machine.
2.	Create a virtual environment: python 3.10
3.	Activate your virtual environment
4.	Install the requirements: ``pip install -r requirements.txt``