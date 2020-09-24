<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

# CNN-Explainer

This repository implements the following techniques for interpreting convolutional neural networks:

1. Saliency maps <sup>[1]</sup>
2. Guided Backpropagation <sup>[2]</sup>
3. Class visualization <sup>[3]</sup>
4. Grad-CAM <sup>[4]</sup>

Apart from this, the following techniques are also implemented

1. Adversarial fooling (by backpropagating classification error of required fooling class into the image) <sup>[5]</sup>

## References

1. Simonyan, K. et al. “Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.” CoRR abs/1312.6034 (2014): n. pag.

2. Springenberg, Jost Tobias et al. “Striving for Simplicity: The All Convolutional Net.” CoRR abs/1412.6806 (2015): n. pag.

3. Yosinski, J. et al. “Understanding Neural Networks Through Deep Visualization.” ArXiv abs/1506.06579 (2015): n. pag.

4. Selvaraju, R. R. et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.” International Journal of Computer Vision 128 (2019): 336-359.

5. Szegedy, Christian et al. “Intriguing properties of neural networks.” CoRR abs/1312.6199 (2014): n. pag.
