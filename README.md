# ErfNet
### Introduction

This repository contains a Keras(TF backend) implementation of [ErfNet](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf). The original implementations can be found here [PyTorch](https://github.com/Eromera/erfnet_pytorch), [Torch](https://github.com/Eromera/erfnet). At the time I am writing this, the owners have restricted the acces to the code.

If you find this work useful for your research, please consider citing:

 - **"Efficient ConvNet for Real-time Semantic Segmentation"**, E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, IEEE Intelligent Vehicles Symposium (IV), pp. 1789-1794, Redondo Beach (California, USA), June 2017. 
**[Best Student Paper Award]**, [[pdf]](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf)

 - **"ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation"**, E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, Transactions on Intelligent Transportation Systems (T-ITS), December 2017. [[pdf]](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)

### Requirements
```
keras==2.*
numpy>=1.10
```

### Architecture
ERFNet is designed for running in real-time while providing an accurate semantic segmentation. The  buiding block of the architecture is a novel layer that uses residual connections and factorized convolutions in order to remain efficient while retaining remarkable accuracy. 

<p align="center">
  <img width="350" height="250" src="https://github.com/rdbch/ErfNet/blob/master/doc/images/model_summary.png">
  <img width="350" height="250" src="https://github.com/rdbch/ErfNet/blob/master/doc/images/building_blocks.png">

</p>

### Repository roadmap
The project road-map:

- [x] Initial model 
- [ ] Training script for Cityscapes (single GPU)
- [ ] Provide eval tool for Cityscapes
