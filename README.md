# Computer Vision Models

## Backbones

- [x] [`AlexNet`](cvm/models/alexnet.py) - [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), NeurIPS, 2012
- [x] [`VGGNets`](cvm/models/vggnet.py) - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556), 2014
- [x] [`GoogLeNet`](cvm/models/googlenet.py) - [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842), 2014
- [x] [`Inception-V3`](cvm/models/inception_v3.py) - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567), 2015
- [x] [`Inception-V4 and Inception-ResNet`](cvm/models/inception_v4.py) - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261), AAAI, 2016
- [x] [`ResNet`](cvm/models/resnet.py) - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015
- [x] [`SqueezeNet`](cvm/models/squeezenet.py) - [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360), 2016
- [x] [`ResNeXt`](cvm/models/resnet.py) - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431), CVPR, 2016
- [ ] `Res2Net` - [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169), TPAMI, 2019
- [x] [`ReXNet`](cvm/models/rexnet.py) - [Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992), CVPR, 2020
- [x] [`Xception`](cvm/models/xception.py) - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357), CVPR, 2016
- [x] [`DenseNet`](cvm/models/densenet.py) - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993), CVPR, 2016
- [ ] `DLA` - [Deep Layer Aggregation](https://arxiv.org/abs/1707.06484), CVPR, 2017
- [ ] `DPN` - [Dual Path Networks](https://arxiv.org/abs/1707.01629), NeurIPS, 2017
- [ ] `NASNet-A` - [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012), CVPR, 2017
- [ ] `PNasNet` - [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559), ECCV, 2017
- [x] [`MobileNets`](cvm/models/mobilenet.py) - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861), 2017
- [x] [`MobileNetV2`](cvm/models/mobilenetv2.py) - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381), CVPR, 2018
- [x] [`MobileNetV3`](cvm/models/mobilenetv3.py) - [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244), ICCV, 2019
- [x] [`ShuffleNet`](cvm/models/shufflenet.py) - [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083), CVPR, 2017
- [x] [`ShuffleNetV2`](cvm/models/shufflenetv2.py) - [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164), ECCV, 2018
- [x] [`MnasNet`](cvm/models/mnasnet.py) - [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626), CVPR, 2018
- [x] [`GhostNet`](cvm/models/ghostnet.py) - [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907), CVPR, 2019
- [ ] `HRNet` - [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919), TPAMI, 2019
- [ ] `CSPNet` - [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929), CVPR, 2019
- [x] [`EfficientNet`](cvm/models/efficientnet.py) - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946), ICML, 2019
- [x] [`EfficientNetV2`](cvm/models/efficientnetv2.py) - [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298), ICML, 2021
- [x] [`RegNet`](cvm/models/regnet.py) - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678), CVPR, 2020
- [ ] `GPU-EfficientNets` - [Neural Architecture Design for GPU-Efficient Networks](https://arxiv.org/abs/2006.14090), 2020
- [ ] `LambdaNetworks` - [LambdaNetworks: Modeling Long-Range Interactions Without Attention](https://arxiv.org/abs/2102.08602), ICLR, 2021
- [ ] `RepVGG` - [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697), CVPR, 2021
- [ ] `HardCoRe-NAS` - [HardCoRe-NAS: Hard Constrained diffeRentiable Neural Architecture Search](https://arxiv.org/abs/2102.11646), ICML, 2021
- [ ] `NFNet` - [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171), ICML, 2021
- [ ] `NF-ResNets` - [Characterizing signal propagation to close the performance gap in unnormalized ResNets](https://arxiv.org/abs/2101.08692), ICLR, 2021
- [x] [`ConvMixer`](cvm/models/convmixer.py) - [Patches are all you need?](https://openreview.net/forum?id=TVHS5Y4dNvM), 2021
- [x] [`VGNets`](cvm/models/vgnet.py) - [Efficient CNN Architecture Design Guided by Visualization](https://arxiv.org/abs/2207.10318), ICME, 2022
- [x] [`ConvNeXt`](cvm/models/convnext.py) - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), CVPR, 2022

### Attention Blocks

- [x] [`Non-Local`](cvm/models/ops/blocks/non_local.py) - [Non-local Neural Networks](https://arxiv.org/abs/1711.07971), CVPR, 2017
- [x] [`Squeeze-and-Excitation`](cvm/models/ops/blocks/squeeze_excite.py) - [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507), CVPR, 2017
- [x] [`Gather-Excite`](cvm/models/ops/blocks/gather_excite.py) - [Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks](https://arxiv.org/abs/1810.12348), NeurIPS, 2018
- [x] [`CBAM`](cvm/models/ops/blocks/cbam.py) - [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521), ECCV, 2018
- [x] [`SelectiveKernel`](cvm/models/ops/blocks/selective_kernel.py) - [Selective Kernel Networks](https://arxiv.org/abs/1903.06586), CVPR, 2019
- [x] [`ECA`](cvm/models/ops/blocks/efficient_channel_attention.py) - [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151), CVPR, 2019
- [x] [`GlobalContext`](cvm/models/ops/blocks/global_context.py) - [GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond](https://arxiv.org/abs/1904.11492), 2019
- [ ] `ResNeSt` - [ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955), 2020
- [ ] `HaloNets` - [Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/abs/2103.12731), 2021

### Transformer

- [x] [`ViT`](cvm/models/vision_transformer.py) - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), ICLR, 2020
- [ ] `DeiT` - [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877), ICML, 2020
- [ ] `Swin Transformer` - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030), ICCV, 2021
- [ ] `Twins` - [Twins: Revisiting the Design of Spatial Attention in Vision Transformers](https://arxiv.org/abs/2104.13840), NeurIPS, 2021

### MLP

- [x] [`MLP-Mixer`](cvm/models/mlp_mixer.py) - [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601), NeurIPS, 2021
- [x] [`ResMLP`](cvm/models/resmlp.py) - [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404), 2021
- [ ] `gMLP` - [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050), 2021

### Self-supervised

- [ ] `MAE` - [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR, 2021

## Object Detection

- [ ] `R-CNN` - [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524), CVPR, 2013
- [ ] `Fast R-CNN` - [Fast R-CNN](https://arxiv.org/abs/1504.08083), ICCV, 2015
- [ ] `Faster R-CNN` - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497), 2015
- [x] `YOLOv1` - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640), 2015
- [ ] `SSD` - [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325), ECCV, 2015
- [ ] `FPN` - [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144), 2016

## Semantic Segmentation

- [x] [`FCN`](cvm/models/seg/fcn.py) - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038), CVPR, 2014
- [x] [`UNet`](cvm/models/seg/unet.py) - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), MICCAI, 2015
- [ ] `PSPNet` - [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105), CVPR, 2016
- [x] [`DeepLabv3`](cvm/models/seg/deeplabv3.py) - [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf), 2017
- [x] [`DeepLabv3+`](cvm/models/seg/deeplabv3_plus.py) - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf), CVPR, 2018
- [ ] `Mask R-CNN` - [Mask R-CNN](https://arxiv.org/abs/1703.06870), 2017

## Generative Models

### GANs

- [x] `GAN` - [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), 2014
- [x] [`DCGAN`](cvm/models/gan/dcgan.py) - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), ICLR, 2016
- [ ] `WGAN` - [Wasserstein GAN](https://arxiv.org/abs/1701.07875), 2017

### VAEs

- [x] [`VAE`](cvm/models/vae/vae.py) - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), 2013
- [ ] `β-VAE` - [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl), ICLR, 2017


### Diffusion Models


### Flow-based


## Adversarial Attacks

 - [x] [`FGSM`](cvm/attacks/fgsm.py) - [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), ICLR, 2014
 - [x] [`PGD`](cvm/attacks/pgd.py) - [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083), ICLR, 2017