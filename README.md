# Spatial Attention Pyramid Network for Unsupervised Domain Adaptation

## Introduction

This is forked SAPNet implementation that have been modified to match our research dataset and some function modification. All rights reserved to the original owner.

This is the forked version of this paper [implementation](https://isrc.iscas.ac.cn/gitlab/research/domain-adaption).

Li, Congcong, et al. "Spatial attention pyramid network for unsupervised domain adaptation." European Conference on Computer Vision. Springer, Cham, 2020.

## Description

Unsupervised domain adaptation is critical in various computer vision tasks, such as object detection, instance segmentation, and semantic segmentation, which aims to alleviate performance degradation caused by domain-shift. Most of previous methods rely on a single-mode distribution of source and target domains to align them with adversarial learning, leading to inferior results in various scenarios. To that end, in this paper, we design a new spatial attention pyramid network for unsupervised domain adaptation. Specifically, we first build the spatial pyramid representation to capture context information of objects at different scales. Guided by the task-specific information, we combine the dense global structure representation and local texture patterns at each spatial location effectively using the spatial attention mechanism. In this way, the network is enforced to focus on the discriminative regions with context information for domain adaption. We conduct extensive experiments on various challenging datasets for unsupervised domain adaptation on object detection, instance segmentation, and semantic segmentation, which demonstrates that our method performs favorably against the state-of-the-art methods by a large margin.

![SAPNET](SAPNET.png)

