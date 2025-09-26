# DGS-Net: Learning Dexterous Grasps from Single-View Point Clouds via a Multi-Object Scene Dataset

## Abstract

Dexterous grasping in multi-object scene constitutes a fundamental challenge in robotic manipulation. Current mainstream grasping datasets predominantly focus on single-object scenarios and predefined grasp configurations, often neglecting environmental interference and the modeling of dexterous pre-grasp gesture, thereby limiting their generalizability in real-world applications. To address this, we propose DGS-Net, an end-to-end grasp prediction network capable of learning dense grasp configurations from single-view point clouds in multi-object scene. Furthermore, this work proposes a two-stage grasp data generation strategy that progresses from dense single-object grasps to dense scene-level grasps. A high-quality grasping dataset with pre-grasp configurations is constructed, featuring multi-object, single-view point clouds. The dataset comprises 307 objects, 240 scenes, and over 350k validated grasps, offering both diverse grasping configurations and complex environmental conditions. Experimental results demonstrate that DGS-Net achieves grasp success rates of 88.63\% in simulation and 78.98\% on a real robotic platform, surpassing existing methods and validating its efficacy and generalization capability. \added{Our dataset is available at \href{https://github.com/4taotao8/DGS-Net}{https://github.com/4taotao8/DGS-Net}.

## Dataset Overview

Our dataset contains over 60k RGB-D images from 240 multi-object scenes, each with 5-15 randomly selected objects, captured from 256 viewpoints per scene, yielding more than 350k valid grasp instances. Each sample includes the scene point cloud, grasp center point semantic labels, dexterous hand grasp poses, pre-grasp and final joint configurations, occlusion information, and segmentation masks. In contrast to most existing datasets that provide only final grasp configurations, we additionally include pre-grasp hand poses, capturing the hand's initial state during approach. This design better reflects real-world grasping behavior and enhances sim-to-real transfer.

### Dataset Specifications

- **Dataset Size:** [Number of samples - Please Replace This]
- **Data Format:** [Format specifications - Please Replace This]
- **Image Resolution:** [Resolution information - Please Replace This]
- **Categories/Classes:** [Number and types of categories - Please Replace This]
- **Annotation Format:** [Annotation format details - Please Replace This]

## Dataset Download

**Main Dataset:**  https://pan.baidu.com/s/1SL4EADnPFlTuN-PTyRcVHg?pwd=7816 提取码: 7816 

**Additional Files:** [Additional files link if any - Please Replace This]
