# MonoCLUE: Object-Aware Clustering Enhances Monocular 3D Object Detection



## Introduction



This repository provides the official implementation of "MonoCLUE: Object-Aware Clustering Enhances Monocular 3D Object Detection" based on the excellent work [MonoDGP](https://github.com/PuFanqi23/MonoDGP). In this work, we propose a DETR-based monocular 3D detection framework that strengthens visual reasoning by leveraging clustering and scene memory, enabling robust performance under occlusion and limited visibility.

本代码库基于优秀的 MonoDGP 工作，提供了《MonoCLUE：基于目标感知聚类提升单目三维目标检测》的官方实现。在本研究中，我们提出了一种基于 DETR 的单目三维检测框架，该框架通过聚类与场景记忆机制增强视觉推理能力，使其在目标遮挡及可见度受限场景下仍能保持稳定性能。

[![img](https://camo.githubusercontent.com/eddcbe3c74da499f8561e676c5a0dc4ee48ad527443db604688224292d27f325/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4d6f6e6f434c55452d61725869762d7265642e737667)](https://arxiv.org/abs/2511.07862) [![img](https://camo.githubusercontent.com/241860a41f223eca1fc85289aae73fb4ae69cdf338d4af2325535e51f4e8bd36/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4b495454492d646174617365742d677265656e2e737667)](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

[![img](https://github.com/SungHunYang/MonoCLUE/raw/main/figures/overall_architecture.png)](https://github.com/SungHunYang/MonoCLUE/blob/main/figures/overall_architecture.png)

[![img](https://github.com/SungHunYang/MonoCLUE/raw/main/figures/explanation.png)](https://github.com/SungHunYang/MonoCLUE/blob/main/figures/explanation.png)

## Demo



### Scene 1



[![img](https://github.com/SungHunYang/MonoCLUE/raw/main/figures/sample1.gif)](https://github.com/SungHunYang/MonoCLUE/blob/main/figures/sample1.gif)

### Scene 2



[![img](https://github.com/SungHunYang/MonoCLUE/raw/main/figures/sample2.gif)](https://github.com/SungHunYang/MonoCLUE/blob/main/figures/sample2.gif)

## Main Result



Note that the randomness of training for monocular detection would cause a variance of ±1 AP3D|R40 on KITTI.

The official results :

| Models   | Val, AP3D\|R40 | Logs     | Ckpts                                                        |                                                              |                                                              |
| -------- | -------------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Easy     | Mod.           | Hard     |                                                              |                                                              |                                                              |
| MonoCLUE | 33.7426%       | 24.1090% | 20.5883%                                                     | [log](https://drive.google.com/file/d/1G51qlnwXMSD5zgRlXPCjkAjBWMlqbjdg/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/183zRv7EaR3ReS4QA9KTfLPbRRxwjHcqU/view?usp=sharing) |
| 31.5802% | 23.5648%       | 20.2746% | [log](https://drive.google.com/file/d/1bTqH3DCnN-PSjchHDX0yG52xsAdMCFPy/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1Sa1J2m0dp0a34ILvWt7YMFl1jJHprDbC/view?usp=drive_link) |                                                              |

The test result :

[![img](https://github.com/SungHunYang/MonoCLUE/raw/main/figures/test_result.png)](https://github.com/SungHunYang/MonoCLUE/blob/main/figures/test_result.png)

## Installation



1. Clone this project and create a conda environment:

	```
	git clone https://github.com/SungHunYang/MonoCLUE.git
	cd MonoCLUE
	
	conda create -n monoclue python=3.8
	conda activate monoclue
	```

	

2. Install pytorch and torchvision matching your CUDA version:

	```
	pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
	```

	

3. Install requirements and compile the deformable attention:

	```
	pip install -r requirements.txt
	
	cd lib/models/monoclue/ops/
	bash make.sh
	
	cd ../../../..
	```

	

4. Download KITTI datasets and prepare the directory structure as:

  ```
  │MonoCLUE/
  ├──...
  │data/kitti/
  ├──ImageSets/
  ├──training/
  │   ├──image_2
  │   ├──label_2
  │   ├──calib
  ├──testing/
  │   ├──image_2
  │   ├──calib
  ```

  

  Note that if you need the Waymo dataset, please follow [DEVIANT](https://github.com/abhi1kumar/DEVIANT)

  请注意，若你需要使用 Waymo 数据集，请遵循 DEVIANT 相关规范

5. Download sam_vit_h.pth from the [SAM](https://github.com/facebookresearch/segment-anything) repository and prepare the SAM-guided dataset.

  ```
  python make_sam.py
  ```

  

  **Important:** Before generating the SAM-guided dataset, please set `self.data_augmentation = False` in `kitti_dataset.py`, and set `batch_size = 1` in `monoclue.yaml`.
  Otherwise, the generated SAM labels may be misaligned.

  Note that if you run it with all_category, a folder named "label_sam_all" should be created.

  重要提示：在生成由 SAM 引导的数据集之前，请在 kitti_dataset.py 文件中设置 self.data_augmentation = False，并在 monoclue.yaml 文件中设置 batch_size = 1。

  否则，生成的 SAM 标注可能会出现错位。请注意，若使用 all_category 参数运行，需创建一个名为 “label_sam_all” 的文件夹。

6. Finally, prepare the directory structure as:

	```
	│MonoCLUE/
	├──...
	│data/kitti/
	├──ImageSets/
	├──training/
	│   ├──image_2
	│   ├──label_2
	│   ├──calib
	|   ├──label_sam/
	|        ├──region
	|        └──depth
	├──testing/
	│   ├──image_2
	│   ├──calib
	```

	

	You can also change the data path at "dataset/root_dir" in `configs/monoclue.yaml`.
	
	你也可以在 configs/monoclue.yaml 文件的 "dataset/root_dir" 处修改数据路径。

## Get Started



### Train



You can modify the settings of models and training in `configs/monoclue.yaml` and indicate the GPU in `train.sh`:

你可以在 configs/monoclue.yaml 文件中修改模型与训练相关的设置，并在 train.sh 文件中指定所用 GPU：

```
bash train.sh configs/monoclue.yaml > logs/monoclue.log
```



### Test



The best checkpoint will be evaluated as default. You must ensure that the checkpoint is located in `outputs/monoclue/`:

系统将默认评估最优检查点。您必须确保该检查点位于 outputs/monoclue/ 路径下

```
bash test.sh configs/monoclue.yaml
```



### Visualize



1. After testing, prepare the directory structure as follows:测试完成后，请按如下方式准备目录结构：

	```
	│MonoCLUE/
	├──outputs/
	│   ├──monoclue/
	│       ├──outputs/
	│           ├──data/
	│               ├──000001.txt
	│               ├──000002.txt
	│               ├──000003.txt
	│               ├──000004.txt
	```

	

2. Navigate to the visualization folder:导航至可视化文件夹：

	```
	cd visualize
	```

	

3. Run the visualization script:运行可视化脚本：

  ```
  python draw3D_bbox.py
  
  # With detailed information
  python draw3D_bbox.py --print_info True
  ```

  

  Note that if you need LiDAR visualization, please follow [kitti_object_vis](https://github.com/kuixu/kitti_object_vis) repository

  请注意，若您需要进行激光雷达可视化操作，请遵循 kitti_object_vis 代码库的相关指引

## Citation



Please cite this work if you find it useful:

```
@inproceedings{yang2026monoclue,
  title={MonoCLUE: Object-Aware Clustering Enhances Monocular 3D Object Detection},
  author={Yang, Sunghun and Lee, Minhyeok and Lee, Jungho and Lee, Sangyoun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={14},
  pages={11721--11729},
  year={2026}
}
```



## Acknowlegment



This repo benefits from the excellent [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR) / [MonoDGP](https://github.com/PuFanqi23/MonoDGPand).