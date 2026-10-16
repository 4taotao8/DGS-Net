# DGS-Net: Learning Dexterous Grasps from Single-View Point Clouds via a Multi-Object Scene Dataset

Official code, pretrained weights, and training-data preparation for **DGS-Net**. The network predicts graspable regions, 6D hand rotations, offsets from observed points to the hand TCP, and 20-joint pre-grasp and final-grasp configurations from single-view point clouds of cluttered scenes. The paper also introduces a two-stage dataset pipeline: dense object-level grasps are synthesized first, then validated in multi-object scenes.

- **Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11719723) · [arXiv preprint](https://arxiv.org/abs/2603.15410)
- **Project:** [GitHub](https://github.com/4taotao8/DGS-Net)
- **Project website:** [4taotao8.github.io/DGS-Net](https://4taotao8.github.io/DGS-Net/) · [website source](docs/index.html)
- **Dataset:** [Hugging Face](https://huggingface.co/datasets/4taotao8/DGS-Net/tree/main) · [Baidu Netdisk](https://pan.baidu.com/s/1SL4EADnPFlTuN-PTyRcVHg?pwd=7816) (access code **7816**)
- **Authors:** Tao Geng, Dapeng Yang, Ziwei Liu, Le Zhang, Le Qi, WangYang Li, Yi Ren, Shan Luo, and Fenglei Ni

The released dataset covers **307 objects, 240 multi-object scenes, 256 views per scene, and over 350,000 validated grasps**. See the [paper](https://arxiv.org/html/2603.15410v1) and the [original project overview](https://github.com/4taotao8/DGS-Net) for the data generation method and results.

![DGS-Net example](images/image1.png)

## 1. Environment setup

The commands below target **Linux, Python 3.10, CUDA 11.7, and an NVIDIA GPU**. Install [Git LFS](https://git-lfs.com/) before cloning. The authors' `gt_grasp` environment was checked with PyTorch 2.0.1, `spconv-cu117` 2.3.6, `torch-scatter` 2.0.9, and `flash-attn` 2.7.2.post1. CUDA extensions must match the installed PyTorch and CUDA build. For another CUDA version, follow the [PyTorch guide](https://pytorch.org/get-started/previous-versions/), [PyG wheel index](https://data.pyg.org/whl/), and [FlashAttention instructions](https://github.com/Dao-AILab/flash-attention#installation).

```bash
git clone https://github.com/4taotao8/DGS-Net.git
cd DGS-Net
git lfs install
git lfs pull

conda create -n gt_grasp python=3.10 -y
conda activate gt_grasp
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
python -m pip install spconv-cu117==2.3.6
python -m pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
python -m pip install flash-attn==2.7.2.post1 --no-build-isolation
```

The checkpoint is about 206 MB and is tracked with Git LFS. If `torch.load` cannot read it, run `git lfs pull` and check that it is a real checkpoint rather than a short LFS pointer. Building `flash-attn` from source may require a CUDA toolkit and C++ compiler. Inference and training require CUDA; annotation generation uses Open3D and can run without a GPU.

Check the environment and checkpoint:

```bash
python -c "import torch, spconv, torch_scatter, flash_attn, open3d; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch; from dgs_net.model import PointTransformerV3; m=PointTransformerV3(); m.load_state_dict(torch.load('checkpoints/dgs_net_best.pth', map_location='cpu', weights_only=True)); print('checkpoint OK')"
```

## 2. Pretrained inference and visualization

Run from the repository root. Without arguments, the script uses the included RGB-D pair, checkpoint, and example camera intrinsics. It saves `outputs/grasps.npz` and opens an Open3D window showing four randomly selected predicted grasps:

```bash
python infer_rgbd.py
```

For another camera, provide **aligned** RGB and single-channel depth PNGs. Depth is measured in **millimeters**; `fx`, `fy`, `cx`, and `cy` are your calibrated intrinsics in pixels. Output positions are in meters in the camera coordinate system.

```bash
python infer_rgbd.py \
  --rgb examples/rgb.png --depth examples/depth.png \
  --checkpoint checkpoints/dgs_net_best.pth \
  --fx 605.0880126953125 --fy 604.061279296875 \
  --cx 323.97198486328125 --cy 240.3543701171875 \
  --output outputs/grasps.npz
```

The output `.npz` contains arrays `score`, `position` (`N×3`), `rotation` (`N×3×3`), `pre_joints` and `final_joints` (`N×20`, radians), and `offset` (`N×3`). Scores are graspability probabilities. The decoder uses the original confidence threshold of 0.6 and suppresses grasps within 2 mm and 5° using the original joint distance/rotation criterion. If there are no candidates, the `.npz` is empty. This script predicts grasps; robot execution still needs your camera-to-robot transform and robot safety checks.

After saving, the command randomly selects **four** post-NMS grasps and opens one Open3D view. Opaque blue hands show final grasps; translucent light-blue hands show pre-grasps 10 cm back along each hand's local Z axis. Purple spheres mark predicted TCPs and red spheres mark observed points before the learned offset. Use `--visualize-count 8` to show eight, `--visualize-seed 42` for repeatable selection, or `--visualize-count 0` to save without opening a window. The Open3D window needs a graphical desktop.

## 3. Dataset

Get the dataset from the [Hugging Face dataset page](https://huggingface.co/datasets/4taotao8/DGS-Net/tree/main) or the [Baidu Netdisk mirror](https://pan.baidu.com/s/1SL4EADnPFlTuN-PTyRcVHg?pwd=7816) (access code **7816**). Place the downloaded or extracted `scenes/` and `grasp_labels/` directories under the repository's `data/` directory as shown below. The dataset contains object-level grasp records (`grasp_labels/`) and scene-level captures (`scenes/`). The scene data includes RGB-D images, segmentation, camera poses, object visibility, and successful hand poses with pre-grasp and final joint angles. The paper describes more than 60,000 RGB-D views from 240 scenes, each containing 5–15 objects. Dataset files and generated annotations are not stored in this Git repository.

For the converter, place the raw scene records under `data/` at the repository root:

```
data/
├── scenes/                         # Stage II Grasp Data
│   ├── scene_0000/
│   │   ├── rgb/                    # RGB images (256 viewpoints)
│   │   ├── depth/                  # Depth images
│   │   ├── pointcloud/             # Scene point clouds
│   │   ├── seg/                    # Segmentation masks
│   │   ├── grasp/                  # Grasp annotations
│   │   ├── cam_extrinsic.txt       # Camera extrinsic parameters for each viewpoint
│   │   ├── cam_info.txt            # Camera intrinsic parameters
│   │   ├── grasp_lables.txt        # Successful grasp poses and joint angles
│   │   ├── obj_info.txt            # Object information and metadata in the scene
│   │   └── visible_mask.txt        # Object visibility information in RGB images for each viewpoint
│   ├── scene_0001/
│   └── ...
└── grasp_labels/                   # Stage I Grasp Data
    ├── bowl/                       # Dexterous hand grasp poses for bowl objects
    │   ├── 024_bowl/               # Grasp data for 5 different scaling factors
    │   └── ...
    ├── cuboid/                     # Dexterous hand grasp poses for cuboid objects
    ├── cylinder/                   # Dexterous hand grasp poses for cylinder objects
    ├── stick/                      # Dexterous hand grasp poses for stick objects
    └── sphere/                     # Dexterous hand grasp poses for sphere objects
```

`grasp_lables.txt` is the spelling in the released files and expected by the converter. It reads `depth/`, `seg/`, `cam_extrinsic.txt`, `obj_info.txt`, `visible_mask.txt`, and `grasp_lables.txt`; `rgb/` helps inspection but is not required for annotation generation. If `scenes/scene_XXXX/pointcloud/` and `grasp/` already contain the required point-level `.npy` files, you can train directly. `data/` is excluded by `.gitignore`.

### Generate point-level annotations

Check that the synthetic camera intrinsics in `config.yaml` match your scene data. First process a single frame:

```bash
python generate_training_data.py --data-root data --scene 0 --frame-start 0 --frame-end 1
```

Add `--seed 6282` to make each generated frame repeatable, including when scenes are processed in a different order. The original annotation generator did not record its random sampling state. Regenerating from the same RGB-D files can therefore select different points and produce a different number of grasps than the archived `scenes_tase` annotations. To use the exact archived training labels, use their matching `pointcloud/` and `grasp/` files together.

Then process all 256 frames for every scene:

```bash
for scene in $(seq 0 239); do
  python generate_training_data.py --data-root data --scene "$scene"
done
```

The converter writes `data/scenes/scene_XXXX/pointcloud/FFFF.npy` and `data/scenes/scene_XXXX/grasp/FFFF.npy`, **replacing existing `.npy` files for each processed frame**. The raw RGB-D images and scene text files remain in the same scene directory. Each point cloud has 20,000 points and fields `points`, `grasp_points_mask` (`-1`: negative, `0`: candidate, `1`: reference), `cls_mask`, and `seg`. The grasp files hold camera-frame poses, target object IDs, grasp type, point indices, and pre/final joints. Labels use 4 mm voxel downsampling, visibility filtering, and the TASE rule of 3–20 nearby candidate points based on the nearest point's segmentation class. The dataset loader calculates the 3D point-to-TCP offset target from these files.

The converter starts from **stage-II scene records**. To create those records from object models instead of downloading the full dataset, follow Section 5 below.

## 4. Train on the dataset

Training reads the `pointcloud/` and `grasp/` annotations inside `data/scenes/`, whether they came with the download or were generated above. `config.yaml` assigns scenes `0–199` to training and `200–239` to validation, with 256 views per scene. NumPy object arrays in this trusted dataset are loaded with `allow_pickle=True` by the dataset loader. Run from the repository root:

```bash
python train.py --data-root data --output-dir outputs
```

To initialize a new run from the released pretrained weights:

```bash
python train.py --data-root data --output-dir outputs --resume checkpoints/dgs_net_best.pth
```

If your dataset is stored elsewhere, pass the directory that contains `scenes/` to `--data-root`, for example `python train.py --data-root /path/to/dataset --output-dir outputs`.

`train.py` saves the best validation-loss state dictionary as `outputs/best.pth`. Defaults in `config.yaml` are batch size 10, 40 epochs, 8 data-loader workers, Adam, and a learning rate of `1e-4`. Adjust memory and worker use with `--batch-size` and `--workers`; use `--epochs` for a shorter run. The paper reports `1e-3` as its learning rate, while both inspected source-code configurations use `1e-4`; this repository retains the code value and documents the difference.

The point loss treats reference and candidate points as positive samples, with weights 2, 1, and 0.1 for reference, candidate, and negative points. Rotation uses matrix MSE plus geodesic loss; offset uses relative L1; pose, offset, and joint losses apply only to reference points. The stage-II model receives ground-truth stage-I labels during training and validation, following the original implementation; inference uses predicted stage-I outputs. Labels and losses follow the [published method](https://arxiv.org/html/2603.15410v1). The supplied checkpoint came from an earlier experimental environment, so a fresh run is not expected to produce identical weights.

## 5. Simulation dataset generation

This optional route creates the dataset from the object models. **You do not need to download the complete RGB-D scene dataset** if you use it. Download only the simulation asset bundle containing `urdfs/` and place it under `data/urdfs/`. The local copy includes the 307 object models used by the paper (cuboid, cylinder, stick, sphere, and bowl), the hand URDF and meshes under `hitdlr3/`, and the scene setup models under `setup/`. An extra `cup/` set from the source project is available but is not part of the default 307-object run. The roughly 308 MB asset tree is excluded from Git by `.gitignore`; it is intended for the separate dataset download link. Check the original rights of third-party meshes before redistributing that bundle.

```text
data/                         # ignored by Git
├── urdfs/
│   ├── cuboid/ cylinder/ stick/ sphere/ bowl/
│   ├── cup/                   # optional source-project objects
│   ├── hitdlr3/               # hand_right_simplify.urdf and meshes
│   └── setup/                 # plane and other scene fixtures
├── grasp_labels/              # written by Stage I
└── scenes/                    # written by Stage II and the point-label converter
```

`requirements.txt` includes PyBullet for this workflow. The simulation uses the same grasp sampling, collision checks, hand model, and camera setup as the original project. Defaults are in `simulation_data/config.yaml`: five scales (0.8–1.2), 2,000 grasp attempts per object and scale, scene IDs 0–239, and 8 × 32 camera views per scene. Simulation runs without the neural-network CUDA extensions, but generating the full dataset is computationally intensive and requires substantial disk space. If you only want to generate data, a separate environment is sufficient:

```bash
conda create -n dgs_sim python=3.10 -y
conda activate dgs_sim
python -m pip install numpy==1.26.4 scipy==1.15.0 PyYAML==6.0.2 \
  open3d==0.18.0 opencv-python==4.10.0.84 pybullet==3.2.7
```

Switch to the CUDA environment from Section 1 before running `train.py`.

**Stage I — object-level grasps.** For a small check, generate one object with a reduced number of attempts:

```bash
python -m simulation_data.stage1_graspdata_generate \
  --object-set cuboid --object-index 0 --attempts 10
```

For the full 307-object set, omit `--object-set`, `--object-index`, and `--attempts`. Each scale is saved to `data/grasp_labels/<category>/<object>/<object>_<scale>.txt`. The default Stage II requires at least 50 successful Stage I grasps for a selected object, so the 10-attempt command above is only a quick check.

```bash
python -m simulation_data.stage1_graspdata_generate --workers 4
```

**Stage II — multi-object scenes.** This stage randomly places objects, validates Stage I grasps against scene collisions, saves `obj_info.txt` and `grasp_lables.txt`, and renders RGB, depth, segmentation, camera poses, and visibility masks. Generate one scene first, then the complete configured range:

```bash
python -m simulation_data.stage2_graspdata_generate --scene 0
python -m simulation_data.stage2_graspdata_generate --workers 4
```

The second command covers scene IDs 0–239 and writes `data/scenes/scene_XXXX/`. An existing scene with the same ID will be overwritten, so use a fresh data directory when regenerating. Stage II uses a per-scene random seed and retries scenes that do not meet the configured minimum number of valid grasps.

**Point labels and training.** After Stage II, run the point-level converter from Section 3 and then train as in Section 4:

```bash
for scene in $(seq 0 239); do
  python generate_training_data.py --data-root data --scene "$scene"
done
python train.py --data-root data --output-dir outputs
```

The first and second stages can also use an alternate data directory set before launch with `DGS_DATA_ROOT=/path/to/dataset`; it must contain `urdfs/`. Pass the same directory to `generate_training_data.py` and `train.py` with `--data-root`.

## 6. Repository contents

| Path | Purpose |
| --- | --- |
| `dgs_net/model.py` | Two-stage Point Transformer V3 grasp network |
| `dgs_net/loss.py` | Graspability, rotation, offset, and joint losses |
| `dgs_net/dataset.py` | Loader for point-level annotations in `scenes/` |
| `generate_training_data.py` | Stage-II scene records to point-level supervision |
| `simulation_data/` | PyBullet Stage I and Stage II dataset generation |
| `train.py` | Training and validation |
| `infer_rgbd.py` | RGB-D inference and Open3D visualization |
| `dgs_net/visualize.py`, `assets/hand/` | Hand mesh rendering |
| `config.yaml` | Camera, data split, hand joints, and training defaults |
| `checkpoints/dgs_net_best.pth` | Pretrained weights (Git LFS) |
| `examples/` | Example RGB and depth images |

## Project website

The [DGS-Net project website](https://4taotao8.github.io/DGS-Net/) is served from [`docs/`](docs/index.html). It presents the method, data-generation pipeline, real-world experiments, and demonstration videos. The selected paper figures are in `docs/fig/`; browser-ready videos and poster images are in `docs/media/`. Figures display at their full aspect ratio and link to their original resolution. The original high-resolution video files under `Media_materials/` remain excluded from Git.

GitHub Pages publishes the `main` branch's **`/docs`** folder when changes are pushed. The page uses relative asset paths, so it also works from a local static server pointed at `docs/`.

## Method notes

DGS-Net has two stages over serialized point-cloud features. Stage I estimates per-point graspability and a 6D rotation. Stage II predicts the point-to-TCP offset and the pre-grasp/final hand joint vectors. The supervision includes graspable neighborhoods rather than only exact grasp centers, and represents both approach and final hand configurations. The point-label converter prepares annotations from stage-II scene records; the separate `simulation_data/` scripts can generate object-level grasps and stage-II scene records from the simulation assets.

The backbone adapts [Point Transformer V3 / Pointcept](https://github.com/Pointcept/Pointcept); please also acknowledge that work when building on this implementation.

## Citation and license

If you use this code or dataset in research, please cite the [published IEEE paper](https://ieeexplore.ieee.org/document/11719723). Use its bibliographic entry from IEEE Xplore. The [arXiv page](https://arxiv.org/abs/2603.15410) has the preprint and author list.

The project code is released under the [MIT License](LICENSE). The separately downloaded dataset and third-party assets retain their respective rights; this code license does not grant additional rights to those materials. The adapted Point Transformer V3 backbone retains the upstream [MIT copyright notice](THIRD_PARTY_NOTICES.md).
