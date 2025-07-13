
# Pointmark

Pointmark is a tool for the automated processing of videogrammetry data and subsequent quality assessment of the generated point clouds. The software can be used in both batch mode and for individual videos and calculates various metrics to evaluate the reconstructed sparse point clouds.

# Content

- [Pointmark](#pointmark)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)  
  - [1. Clone the Repository](#1-clone-the-repository)  
  - [2. Set Up Python Environment](#2-set-up-python-environment)  
  - [3. Install External Dependencies](#3-install-external-dependencies)  
- [Usage](#usage)  
  - [CLI](#option-1-cli)  
    - [pointmark.py – CLI (Single Execution)](#pointmarkpy--cli-single-execution)  
    - [pointmark_batch.py – CLI (Batch Execution)](#pointmark_batchpy--cli-batch-execution)  
  - [Python Integration](#option-2-python-integration)  
    - [Required Setup](#required-setup)  
    - [Single Project](#single-project)
    - [Batch Project](#batch-project)
- [How-To's](#how-tos)  
  - [How-To: Prepare the input data](#how-to-prepare-the-input-data) 
  - [How-To: Get sfm data](#how-to-get-sfm-data)   
  - [How-To: Create a Pointmark Batch Setup](#how-to-create-a-pointmark-batch-setup)  
    - [Structure (general schema)](#structure-general-schema)  
    - [Content of `batch_setup.json`](#content-of-batch_setupjson)  
    - [Create the Settings Files](#how-to-create-the-settings-files)  
- [Execution Logic](#execution-logic)  
- [GUI for Metric Visualization](#gui-for-metric-visualization)
- [Third-Party Projects Used](#third-party-projects-used)  
- [References](#references)  
- [License](#license)  
- [Contact](#contact)


# Features

- Automated execution of a videogrammetry pipeline (JKeyframer + Meshroom / Colmap)
- Automated execution of a photogrammetry pipeline (Meshroom / Colmap)
- Support for batch and single input processing
- Quality assessment of generated point clouds (Meshroom)
- Usable via CLI or Python API

---
# Requirements
- Windows 10 or 11 (64-bit)
- Python 3.12 (should work with 3.10–3.12)
- optional: cuda GPU
---

# Installation
This guide explains how to set up **Pointmark** and all required dependencies (Python and external tools) from scratch.

---

## 1. Clone the Repository

```
git clone https://github.com/GameOfWorldREAL/Pointmark.git
cd pointmark
```

---

## 2. Set Up Python Environment

It is recommended to use a virtual environment

virtual environment:
```
python -m venv venv
venv\Scripts\activate
```

Then install all dependencies:

```
pip install -r requirements.txt
```

---

## 3. Install External Dependencies

### Option 1: Meshroom (full support)

- Download the precompiled version from the [official release page](https://github.com/alicevision/meshroom/releases)
- Extract it and **note the path**

> Meshroom is used for photogrammetry pipelines and template creation.

### Option 2: COLMAP (only automatic reconstruction)

- Recommended for advanced photogrammetry
- Follow the [COLMAP installation guide](https://colmap.github.io/install.html)
- or download the precompiled version from the [official release page](https://github.com/colmap/colmap/releases)
- Extract it and **note the path**
---

# Usage

## Option 1: CLI

---
## pointmark.py – CLI (Single Execution)

This script executes a single Pointmark pipeline. The input can be either a video file, an image folder, or an already reconstructed SFM project. Optionally, metrics and aggregations can be calculated.

### Usage

python pointmark.py [OPTIONS] (-i IMAGES_PATH | -v VIDEO_PATH | -d SFM_DATA_PATH) -p PROJECT_PATH

### Flag Summary for pointmark.py

| Flag                | Type        | Description                                                                 |
|:--------------------|:------------|:----------------------------------------------------------------------------|
| -p, --project_path  | Output Path | Target directory where the project will be saved.                           |
| -i, --images_path   | Input Path  | Path to a folder containing images.                                         |
| -v, --video_path    | Input Path  | Path to a video file.                                                       |
| -d, --sfm_data_path | Input Path  | Path to a folder containing existing SFM (Structure from Motion) data.      |
| -y, --yes           | Option      | Automatically answers all prompts with "Yes."                               |
| -m, --metrics       | Option      | Calculates metrics and saves them in the project directory.                 |
| --ooi               | Option      | Performs Object of Interest (OOI) detection.                                |
| --aggregation_mode  | Option      | Defines how metrics are aggregated (min, max, mean, none). Default is none. |
| -e, --export        | Output Path | Exports the metric data with the corresponding point cloud.                 |

---

### Required Arguments (One of the following options):
- -**p**, --**project_path** <Path>  
  Target directory where the project will be saved. if not set, you will be asked to choose a directory.

Only one of these arguments may be used (mutually exclusive):

- -**i**, --**images_path** <Path>  
  Path to a folder containing images. If no path follows, None is passed.

- -**v**, --**video_path** <Path>  
  Path to a folder containing video files. If no path follows, None is passed.

- -**d**, --**sfm_data_path** <Path>  
  Path to a folder containing existing SFM data. If no path follows, None is passed.

> **Important:** if you set only the flag, it is expected that the project folder contains a functional data structure.


### Optional Arguments

- -**y**, --**yes**  
  Automatically answers all prompts with "Yes."


- -**m**, --**metrics**  
  Calculates metrics and saves them in the project directory.


- --**ooi**  
  Performs Object of Interest (OOI) detection.  
  (will only be executed if -m is set)  


- --**aggregation_mode** [min|max|mean|none]  
  Defines how metrics are aggregated across regions or time.
  Default value: "none"  
  (will only be executed if -m is set)  


- -**e**, --**export**  
  Exports the metric data with the corresponding point cloud into a JSON file. If the path not set default will be set
  Default value: `./path_to_project/export/project_name_pm.json`  
  (will only be executed if -m is set)  
  
### Example Calls

1. Processing a video with automatic confirmation:
```
python pointmark.py -v ./path_to_video/video.mp4 -p ./path_to_project/project -y
```

2. Processing an image folder with metrics and mean aggregation:
```
python pointmark.py -i ./path_to_images/images/ -p ./path_to_project/project -m --aggregation_mode "mean"
```
3. Quality assessment on existing SFM data with ooi and aggregation:
```
python pointmark.py -d ./path_to_data/sfm_data/ -p ./path_to_project/project -m --ooi --aggregation_mode "min"
```

---

## pointmark_batch.py – CLI (Batch Execution)

This script executes a batch Pointmark pipeline. The input can be a folder of videos, image sets, or SFM data sets. Optionally, metrics and aggregations can be calculated.

### Usage
````
python pointmark_batch.py [OPTIONS] (-i IMAGE_SETS_PATH | -v VIDEO_FILES_PATH | -d SFM_DATA_PATH) -p PROJECT_PATH -s SETUP_PATH
````


### Flag Summary for pointmark_batch.py

| Flag                   | Type        | Description                                                                 |
|:-----------------------|:------------|:----------------------------------------------------------------------------|
| -p, --project_path     | Output Path | Target directory where the batch project data will be stored.               |
| -s, --setup_path       | Input Path  | Path to an existing batch setup configuration.                              |
| -i, --image_sets_path  | Input Path  | Path to a folder containing multiple image sets.                            |
| -v, --video_files_path | Input Path  | Path to a folder containing multiple video files.                           |
| -d, --sfm_data_path    | Input Path  | Path to a folder containing multiple SFM data sets.                         |
| -m, --metrics          | Option      | Calculates metrics and saves them in each subproject directory.             |
| --ooi                  | Option      | Performs Object of Interest (OOI) detection.                                |
| --aggregation_mode     | Option      | Defines how metrics are aggregated (min, max, mean, none). Default is none. |
| -e, --export           | Output Path | Exports the metric data with the corresponding point cloud.                 |

---

### Required Arguments (One of the following options):

- -**p**, --**project_path** <Path>  
  Target directory where the batch project will be saved.

- -**s**, --**setup_path** <Path>  
  Provides a folder with setup config and settings.

> if --setup_path is not set, it is expected that the project folder contains a correct internal structure.

Only one of these arguments may be used (mutually exclusive):

- -**i**, --**image_sets_path** <Path>  
  Path to a folder containing multiple image sets.

- -**v**, --**video_files_path** <Path>  
  Path to a folder containing video files.

- -**d**, --**sfm_data_path** <Path>  
  Path to a folder containing SFM data sets.

> **Important:** If only the flag is set without a path, it is expected that the project folder contains a correct internal structure.

---

### Optional Arguments

- -**m**, --**metrics**  
  Calculates metrics and saves them per subproject.


- --**ooi**  
  Performs Object of Interest (OOI) detection.  
  (will only be executed if -m is set)  


- --**aggregation_mode** [min|max|mean|none]  
  Defines how metrics are aggregated across subprojects. Default: "none"  
  (will only be executed if -m is set)  


- -**e**, --**export**  
  Exports the metric data with the corresponding point cloud into a JSON file for each project to:  
  `./path_to_project/export/project_name_pm.json`  
  (will only be executed if -m is set)

---

### Example Calls

1. Batch processing of videos with auto-confirm and metrics:
```
python pointmark_batch.py -v ./videos/ -p ./batch_project/ -s ./setup_path/ -m
```

2. Batch processing of image sets with metrics and mean aggregation:
```
python pointmark_batch.py -i ./image_sets/ -p ./batch_project/ -s ./setup_path/ -m --aggregation_mode "mean"
```

3. Quality assessment on existing SFM data with ooi and aggregation:
```
python pointmark_batch.py -d ./sfm_sets/ -p ./batch_project/ -s ./setup_path/ -m --ooi --aggregation_mode "max"
```


---
## Option 2: Python Integration

Pointmark can be used directly as a Python module. This allows programmatic control of all pipelines, equivalent to the CLI tools.

### Required Setup
You must initialize the following helper class:

```python
from src.general.python.filePaths import PointmarkPaths
pointmark_paths = PointmarkPaths("path/to/pointmark")
```

This class automatically resolves all internal paths required by Pointmark.

> this will change in future updates and will be automatically done
---

### Single Project

These functions process one input item (video, image folder, or SFM data).

#### Available Functions

```python
pointmarkFromVideo(pointmark_paths, project_path, src_video_path, ...)
pointmarkFromImages(pointmark_paths, project_path, images_path, ...)
pointmarkFromPG(pointmark_paths, project_path, sfm_data_path, ...)
```

#### Common Parameters

| Name              | Type   | Description                                      |
|-------------------|--------|--------------------------------------------------|
| `pointmark_paths` | `PointmarkPaths` | Required root path object                        |
| `project_path`    | `Path` | Output directory for the project                 |
| `*_path`          | `Path` | Input (video, images, or sfm data, depending)    |
| `skip`            | `bool` | Skip questions and answer with yes               |
| `metrics`         | `bool` | Compute quality metrics                          |
| `aggregation_mode`| `str`  | `"min"`, `"max"`, `"mean"`, `"none"`             |
| `ooi`             | `bool` | Run Object of Interest detection (needs metrics) |

---
### Example (SfM Data)

```python
from src.pointmark.singleRun.pointmarkFromVideo import pointmarkFromPG
from src.general.python.filePaths import PointmarkPaths
from pathlib import Path

paths = PointmarkPaths("path/to/pointmark/")

pointmarkFromPG(
    pointmark_paths=paths,
    project_path=Path("output/project1/"),
    src_video_path=Path("input/sfm_data/"),
    metrics=True,
    aggregation_mode="min",
    ooi=True
)
```
---

### Batch Project

These functions process multiple input folders using a batch setup.

#### Available Functions

```python
batchFromVideo(pointmark_paths, project_path, videos_path, setup_path, ...)
batchFromImages(pointmark_paths, project_path, image_sets_path, setup_path, ...)
batchFromPG(pointmark_paths, project_path, sfm_data_path, setup_path, ...)
```

#### Additional Parameters

| Name               | Type             | Description                                            |
|--------------------|------------------|--------------------------------------------------------|
| `pointmark_paths`  | `PointmarkPaths` | Required root path object                              |
| `project_path`     | `Path`           | Output directory for the project                       |
| `*_path`           | `Path`           | Folder containing multiple inputs                      |
| `setup_path`       | `Path`           | Batch config folder with setup JSON files              |
| `skip`             | `bool`           | Skip questions and answer with yes                     |
| `metrics`          | `bool`           | Compute quality metrics                                |
| `aggregation_mode` | `str`            | `"min"`, `"max"`, `"mean"`, `"none"`                   |
| `ooi`              | `bool`           | Run Object of Interest detection (needs metrics)       |

---
### Example (Video)
```python
from src.pointmark.batchRun.batchFromVideo import batchFromVideo
from src.general.python.filePaths import PointmarkPaths
from pathlib import Path

paths = PointmarkPaths("path/to/pointmark/")

batchFromVideo(
    pointmark_paths=paths,
    project_path=Path("output/batch_project/"),
    videos_path=Path("input/videos/"),
    setup_path=Path("input/setup/"),
    metrics=True,
    aggregation_mode="mean",
    ooi=True
)
```
---
> the required data structures are explained in `How To's`
---
# How-To's

---
## How-To: Prepare the input data

### videos
- single: path to video
- batch: path to folder with videos

### images
- single: path to folder with images
- batch: path to folder of folders with images

### sfm data
- single: path to folder with sfm data
- batch: path to folder of folders with sfm data

---
## How-To: Get sfm data
### Meshroom 
- add a `ConvertSFMFormat` Node
- connect the `StructureFromMotion` `SfMData` output to the `input` of `ConvertSFMFormat`
- connect the `StructureFromMotion` `Describer Types`  output to the `Describer Types` of `ConvertSFMFormat`
- in `ConvertSFMFormat` change `SfM File Format` to `json`
- keep `Views`, `Intrinsics`, `Extrinsics`, `Structure` and `Observations` activated in `ConvertSFMFormat`
- execute this node
- the `MeshroomCache` will contain the exported `sfm.json`

### Colmap
not supported yet

---
## How-To: Create a Pointmark Batch Setup

This guide explains how to create a valid setup for running Pointmark in batch mode.

---

### 1. Create a Batch Setup Directory

Create a directory called `setup/` (or any name you choose). This folder will contain:

- `batch_setup.json` → defines which settings file is applied to which projects
- one or more `settings.json` files → defines processing parameters

No project folders exist at this point. They will be automatically created during execution.

---

### Structure (general schema)

```
setup/
├── batch_setup.json
├── settings_1.json
├── settings_2.json
```

---

### Content of `batch_setup.json`

This file maps each **settings file** to a list of **project names** to be created using it.

### Rules

- Each settings file (key) must appear only once.
- Each project name must be **unique** across all settings.
- All file paths are interpreted **relative to the location of `batch_setup.json`**.
- Project names refer to **input data name**.  
  Examples:
  - A file `video_1.mp4` in `--video_files_path` → project name: `video_1`
  - A folder `./image_set_1/` in `--image_sets_path` → project name: `image_set_1`
  - A folder `./sfm_data_set_1/` in `--sfm_data_path` → project name: `sfm_data_set_1`

> Each project name must exactly match the name of the corresponding input, depending on which input mode you use when running the batch.  
> If names do not match, Pointmark will not associate data correctly.
---

### Example

```json
{
  "settings_fast.json": ["project_1", "project_3"],
  "settings_highres.json": ["project_2", "project_4", "project_5"]
}
```

In this case:
- Five projects will be created.
- Projects `project_1` and `project_3` will use `settings_fast.json`.
- The rest will use `settings_highres.json`.


### How-To: Create the Settings Files

Each settings file defines how a group of projects should be processed. These files are referenced in `batch_setup.json` and control the use of keyframing tools, reconstruction pipelines, and external dependencies like Meshroom or COLMAP.

---

### What goes into a settings file?

Each settings file is a `.json` file with specific configuration keys. The following options are currently supported:

---

### Example – Using Meshroom

```json
{
  "keyframer": "1",
  "reconst_pipeline": "1",
  "meshroom_path": "path_to_meshroom/Meshroom-2023.3.0/",  #contains meshroom_batch.exe
  "template_path": "path_to_template/template_1.mg"
}
```

- Uses JKeyframer
- Uses Meshroom for reconstruction
- Requires a .mg template file. The recommended way to create this is by building the pipeline in Meshroom's Graph Editor and exporting it as a template. The graph must include a StructureFromMotion node, as it serves as the core of the reconstruction pipeline.
- a standard template is provided: `ressources/templates/meshroom.mg`

---

### Example – Using COLMAP

```json
{
  "keyframer": "1",
  "reconst_pipeline": "2",
  "colmap_path": "path_to_colmap/colmap"  #contains COLMAP.bat
}
```

- Uses JKeyframer
- Uses COLMAP for reconstruction

---

### Explanation of Fields

| Key               | Description                                                            |
|-------------------|------------------------------------------------------------------------|
| `keyframer`       | `"1"` → selects JKeyframer (currently the only option)                 |
| `reconst_pipeline`| `"1"` for Meshroom, `"2"` for COLMAP                                   | |

---

### Validation Rules

- You **must** define both `keyframer` and `reconst_pipeline`.
- You **must not** define `colmap_path` and `meshroom_path` in the same settings file.
- Only include the fields relevant to your selected pipeline.

> COLMAP is not supported for Quality assessment
---


# Execution Logic

- Depending on the input type (video, images, sfm_data), a suitable pipeline function is called:
  - pointmarkFromVideo
  - pointmarkFromImages
  - pointmarkFromPG

- The script checks if the project folder exists and prepares it if necessary.
- If a project is already built, it will not be recomputed, it will try to restore all data

# GUI for Metric Visualization

After processing your project, you can visualize the metrics in an interactive interface.

## Launch the GUI

```
python GUI.py
```

---

## Open a Project

1. Start the GUI  
2. From the menu, select **`File → Open`**  
3. Choose the **path to your project folder**

---

## Automatic Metric Computation

If your project **does not contain precomputed metrics**, the GUI will automatically compute them using:

- `--ooi` (Object of Interest)
- `--aggregation_mode "none"`

> These auto-generated metrics will **not be saved**.

---

## Save Metrics Manually

1. **Right-click** on the project in the list (left panel)  
2. Select **`Save`** – this will permanently store the current metric data.

---

## Tips

- Switch between projects, metrics, and visualization settings at any time.
- The GUI automatically remembers your camera position for each project.
- Use the **OOI visibility slider** to dim or hide points not marked as objects of interest.

# Third-Party Projects Used

Pointmark makes use of several external projects for videogrammetry and analysis. Please refer to their respective license terms and citation guidelines:

- **JKeyframer** (by Marco Block-Berlitz) (https://vividus-verlag.de/3d_rekonstruktion/index.html)  
  A tool for intelligent keyframe selection from video data.  
  License: CC BY-NC 4.0

- **BiRefNet** (https://github.com/ZhengPeng7/BiRefNet)  
  A deep learning model for image correspondence processing and feature detection.  
  Used for: improving sparse point cloud quality.  
  License: MIT License

If you use this software, please also cite the above third-party projects according to their requirements (e.g., BiRefNet).

# References

The following publications were used as scientific foundations for the development of Pointmark:

- Neumann, K. A., Tausch, R., Kutlu, H., Kuijper, A., Santos, P., & Fellner, D. (2025).  
  *Point cloud quality metrics for incremental image-based 3D reconstruction*.  
  **Multimedia Tools and Applications**.  
  [https://doi.org/10.1007/s11042-025-20596-6](https://doi.org/10.1007/s11042-025-20596-6)

- di Filippo, A., Antinozzi, S., Cappetti, N., & Villecco, F. (2024).  
  *Methodologies for Assessing the Quality of 3D Models Obtained Using Close-Range Photogrammetry*.  
  **International Journal on Interactive Design and Manufacturing (IJIDeM)**, 18, 5917–5924.  
  [https://doi.org/10.1007/s12008-023-01428-z](https://doi.org/10.1007/s12008-023-01428-z)

- Javadnejad, F., Slocum, R., Gillins, D., Olsen, M., & Parrish, C. (2021).  
  *Dense Point Cloud Quality Factor as Proxy for Accuracy Assessment of Image-Based 3D Reconstruction*.  
  **Journal of Surveying Engineering**, 147(1), 04020021.  
  [ASCE Library](https://ascelibrary.org/doi/10.1061/%28ASCE%29SU.1943-5428.0000331)

- Zheng, P., Gao, D., Fan, D.-P., Liu, L., Laaksonen, J., Ouyang, W., & Sebe, N. (2024).  
  *Bilateral Reference for High-Resolution Dichotomous Image Segmentation*.  
  **CAAI Artificial Intelligence Research**, 3, 9150038.

- Mauro, M., Riemenschneider, H., Signoroni, A., Leonardi, R., & Van Gool, L. (2014).  
  *A Unified Framework for Content-Aware View Selection and Planning through View Importance*.  
  In: **Proceedings of the British Machine Vision Conference (BMVC)**, September 2014.  
  University of Brescia and ETH Zurich.

- Luhmann, T. (2023).  
  *Nahbereichsphotogrammetrie – Grundlagen – Methoden – Beispiele* (5th ed.).  
  **Wichmann Verlag**, Berlin. ISBN: 978-3-87907-732-8


# License

This project is licensed under the [MIT License](LICENSE).

If you use Pointmark in scientific or production work, please cite:

**Martin Simon (2025):**  
*Pointmark – Automated quality assessment of a point cloud using the example of videogrammetry.*   
GitHub: https://git.imp.fu-berlin.de/martis61/pointcloud_benchmark.  

```
@misc{pointmark2025,
  author       = {Martin Simon},
  title        = {{Pointmark -- Automated quality assessment of a point cloud using the example of videogrammetry}},
  year         = {2025},
  howpublished = {\url{https://github.com/GameOfWorldREAL/Pointmark}},
}
```
# Contact
For questions, feedback, or academic collaboration:  
martin.simon@fu-berlin.de
