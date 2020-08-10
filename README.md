# Pose Estimation with TensorFlow Lite 🚶🏻‍♂️

<!-- Project description -->
This repository contains a trained version of PoseNet that runs directly with TensorFlow Lite.

The trained model returns a set of `(x, y)` paired keypoints containing the ouput of inference. All keypoints are indexed by an ID. You can see the IDs and parts in the following table:

| ID | PART           |
| -- | -------------- |
|  0 | Nose           |
|  1 | Left eye       |
|  2 | Right eye      |
|  3 | Left ear       |
|  4 | Right ear      |
|  5 | Left shoulder  |
|  6 | Right shoulder |
|  7 | Left elbow     |
|  8 | Right elbow    |
|  9 | Left wrist     |
| 10 | Right wrist    |
| 11 | Left hip       |
| 12 | Right hip      |
| 13 | Left knee      |
| 14 | Right knee     |
| 15 | Left ankle     |
| 16 | Right ankle    |


## Prerequisities

Before you begin, ensure you have met the following requirements:

* You have a _Windows/Linux/Mac_ machine running [Python 3.6+](https://www.python.org/).
* You have installed the latest versions of [`pip`](https://pip.pypa.io/en/stable/installing/) and [`virtualenv`](https://virtualenv.pypa.io/en/stable/installation/) or `conda` ([Anaconda](https://www.anaconda.com/distribution/)).


## Setup

To install the dependencies, you can simply follow this steps.

Clone the project repository:
```bash
git clone https://github.com/Estandarte-Digital/pose-estimation.git
cd pose-estimation
```

To create and activate the virtual environment, follow these steps:

**Using `conda`**

```bash
$ conda create -n pose-estimation python=3.7

# Activate the virtual environment:
$ conda activate pose-estimation

# To deactivate (when you're done):
(pose-estimation)$ conda deactivate
```

**Using `virtualenv`**

```bash
# In this case I'm supposing that your latest python3 version is +3.6
$ virtualenv pose-estimation --python=python3

# Activate the virtual environment:
$ source pose-estimation/bin/activate

# To deactivate (when you're done):
(pose-estimation)$ deactivate
```

To install the requirements using `pip`, once the virtual environment is active:
```bash
(pose-estimation)$ pip install -r requirements.txt
```

#### Running the script

Finally, if you want to run the main script:
```bash
$ python run.py
```

This will start your camera and open a window with the output.

#### Modify parameters

If you want to change the parameters of the viewer, the model used, etc., you can directly modify the specs from the `run.py` script.

## Output file

The generated output file is modified in real time. An example of the generated output is the following:

```json

```
