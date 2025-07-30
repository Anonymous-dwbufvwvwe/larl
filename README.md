## Installation

#### Create conda env

```shell
conda create -n IDEAM python=3.10.9
```

#### Install dependency

The `requirements.txt` file should list libraries needed:

```bash
pip install -r requirements.txt
```



## Usage

#### Training process is straight forward:

Follow the installation instructions in: https://github.com/huggingface/alignment-handbook/tree/main?tab=readme-ov-file#installation-instructions

Then: 

```shell
python PPO_train.py
```

#### So is the test process:

```shell
python test.py
```



### Note:

Remember to change kinematics.py in highway_env for MPC-DCBF implemention.