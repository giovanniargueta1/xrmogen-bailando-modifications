# xrmogen_colab_tutorial

The framework(xrmogen) can be found here:https://github.com/openxrlab/xrmogen
Along with the original repository of Bailando:https://github.com/lisiyao21/Bailando  

The notebook shows the appropriate steps to run the Baliando/DanceRev model (from Xrmogen framework) on Google Colab.

All code in the train_bailando.ipynb file sets up the Python virtual environment to train/test both models available in Xrmogen, but it only displays the commands for Bailando. Please
reference the original repo to mimic the commands for testing/training DanceRev. 

First to ensure all cloned repos are saved in your drive, make sure to mount /content/drive by running 
```python
from google.colab import drive
drive.mount('/content/drive')
```
Now use the following commands to install miniconda
```python
! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
! chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
! bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local


```
*Sidenote if you want an updated conda version we can run the command !conda update -n base -c defaults conda , but for stability reasons you may want to stick with this verision 
Now we want to create our virtual env called xrmogen that uses python 3.8
```python
!conda create -y --name xrmogen python=3.8
```

To make sure we are writing to the env's bin directory, we can add to our PATH by running these commands, ensuring the Python version is Python 3.8.20
```python
import os

# Add Conda environment's bin directory to the PATH
os.environ["PATH"] = "/usr/local/envs/xrmogen/bin:" + os.environ["PATH"]

!which python
```

Next we can clone the xrmogen and mmhuman3d repos, I recommend storing them in the /content/drive/MyDrive folder, for further organization put them under a parent folder called "workspace"

```python
!git clone https://github.com/openxrlab/xrmogen.git /content/drive/MyDrive/workspace/xrmogen
!git clone https://github.com/open-mmlab/mmhuman3d.git /content/drive/MyDrive/workspace/mmhuman3d
```
Then we will install all requirements of both xrmogen and mmhuman3d. Note we only use conda install for a few libraries and the rest are pip-installed. Please ignore installing pickle5 for mmhuman3D as Python 3.8 has the proper pickle wheel for this project.
After the installation is done please ensure the conda list displays the libraries installed under conda and the pip list shows its respective libraries (right versions) 
```python
#xrmogen requirements
!conda install -n xrmogen ffmpeg -y
#ignore the line below if you are relying on Colab GPU, find compatible version of pytorch with CUDA
!conda install -n xrmogen pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
!conda run -n xrmogen pip install "imagio==2.15.0"
!conda run -n xrmogen pip install Pillow

#mmhuman3d requirements (ignore pickle)

!conda install -n xrmogen astropy -y
!conda run -n xrmogen pip install "cdflib==0.3.20"

!conda run -n xrmogen pip install chumpy
!conda run -n xrmogen pip install colormap
!conda run -n xrmogen pip install easydev
!conda run -n xrmogen pip install einops
!conda run -n xrmogen pip install h5py
!conda run -n xrmogen pip install matplotlib
!conda run -n xrmogen pip install "numpy==1.23.5"
!conda run -n xrmogen pip install opencv-python
!conda run -n xrmogen pip install "pandas==1.5.3"
!conda run -n xrmogen pip install plyfile
!conda run -n xrmogen pip install scikit-image
!conda run -n xrmogen pip install scipy
!conda run -n xrmogen pip install smplx
!conda run -n xrmogen pip install tqdm
!conda run -n xrmogen pip install trimesh
!conda run -n xrmogen pip install vedo
!conda run -n xrmogen pip install "mmcv==1.6.1"
!conda run -n xrmogen pip install "torch==1.7.1"
!conda run -n xrmogen pip install torchvision



!conda run -n xrmogen conda list
!conda run -n xrmogen pip list     
```
Before running any test or train config files, cd into the parent xrmogen folder. Afterwards please download the preprocessed data as instructed in the xrmogen repo under a folder name /data. Also for testing purposes please install the bailando
pre-trained weights under a created /example folder. Remember the only difference between running the python file under a traditional IDE is in colab you have to include "!" before python, so "!python main.py.." 

*Please note if you want to take advantage of Colab's selection of GPUS you have to install the appropriate CUDA version. For example if you are using the A100 GPU, you should install
PyTorch with CUDA 11.8 by using the command

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#then check for GPU and CUDA compatibility
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
```


