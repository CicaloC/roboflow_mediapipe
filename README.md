# Installation guide

conda create -n newenv python=3.8

activate newenv

conda install pytorch torchvision torchaudio -c pytorch
conda install ftfy regex tqdm requests pandas seaborn
pip install pycocotools tensorflow

pip install freemocap

***From my attempts installing roboflow dependencies after having freemocap
installed will create an error from version control***************

