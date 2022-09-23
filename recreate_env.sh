#!/bin/bash

envname="seqnn"

conda deactivate

conda env remove -n $envname
conda update -n base conda
conda env create -n $envname -f ./environment.yaml
#python -m nltk.downloader all

conda activate $envname
pip install -e .
