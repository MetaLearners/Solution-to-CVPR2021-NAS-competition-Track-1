### Code of CVPR2021 NAS track1 stage2 3rd Solution Illustration

Please follow [readme.ipynb](readme.ipynb) to reproduce the results.

You can also reproduce results by running following commands.

First, please download [dataset](https://aistudio.baidu.com/aistudio/datasetdetail/76994) and [architectures](https://aistudio.baidu.com/aistudio/datasetdetail/73326) and put them under `./data`. Then, download trained supernet models from released package and put them under `./saved_models`. The files should in the following organization:
- data
    - cifar-100-python.tar.gz
    - Track1_final_archs.json
- saved_models
    - model_97122.pdparams
    - model_97131.pdparams

#### Train

Run following command to train the supernet.
```cmd
python -m supernet.scripts.train --n 18 --kd --sandwich
```

#### Evaluate

Run following command to reproduce the 0.97122 and 0.97131 submission.
```
python -m supernet.scripts.evaluate --path ./saved_models/model_97122.pdparams --output ./saved_models/submit_97122.json
python -m supernet.scripts.evaluate --path ./saved_models/model_97131.pdparams --output ./saved_models/submit_97131.json
```