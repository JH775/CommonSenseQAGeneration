## README

The project was installed using these commands 
```
module load anaconda/py3

conda create -n transformer
source activate transformer

pip install tensorflow
pip install torch 
pip install transformers
conda install -c anaconda tensorflow-gpu
git clone https://github.com/JH775/CommonSenseQAGeneration.git
cd ./transformer_model
sbatch run.sh
```


in case one should experience issues with importing numpy: 
```
pip uninstall numpy 
# delete numpy folder at location displayed in error message  
conda install numpy  
```


To run parameter tuning:
```
pip install "ray[tune]"
pip install optuna

```

#### Results: 
| dataset                                                      | AlBERT |
|--------------------------------------------------------------|--------|
| CustomMC                                                     | 96%    |
| ProtoQA                                                      | 77%    |
| Winogrande                                                   | 81%    |
| CustomMC + ProtoQA (tested on Winogrande)                    | 50%    |
| CustomMC + ProtoQA (tested on Winogrande, tuned with optuna) | 55%    |
