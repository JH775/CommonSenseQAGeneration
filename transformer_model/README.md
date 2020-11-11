## README

The project was installed using these commands 
```
module load anaconda/py3

source env create -n transformer
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

Results: 

eval_loss = 0.6931444406509399
eval_acc = 0.5319634703196348
epoch = 2.9879518072289155
total_flos = 943164415374720
