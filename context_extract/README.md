# CommonSenseQA dataset generation for CSE576



### Used datasets 
    - COSMOQA (curl  https://github.com/wilburOne/cosmosqa/raw/master/data/train.csv --output ./input/train.csv )
    - Common Sense Explanations (CoS-E),
    - CommonGen, 
    - Social IQa from Hugging Face datasets (https://huggingface.co/datasets)

### Instructions
Clone repo
```
    git clone https://github.com/JH775/CommonSenseQAGeneration.git

```
Prepare data using /input folder code.
```
    cd ./input 
    curl  https://github.com/wilburOne/cosmosqa/raw/master/data/train.csv --output ./train.csv '
    python extract.py
    python combine_files.py

    
```
Run GPT-2 Text-Generating Model w/ GPU in Google Colab and follow instructions to upload data_ALL.txt and train model and download   context_500_ALL.csv and context_1000_ALL.csv and move to /generated_context/

Clone and deploy nrl-qasrl model from https://github.com/nafitzgerald/nrl-qasrl to agave.asu.edu.
```
    module load anaconda/py3
    conda create --name nrlQA python=3.6
    source activate nrlQA
    git clone https://github.com/JH775/CommonSenseQAGeneration.git
    git checkout master     
    cd ./data
    mkdir glove
    cd ./glove
    curl http://nlp.stanford.edu/data/glove.6B.zip 
    unzip glove.6B.zip
    cd ../ 
    mkdir elmo
    cd ./glove
    curl https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
    curl https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json

    cd ../../scripts
    ./download_pretrained.sh    

```
Prepare input data by running and upload to agave.asu.edu/path/to/dataset/folder/of/ from local folder /input/nrlQA-input.
 ```
 python3 /input/nrlQA-input/getSentences.py
 scp -r /input/nrlQA-input/ <ASURITE>@agave.asu.edu:/home/<ASURITE>/nrl-qasrl/

```
Run model to create QA sets.
```
    sbatch nrl_run.sh
```
Download output.txt to local machine.
```
     scp <ASURITE>@agave.asu.edu:/home/<ASURITE>/nrl-qasrl/out_nrlQA.txt  ./generated_questions
```
Extract QA-set and vocab.txt from output.txt
```
    python3 ./generated_questions/fill_DF.py
    python3 ./input/BERT_data/create_BERT_date.py

```
Clone paraphrasing model in anaconda terminal.
```
    conda create --name bert_para python=3.6
    source activate bert_para 
    git clone 
    pip install -r requirements.py
```
Add vocab.txt, QA-set to data folder of project and run Answer_BERT.ipynb.
Run `combine_datasets.py` to combine all datasets into single csv-file.
Collect finished dataset /final_dataset/final_CS_dataset.csv.


