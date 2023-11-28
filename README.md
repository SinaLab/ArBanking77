ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic
======================
ArBanking77 is an MSA and Dialectal Arabic Corpus for Arabic Intent Detection in Banking Domain. It consists of 31,404
samples (MSA and Palestinian dialects). This repo contains the source-code and sample dataset to train and evaluate
Arabic Intent Detection model.


ArBanking77 Corpus
--------
ArBanking77 consists of 31,404 (MSA and Palestinian dialects) that are manually Arabized and localized from the original
English Banking77 dataset; which consists of 13,083 queries. Each query is classified into one of the 77 classes (
intents) including card arrival, card linking, exchange rate, and automatic top-up. You can find the list of these 77
intents in the `data/bank77_intents.csv` file. A neural model based on AraBERT was fine-tuned on the ArBanking77
dataset (F1-score 92% for MSA, 90% for PAL)


Full Corpus Download
--------
A sample data is available in the `data` directory. However, the entire ArBanking77 corpus is
available to download upon request for academic and commercial use. However, we cannot provide the augmented data.

[Request to download ArBanking77 (corpus and the model)](https://sina.birzeit.edu/arbanking77/)


Model Download
--------
[SinaLab HuggingFace](https://huggingface.co/SinaLab/ArBanking77)

Online Demo
--------
You can try our model using this [demo link](https://sina.birzeit.edu/arbanking77/).

Requirements
--------
At this point, the code is compatible with `Python 3.11`

Clone this repo

    git clone https://github.com/SinaLab/ArabicNER.git

This package has dependencies on multiple Python packages. It is recommended to Conda to create a new environment
that mimics the same environment the model was trained in. Provided in this repo `requirements.txt` from which you
can create a new conda environment using the command below.

    conda create -n env_name python=3.11

Install requirements using pip command:

    pip install -r requirements.txt


Project Structure
--------
```
.
├── data                            <- data dir
│   ├── Banking77_Arabized_test_sample.csv
│   ├── Banking77_Arabized_train_sample.csv
│   ├── Banking77_Arabized_val_sample.csv
│   ├── Banking77_intents.csv
├── outputs
│   ├── models                      <- trained models
│   ├── results                     <- evaluation results and reports
├── src                             <- training and evaluation scripts
│   ├── run_glue_no_trainer.py
│   ├── run_glue_no_trainer_eval.py
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Model Training
--------
You can start model training by running the following command. It's recommended to pass the arguments demonstrated below
to get results similar to the ones reported in the paper.

    python ./src/run_glue_no_trainer.py
        --model_name_or_path aubmindlab/bert-base-arabertv02 
        --train_file ./data/Banking77_Arabized_MSA_PAL_train_sample.csv
        --validation_file ./data/Banking77_Arabized_MSA_PAL_val_sample.csv 
        --seed 42 
        --max_length 128 
        --learning_rate 4e-5 
        --num_train_epochs 20 
        --per_device_train_batch_size 64 
        --output_dir ./outputs/models

Evaluation
--------
Additionally, you can evaluate the trained model on `Banking77_Arabized_MSA_test_sample.csv` and `Banking77_Arabized_PAL_test_sample.csv` test sets as follows:

    python ./src/run_glue_no_trainer_eval.py 
        --model_name_or_path ./outputs/models 
        --validation_file ./data/Banking77_Arabized_MSA_test_sample.csv 
        --seed 42 
        --per_device_eval_batch_size 64 
        --results_dir ./outputs/results 
        --log_path ./outputs/logs/log.txt

Credits
-------
This research was funded by the Palestinian Higher Council for Innovation and Excellence and the Scientific and
Technological Research Council of Türkiye (TÜBİTAK) under project No. 120N761 - CONVERSER: Conversational AI System for
Arabic.


Citation
-------
Mustafa Jarrar, Ahmet Birim, Mohammed Khalilia, Mustafa Erden, and Sana
Ghanem: [ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic](http://www.jarrar.info/publications/JBKEG23.pdf).
In Proceedings of the 1st Arabic Natural Language Processing Conference (ArabicNLP), Part of the EMNLP 2023. ACL.