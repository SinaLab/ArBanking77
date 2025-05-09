ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic
======================
ArBanking77 is an MSA and Dialectal Arabic Corpus for Arabic Intent Detection in the Banking Domain. It consists of 31,038
samples (MSA, Palestinian, Saudi, Moroccan, and Tunisian dialects). This repo contains the source-code and dataset to train and evaluate
Arabic Intent Detection model.


ArBanking77 Corpus
--------
ArBanking77 consists of 31,038 (MSA, Palestinian, Saudi, Moroccan, and Tunisian dialects) that are manually Arabized and localized from the original
English Banking77 dataset; which consists of 13,083 queries. Each query is classified into one of the 77 classes (intents) including card arrival, card linking, exchange rate, and automatic top-up. You can find the list of these 77 intents in the `./data/Banking77_intents.csv` file. A neural model based on AraBERT was fine-tuned on the ArBanking77 dataset (F1-score 92% for MSA, 90% for PAL).
You can also find the `./data/Banking77_full_corpus.csv` file, which contains 31,038 samples spanning Modern Standard Arabic (MSA), Palestinian, Saudi, Moroccan, and Tunisian dialects. 

**Full Data Format Banking77_full_corpus.csv** <br>
| Column Name          | Description |
|----------------------|-------------|
| Intent_ID       | Unique identifier for each intent |
| Intent_en       | Intent description in English |
| Intent_ar       | Intent description in Arabic |
| QID             | Unique identifier for each question record in the corpus |
| Question_en     | The English version of the question |
| QuestionID_MSA1 | Identifier for the first MSA-generated question |
| Question_MSA1   | First MSA version of the question |
| QuestionID_MSA2 | Identifier for the second MSA-generated question |
| Question_MSA2   | Second MSA version of the question |
| QuestionID_PAL1 | Identifier for the first Palestinian dialect question |
| Question_PAL1   | First Palestinian dialect question |
| QuestionID_PAL2 | Identifier for the second Palestinian dialect question |
| Question_PAL2   | Second Palestinian dialect question |
| QuestionID_Saudi1 | Identifier for the first Saudi dialect question |
| Question_Saudi1 | First Saudi dialect question |
| QuestionID_Saudi2 | Identifier for the second Saudi dialect question |
| Question_Saudi2 | Second Saudi dialect question |
| QuestionID_Moroccan | Identifier for the Moroccan dialect question |
| Question_Moroccan | Moroccan dialect question |
| QuestionID_Tunisian | Identifier for the Tunisian dialect question |
| Question_Tunisian | Tunisian dialect question |

Each question ID contains a prefix indicating the dataset split: <br>
Tr... → Training set <br>
Te... → Test set <br>
D... → Development (Dev) set <br>
The numerical part of the ID is unique within each split. When removing the prefix, all numbers remain unique within their respective training, test, or development set.


Full Corpus Download
--------
Data is available in the `data` directory for academic and commercial use. However, we cannot provide the augmented data.

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

    git clone https://github.com/SinaLab/ArBanking77.git

This package has dependencies on multiple Python packages. It is recommended that Conda be used to create a new environment
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
│   ├── Banking77_Arabized_MSA_PAL_train.csv
│   ├── Banking77_Arabized_MSA_PAL_val.csv
│   ├── Banking77_Arabized_MSA_test.csv
│   ├── Banking77_Arabized_PAL_test.csv
│   ├── Banking77_Arabized_Moroccan_test.csv
│   ├── Banking77_Arabized_Saudi_test.csv
│   ├── Banking77_Arabized_Tunisian_test.csv
│   ├── Banking77_intents.csv
│   ├── Banking77_full_corpus.csv
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
        --train_file ./data/Banking77_Arabized_MSA_PAL_train.csv
        --validation_file ./data/Banking77_Arabized_MSA_PAL_val.csv 
        --seed 42 
        --max_length 128 
        --learning_rate 4e-5 
        --num_train_epochs 20 
        --per_device_train_batch_size 64 
        --output_dir ./outputs/models

Evaluation
--------
Additionally, you can evaluate the trained model on `Banking77_Arabized_MSA_test.csv`, `Banking77_Arabized_PAL_test.csv`, `Banking77_Arabized_Moroccan_test.csv`, `Banking77_Arabized_Saudi_test.csv`, and `Banking77_Arabized_Tunisian_test.csv` test sets as follows:

    python ./src/run_glue_no_trainer_eval.py 
        --model_name_or_path ./outputs/models 
        --validation_file ./data/Banking77_Arabized_MSA_test.csv 
        --seed 42 
        --per_device_eval_batch_size 64 
        --results_dir ./outputs/results 
        --log_path ./outputs/logs/log.txt

Credits
-------
The first phase of this research was partially funded by the Palestinian Higher Council for Innovation and Excellence and the Scientific and TÜBİTAK under project No. 120N761 - CONVERSER: Conversational AI System for Arabic.

Citation
-------
Mustafa Jarrar, Ahmet Birim, Mohammed Khalilia, Mustafa Erden, and Sana Ghanem: [ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic](http://www.jarrar.info/publications/JBKEG23.pdf).
In Proceedings of the 1st Arabic Natural Language Processing Conference (ArabicNLP), Part of the EMNLP 2023. ACL.

Sanad Malaysha, Mo El-Haj, Saad Ezzini, Mohammed Khalilia, Mustafa Jarrar, Sultan Nasser, Ismail Berrada, Houda Bouamor: [AraFinNLP 2024: The First Arabic Financial NLP Shared Task](https://www.jarrar.info/publications/MEEKJNBB24.pdf). In Proceedings of the Second Arabic Natural Language Processing Conference (ArabicNLP 2024), Bangkok, Thailand. Association for Computational Linguistics.
