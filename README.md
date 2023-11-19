## ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic


Online Demo
--------
You can try our model using the demo link below

[https://sina.birzeit.edu/arbanking77/](https://sina.birzeit.edu/arbanking77/)


ArBanking77 Corpus
--------
ArBanking77 consists of 31,404 (MSA and Palestinian dialects) that are manually Arabized and localized from the original English Banking77 dataset; which consists of 13,083 queries. Each query is classified into one of the 77 classes (intents) including card arrival, card linking, exchange rate, and automatic top-up. You can find the list of these 77 intents in the `data\bank77_intents.csv` file. A neural model based on AraBERT was fine-tuned on the ArBanking77 dataset (F1-score 92% for MSA, 90% for PAL)


Corpus Download
--------
A sample data is available in the `data` directory. However, the entire ArBanking77 corpus is 
available to download upon request for academic and commercial use. Request to download 
ArBanking77 (corpus and the model).

[https://sina.birzeit.edu/arbanking77/](https://sina.birzeit.edu/arbanking77/)


Model Download
--------
HuggingFace: [https://huggingface.co/SinaLab/ArBanking77](https://huggingface.co/SinaLab/ArBanking77)


Model Training
--------

```commandline
    python run_glue_no_trainer.py 
    --model_name_or_path aubmindlab/bert-base-arabertv2
    --train_file ./data/Banking77_Arabized_train_sample.csv 
    --validation_file ./data/Banking77_Arabized_val_sample.csv 
    --seed 42 
    --max_length 128 
    --learning_rate 4e-5
    --num_train_epochs 20  
    --per_device_train_batch_size 32 
    --output_dir ./results
```

File
The training code is available in the `run_glue_no_trainer.py` file, but make sure to install the libraries listed in the `requirements.txt` file before starting the training.


Credits
-------
This research is partially funded by the Palestinian Higher Council for Innovation and Excellence.


Citation
-------
Mustafa Jarrar, Ahmet Birim, Mohammed Khalilia, Mustafa Erden, and Sana Ghanem: [ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic](http://www.jarrar.info/publications/JBKEG23.pdf). In Proceedings of the 1st Arabic Natural Language Processing Conference (ArabicNLP), Part of the EMNLP 2023. ACL.
