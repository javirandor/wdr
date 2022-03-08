# FGWS benchmark reproduction
#### Adadpted from original repository: Frequency-Guided Word Substitutions for Detecting Textual Adversarial Examples

## Obtaining the data
To download the necessary datasets and pre-trained embeddings, run
```
cd data
sh download_data.sh
```

## Detect adversarial examples
Since we are using pre-existing models and datasets, we can directly start detecting attacks. For this, several steps are required.

1. Using the script `transform_data.ipynb` to transform your attack data into the required format for FGWS execution. It will create within `Data FGWS/` a folder for each experimental setup with a name defined as follows `[dataset]_[attack]_[model]`. Within each of these folders, you will find 3 different files: `adv_examples.pkl`, `test_pols.pkl`, `test_texts.pkl`
2. Data for each setup must be manually placed within FGWS folders before detection as follows:
    * `test_pols.pkl` and `test_texts.pkl` must overwrite those within `data/models/roberta/imdb/data/`.
    * `adv_examples.pkl` must overwrite the file in `data/attacks/limit_10000/roberta/prioritized/imdb/test_set/`
3. Choose target model in `detect.py`. You will find a comment where the model can be defined. Uncomment the lines and adjust the parameter referring to your target model. For example, for LSTM pretrained on IMDB by textattack, the code should look as follows:

``` 
model = textattack.models.helpers.LSTMForClassification.from_pretrained('lstm-ag-news')
tokenizer = model.tokenizer
is_huggingface = False
```

Once everything is configured, the detection can be executed using the following command:

```
python3 detect.py -mode detect -dataset imdb -model_type roberta -gpu -limit 10000 -attack prioritized -fp_threshold 0.9
```

Make sure that `limit` value is always greater or equal to your number of samples.

Results will be printed after completion.
