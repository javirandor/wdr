# "That is a Suspicious Reaction!": Interpreting Logits Variation to Detect NLP Word-Level Adversarial Attacks
### Supplementary material

## Introduction
Adversarial attacks are a major challenge faced by current machine learning research. These purposely crafted inputs fool even the most advanced models, precluding their deployment in safety-critical applications. Extensive research in computer vision has been carried to develop reliable defense strategies. However, the same issue remains less explored in natural language processing. Our work presents a model-agnostic detector of adversarial examples. The approach identifies patterns in the logits of the target classifier when perturbing the input text. The proposed detector improves the current state-of-the-art performance in recognizing adversarial inputs and exhibits strong generalization capabilities across different models, datasets, and word-level attacks.

## Code usage guide
In this section, we explain how to use the code to reproduce or extend the results. Keep in mind that this version of the code is built to be self-contained and easy to follow. You may want to split it up for efficiency; e.g. pre-compute and store logits. This way, WDR will not be generated in each execution. Also, you may want to increase the number of detectors trained as we did to report statistical significance.
Important considerations:
* Install the required libraries using `pip install -r requirements.txt`
* We really recommend using GPUs. Otherwise, adversarial attacks generation and model predictions become really slow. Our experimental setup was hosted in Google Colab.
* Using these scripts, all results can be reproduced. Also adversarial sentences are provided in `Generating Adversarial Samples/Data` to avoid repeating the whole process of generating adversarial examples and help future researchers.
* This code is a simplification of the experimental setup. Therefore, exact same results won't most likely be obtained. 

## Code structure
The code is divided following this structure:
* `/Generating Adversarial Samples`: this folder contains the code required to generate adversarial attacks using TextAttack library and structure the results in a DataFrame.
    * `Command Line Adversarial Attack.ipynb` -> script to produce adversarial sentences
    * `Data` -> This folder that contains all samples generated. The dataset files are named using the convention `{dataset}_{attack}_{model}.csv`
* `/Classifier`:
    * `/Training Classifier/Training_Classifier.ipynb` -> transforms sentences into logits differences that will be used as input for the classifier and then  adversarial classifier training and models performance comparison using logit difference approach.
    * `/Testing Classifier/Testing_Classifier.ipynb` -> transforms sentences into logits differences that will be used as input for the classifier and then test the test classifier performance on the dataset and model considered.
    * `cnn_imdb_textattack.py` -> this file loads TextAttack CNN for IMDB pretrained model as required by the library.
    * `lstm_imdb_textattack.py` -> this file loads TextAttack LSTM for IMDB pretrained model as required by the library.
    * `cnn_agnews_textattack.py` -> this file loads TextAttack CNN for AGNEWS pretrained model as required by the library.
    * `lstm_agnews_textattack.py` -> this file loads TextAttack LSTM for AGNEWS pretrained model as required by the library.
    * `imdb_classifier.pickle` -> best adversarial classifier stored for analysis trained on the `imdb_pwws_distilbert.csv` setup. Used to generate paper results.
    * `ag-news_classifier.pickle` -> classifier stored for analysis trained on the `ag-news_pwws_distilbert.csv` setup. Used to generate paper results.
* `/FGWS`: adapted code from the [official implementation](https://github.com/maximilianmozes/fgws) for our experimental setup. Can be used to reproduce benchmark results.

## Code usage and execution pipeline
These are the steps required to reproduce the project results:
1. `Generating Adversarial Samples/Command Line Adversarial Attack.ipynb` -> generate adversarial samples for the desired setup and store them. This step can be skipped because results are already provided in the repository (see `Generating Adversarial Samples/Data`)
2. `Classifier/Training Classifier/Training_Classifier.ipynb` -> Compute logits difference for adversarial and original samples for the desired dataset and create an input dataframe for the model. Then, train and store the adversarial classifier.
3. `Classifier/Testing Classifier/Testing Classifier.ipynb` -> Compute logits difference for adversarial and original samples for the desired dataset and then test the results on the created input dataframe containing the logit differences.

Optional: You can use the code within `FGWS/` to reproduce the baseline results against which our method was benchmarked.
