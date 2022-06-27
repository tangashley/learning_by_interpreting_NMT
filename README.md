# Learning by Interpreting (Part 1: Neural Machine Translation)
We uses [Fairseq](https://github.com/facebookresearch/fairseq) for the NMT experiments. 
It is modified for attention words extraction.

### Required Packages
* Moses for tokenization
* Subword NMT for BPE pre-processing
* [My fork of Fairseq](https://github.com/tangashley/learning_by_interpreting_NMT.git)
* Pytorch
### Run experiment (IWSLT'17 Fr-En with Transformer)

* Download [IWSLT'17 fr-en dataset](https://wit3.fbk.eu/2017-01-c). 
After downloading, creat a folder ``data`` and unzip file ``text/fr/en/fr-en.tgz`` to folder ``data``.
Run the following command to download needed packages (Moses and Subword NMT) and preprocess 
the dataset located in folder ``data``:
    ```
    bash ./iwslt17_fr_en_data_preprocess.sh
    ```
    This script will create a ``iwslt17.tokenized.fr-en`` folder.


* Install fairseq pacekage using ``--editable`` option as the following:
    ```
    cd fairseq
    pip install --editable .
    ```

* To train Transformer baseline model, run the following command:
  ```
  bash ./iwslt17_train_xfmr_baseline.sh
  ```

* To get baseline BLEU score:
    ```
    bash ./iwslt17_xfmr_translate.sh
    ```

* To extract attention words (and positions) from the baseline model, run the following:
    ```
    bash ./iwslt17_fr_en_xfmr_get_attn_files.sh
    ```

* To fine-tune with attention words, run the following:
    ```
    bash ./wmt14_de_en_xfmr_concate_words_preprocess_and_train.sh
    ```
  
* To get BLEU score of the fine-tune model:
    ```
    bash ./iwslt17_xfmr_translate_merged_word.sh
    ```