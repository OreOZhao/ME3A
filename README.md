# ME3A
Code for "ME3A: A Multimodal Entity Entailment Framework for Multimodal Entity Alignment".

## Dependencies
- python 3.9
- pytorch 1.12.1
- transformers 4.24.0
- tqdm

## Dataset

You can download the datasets from:
- DBP15K and SRPRS dataset from [JAPE](https://github.com/nju-websoft/JAPE), [RSN](https://github.com/nju-websoft/RSN), or [SDEA](https://github.com/zhongziyue/SDEA);
- MMKG dataset from [MEAformer](https://github.com/zjunlp/MEAformer) or [EVA](https://github.com/cambridgeltl/eva);
- OpenEA100K dataset from [OpenEA](https://github.com/nju-websoft/OpenEA).

1. Unzip the datasets in `ME3A/data`.
2. Preprocess the datasets.

```bash
cd src
python DBPPreprocess.py
python SRPRSPreprocess.py
python MMKGProcess.py
python OpenEAProcess.py

```

3. Preprocess the image feature pickle files with `src/ImageProcess.py`

## Pre-trained Language Model

You can download the pre-trained language model `bert-base-multilingual-uncased` from [huggingface](https://huggingface.co/bert-base-multilingual-uncased) and put the model in `ME3A/pre_trained_models`

## Project Structure

```
ME3A/
├── src/: The soruce code. 
├── data/: The datasets. 
│   ├── DBP15k/: The downloaded DBP15K benchmark. 
│   │   ├── fr_en/
│   │   ├── ja_en/
│   │   ├── zh_en/
│   ├── entity-alignment-full-data/: The downloaded SRPRS benchmark. 
│   │   ├── en_de_15k_V1/
│   │   ├── en_fr_15k_V1/
│   ├── mmkg/: The downloaded MMKG benchmark. 
│   │   ├── fb_db_15k/
│   │   ├── fb_yg_15k/
│   ├── OpenEA_dataset_v1.1/: The downloaded OpenEA benchmark. 
│   │   ├── EN_FR_100K_V1/
│   │   ├── EN_DE_100K_V1/
├── pre_trained_models/: The pre-trained transformer-based models. 
│   ├── bert-base-multilingual-uncased: The model used in our experiments.
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
│   ├── bert-base-uncased
```

## How to run

To run ME3A, use the example script `run_dbp15k_me3a.sh`, `run_srprs.sh`, `run_mmkg_me3a.sh`, `run_openea100k.sh`. 
You could customize the following parameters:

- --nsp: trains with ME3A-NSP. If not specified, the default is ME3A-MLM.
- --neighbor: adds neighbor sequences in training. If not specified, the model is only trained with attribute sequences.
- --woattr: uses only neighbor sequences in training. If not specified, the model is only trained with attribute sequences.
- --template_id: changes the template of prompt according to the `template_list` in `src/KBConfig.py`.
- --visual: adds visual prefixes in training. 
- --prefix: the count of visual prefixes of one entity.

You could also run FT-EA with `src/FTEATrain.py`.

## Acknowledgements
Our codes are modified based on [SDEA](https://github.com/zhongziyue/SDEA). We would like to appreciate their open-sourced work.
