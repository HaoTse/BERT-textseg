# Text segmentaion based on BERT

## Data preprocess
```
python preprocess.py --src_path [source data location] --tgt_path [pt files location]
```

## Training
```
python train.py --mode [train, test] --model_path [model location] --result_path [result location] --data_path [pt files location]
```