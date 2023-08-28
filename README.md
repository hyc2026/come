# COME: Commit Message Generation with Modification Embedding

## Environment
```
conda env create -f environment.yml python=3.6
```

## Dataset
Download Link: https://pan.baidu.com/s/1ykwgZFT6KpXmb7tSqNbp9g?pwd=37o8

Extracted Code: 37o8
```
come
└── data
    ├── jit
    │   ├── openstack
    │   │   ├── test.jsonl
    │   │   └── train.jsonl
    │   └── qt
    │       ├── test.jsonl
    │       └── train.jsonl
    └── summarize
        ├── cpp
        │   ├── test.jsonl
        │   ├── train.jsonl
        │   └── valid.jsonl
        ├── csharp
        │   ├── test.jsonl
        │   ├── train.jsonl
        │   └── valid.jsonl
        ├── java
        │   ├── test.jsonl
        │   ├── train.jsonl
        │   └── valid.jsonl
        ├── java1
        │   ├── test.jsonl
        │   ├── train.jsonl
        │   └── valid.jsonl
        ├── javascript
        │   ├── test.jsonl
        │   ├── train.jsonl
        │   └── valid.jsonl
        └── python
            ├── test.jsonl
            ├── train.jsonl
            └── valid.jsonl
```

## Train and Eval
```
bash run.sh java
```
java can be replaced to java1, cpp, csharp, javascript, python

## Autometrics Test
```
bash test_all.sh
```

## Checkpoints


|dataset|bleu|meteor|rouge-l|cider|link|
| :------: | :------: | :------: |:------:|:------:|:------:|
| MCMD-java | 27.17 | 16.91 |34.59|1.9|https://pan.baidu.com/s/1IjSzW03fvB2Eo9xt7saU5Q?pwd=flcv|
| MCMD-C# | 27.29 | 17.77 |33.33|1.91|https://pan.baidu.com/s/1-pCw8-0ryRITlX6fO_l3bg?pwd=da8l |
| MCMD-C++ | 20.8 | 14.55 |27.01|1.25|https://pan.baidu.com/s/1HfIo3_WszbllfsVNKIRJxw?pwd=ab9i |
| MCMD-python | 23.17 | 16.46 |30.48|1.5|https://pan.baidu.com/s/1-LmdyAldpIcDM0KO02gSLA?pwd=3eq2 |
| MCMD-javascript | 26.91 | 17.84 |34.44|1.92|https://pan.baidu.com/s/1nkJQ7P6s1OHa6qzK1TG77g?pwd=r2pg |
| CoDiSum-java | 19.46 | 10.7 |24.56| 0.82|https://pan.baidu.com/s/1GU_ccBFLsQt9L3-F53ddcQ?pwd=lkab |