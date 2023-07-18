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
