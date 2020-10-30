# paragraph-level-QG
paragraph level question generation

## Dataset
Get the data under the ```datasets``` directory.

### 1. SQuAD Split-1: [Article-level Split]
**[Learning to Ask: Neural Question Generation for Reading Comprehension](https://www.aclweb.org/anthology/P17-1123.pdf)**. *ACL 2017*. [[Github]](https://github.com/xinyadu/nqg/tree/master/data)

Randomly divide the dataset at the articlelevel into a training set (80%), a development set
(10%), and a test set (10%).

We use the original dev set in the SQuAD dataset as our dev set, we split the original training set into our training set and test set.

```shell script
cd data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
mkdir squad_split_v1
git clone https://github.com/xinyadu/nqg
cp nqg/data/raw/* squad_split_v1/
cd squad_split_v1
python convert_squad_split1_qas_id.py
cd ..
python preprocess.py --data_dir data/squad_split_v1
```


### 2. SQuAD Split-2: [Sentence-level Split]
**[Neural Question Generation from Text: A Preliminary Study](https://arxiv.org/pdf/1704.01792.pdf)**. *NLPCC 2017*. [[Data]](https://res.qyzhou.me/)

Randomly halve the development set to construct the new development and test sets.


```shell script
cd data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
mkdir squad_split_v2
wget http://res.qyzhou.me/qas_id_in_squad.zip
unzip qas_id_in_squad.zip
cp qas_id_in_squad/train.txt.id squad_split_v2/
cp qas_id_in_squad/dev.txt.shuffle.dev.id squad_split_v2/dev.txt.id
cp qas_id_in_squad/dev.txt.shuffle.test.id squad_split_v2/test.txt.id
cd ..
python preprocess.py --data_dir data/squad_split_v2
```


### 3. NewsQA:
Following the `README.md` in https://github.com/Maluuba/newsqa to download `newsqa.tar.gz`, `cnn_stories.tgz` and `stanford-postagger-2015-12-09.zip` into `maluuba/newsqa` folder; and use the Maluuba's tool to split data as follow.
```shell script
cd data
git clone https://github.com/Maluuba/newsqa
cd newsqa
conda create --name newsqa python=2.7 "pandas>=0.19.2"
conda activate newsqa && pip install --requirement requirements.txt
python maluuba/newsqa/data_generator.py
```

Then, we will have `train.tsv`, `dev.tsv` and `test.tsv` in `datasets/newsqa/split_data` folder.
```shell script
cd ..
python preprocess.py --data_dir data/newsqa
```