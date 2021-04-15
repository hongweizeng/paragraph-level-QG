# paragraph-level-QG
paragraph level question generation

## 1. Preprocess Dataset
Get the data under the ```data``` directory.
```shell script
git clone https://github.com/hongweizeng/paragraph-level-QG
cd paragraph-level-QG
mkdir data
```

### 1.1 [Article-level Split] [Du et al., 2017] (SQuAD Split-1)
**[Learning to Ask: Neural Question Generation for Reading Comprehension](https://www.aclweb.org/anthology/P17-1123.pdf)**. *ACL 2017*. [[Github]](https://github.com/xinyadu/nqg/tree/master/data)

[Du et al., ACL 2017](https://arxiv.org/pdf/1705.00106.pdf) (70484 | 10570 | 11877): We use the original dev* set in the SQuAD dataset as our dev set, we split the original training* set into our training set and test set.

[Zhao et. al, EMNLP 2018](https://www.aclweb.org/anthology/D18-1424.pdf) [Reversed dev-test setup]: (70484 | 11877 | 10570) + (dropped samples): we use dev* set as test set, and split train* set into train and dev sets randomly with ratio 90%-10%.
we keep all samples instead of only keeping the sentence-question pairs that have at least one non-stop-word in common (with 6.7% pairs dropped) as in (Du et al., 2017). 
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
python preprocess.py -data_dir data -dataset squad_split_v1
```


### 1.2 [Sentence-level Split] [Zhou et al., 2017] [Zhao et al., 2018] (SQuAD Split-2)
**[Neural Question Generation from Text: A Preliminary Study](https://arxiv.org/pdf/1704.01792.pdf)**. *NLPCC 2017*. [[Github]](https://github.com/magic282/NQG) [[Data]](https://res.qyzhou.me/)

[Zhou et al., NLPCC 2017](https://arxiv.org/pdf/1704.01792.pdf) (86,635 | 8,965 | 8,964): Randomly halve the development set to construct the new development and test sets.

[Zhao et. al, EMNLP 2018](https://www.aclweb.org/anthology/D18-1424.pdf) (? | ? | ?): Similar to (**_Zhou_** et al., 2017), we split dev* set into dev and test sets randomly with ratio 50%-50%. 
The split is done at sentence level.

[Tuan et al., AAAI 2020](https://arxiv.org/pdf/1910.10274.pdf) (87,488 | 5,267 | 5,272): similar to (Zhao et al., 2018), we keep the SQuAD train set and randomly split the SQuAD dev set into our dev and test set with the ratio 1:1.

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
python preprocess.py -data_dir data -dataset squad_split_v2
```


### 1.3 NewsQA:
**[NewsQA: A Machine Comprehension Dataset](https://arxiv.org/pdf/1611.09830.pdf)**. *Rep4NLP@ACL 2017*.

[Liu et al., WWW 2019](https://arxiv.org/pdf/1902.10418.pdf) (77,538 | 4,341 | 4,383): 
In our experiment, we picked a subset of NewsQA where answers are top-ranked and are composed of a contiguous sequence of words within the input sentence of the document.

[Tuan et al., AAAI 2020](https://arxiv.org/pdf/1910.10274.pdf) (76,560 | 4,341 | 4,292): 
In our experiment, we select the questions in NewsQA where answers are sub-spans within the articles. 
As a result, we obtain a dataset with 76k questions for train set, and 4k questions for each dev and test set.

[Ours] ((92,549 | 5,166 | 5,126) = [Consensus Statistics: 102,841](https://www.microsoft.com/en-us/research/project/newsqa-dataset/#!stats))

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
python preprocess.py -data_dir data -dataset newsqa 
```


## 2. Train & Test
Specify the configurations in `.yml` file.
```shell script
python main.py -train -test -config configs/test.yml 
```


## References

[1]. https://github.com/magic282/NQG

[2]. https://github.com/seanie12/neural-question-generation

## Citation
If you find this code is helpful, please cite our paper:
```
@article{zeng-etal-2021-EANQG,
    title = {Improving Paragraph-level Question Generation with Extended Answer Network and Uncertainty-aware Beam Search},
    author = {Hongwei Zeng, Zhuo Zhi, Jun Liu and Bifan Wei},
    url = {https://github.com/hongweizeng/paragraph-level-QG},
    booktitle = {Information Sciences},
    year = {2021}
}
```
