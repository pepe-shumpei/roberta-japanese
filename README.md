# Pretraining RoBERTa using Japanese Wikipedia 

# Overview
 
[RoBERTa](https://aclanthology.org/2021.ccl-1.108/)を日本語Wikipediaを用いて事前学習する
 
# Requirement
 
* certifi==2021.5.30
* sentencepiece==0.1.96
* wikiextractor==3.0.4
 
# Installation
 
```bash
pip install -r requirements.txt
```
[fairseq](https://github.com/pytorch/fairseq)のインストール

```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
 
# Usage
```bash
git clone https://github.com/pepe-shumpei/roberta-japanese.git
cd roberta-japanese
```

## (1) Preprocess Japanese Wikipedia
日本語Wikipediaをダウンロードして解凍する
```bash
cd data
wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2
bunzip2 jawiki-latest-pages-articles.xml.bz2
```

日本語Wikipediaの前処理
```bash
python -m wikiextractor.WikiExtractor jawiki-latest-pages-articles.xml
find text/ | grep wiki | awk '{system("cat "$0" >> wiki.txt")}'
sed -i '/^<[^>]*>$/d' wiki.txt
sed -i '/^$/d' wiki.txt
```

## (1') Preprocess small data
日本語Wikipediaのすべてのダウンロードと前処理には時間がかかるため、先頭100万文の日本語Wikipedia(wiki.txt.tar.gz)を用意した。

wiki.txt.tar.gzを解凍する。
```bash
cd data
tar -zxvf wiki.txt.tar.gz
```

**Note:** (1)もしくは(1')の片方を実行する。

## (2) Split dataset into train,validation and test data
```bash
bash split_dataset.sh
```

## (3)Tokenize
トーカナイズには[sentencepiece](https://github.com/google/sentencepiece)を用いる。

まず、sentencepieceモデルを訓練する。
```bash
python train-sp.py
```
次に、訓練したsentencepieceを適用する。
```bash
python apply-sp.py
```

## (4) Pretraining RoBERTa
fairsq-preprocessとfairseq-trainを用いてRoBERTaの事前学習を行う。
fairseq-preprocessとfairseq-trainの詳細は[fairseq Command-line Tools](https://fairseq.readthedocs.io/en/latest/command_line_tools.html)を参照。

まず、fairseq-preprocessを用いてデータをバイナリ化する。
```bash
bash preprocess.sh
```

次に、fairseq-trainを用いてRoBERTaで事前学習を行う。
```bash
bash train.sh
```
