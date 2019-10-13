# django-transformer

###  本リポジトリのプログラムは下記の書籍「つくりながら学ぶ！PyTorchによる発展ディープラーニング」第7章「自然言語による感情分析（Transformer)を参考に作成したものです。

https://github.com/YutaroOgawa/pytorch_advanced/blob/master/LICENSE  

上記書籍では、IMDbのデータセットを用いた英語文章のネガポジ分類になっていますが、このリポジトリでは日本語文章でのネガポジ分類モデルになっています。  

学習データにはTISが無料で公開している機械学習で感情解析を行うためのデータセット「chABSA-dataset」を用いています。  

https://github.com/chakki-works/chABSA-dataset

chABSA-datasetのデータの本文がネガティブか、ポジティブ化を自動判定するDjangoのWEBアプリケーションのコードも付属しています。     
なお、全データセットのうち、1970個を訓練データ数、843個をテストデータとしてモデルを構築しています。

Djangoアプリのデモ動画  
![transformer](https://user-images.githubusercontent.com/34405452/66707507-7f5c6500-ed7c-11e9-9a2d-342100379cfd.gif)

# フォルダ構成  

- notebook  
　　モデル作成に至るまでの各種コード（JupyterNotebook形式）
- source  
    Djangoアプリケーションのソースコード

- data  
   chABSA：chABSA-datasetデータセットからネガポジ分類用に生成した学習データファイル（tsv)  
   14_steps_fastText_weight.pth：fastTextを用いた学習済みモデル(pytorch)  
   ※fastText(日本語学習済みモデルは容量が大きいため各自JupterNotebook記載の手順に従いmodel.vecをダウンロードのこと)  
   text.pkl：torchtextを使って学習データから生成した以下のようなVocabデータが格納されています。  

```
{'<unk>': 0,'<pad>': 1, '<cls>': 2, '<eos>': 3,'0': 4, '、': 5,'の': 6,'は': 7,'た': 8,'まし': 9,'円': 10,'に': 11,　　・・・省略・・・
```
※pytorchでVocabデータを生成するには時間がかかるため、WEBアプリで推論時間を短縮するためにtext.pklをロードすることで高速化しています。


# Djangoアプリ構築手順


### 1.1 リポジトリをClone

### 1.2 各種ライブラリー導入  

主に以下のモジュールを利用している。
```
Django==2.2.6
django-bootstrap4==1.0.1
django-widgets-improved==1.5.0
mecab-python3==0.996.2
torch==1.1.0
torchsummary==1.5.1
torchtext==0.4.0
torchvision==0.3.0
```

必須のモジュールを整理しきれなかったので、エラーが出る場合はrequirements.txtを参照して環境をそろえてみて下しさい。  

**※本アプリはWindowsのUbuntu環境で構築しています。**    


### 1.3 text.pklの作成手順  

```
python manage.py shell
from app1.utils import *
from app1.config import *
TEXT = create_vocab(pkl_path)
```
※sample\app1\data\text.pklが生成される。

### 1.4 アプリ起動

```
cd sample  
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

アクセスＵＲＬ  
http://127.0.0.1:8000/demo/sentiment


###各種ファイルの概要
```
- model.vec(FastText日本語モデルファイル）は\sample\app1\data配下に配置する。  
- 14_steps_fastText_weight.pth(14epochでの学習済みモデルファイル：14epoch以上は過学習してしまうため)
- train.tsv（学習用データ)
- test.tsv(検証用データ)
- text.pkl(日本語学習済みfastTextを使ってTEXT.build_vocabで生成したvocabデータファイル)
- data\chABSA\e000xx_ann.json(chABSA-datasetデータセット生データ)
```

