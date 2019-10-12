# 自然言語処理による感情分析（Transformer）


import glob
import os
import io
import string
import re
import sys
import random
import spacy
import torchtext
import mojimoji
import string
import MeCab
from torchtext.vocab import Vectors



def get_chABSA_DataLoaders_and_TEXT(max_length=256, batch_size=8):
    """IMDbのDataLoaderとTEXTオブジェクトを取得する。 """


    def preprocessing_text(text):
        
        # 半角・全角の統一
        text = mojimoji.han_to_zen(text) 
        # 改行、半角スペース、全角スペースを削除
        text = re.sub('\r', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('　', '', text)
        text = re.sub(' ', '', text)
        # 数字文字の一律「0」化
        text = re.sub(r'[0-9 ０-９]+', '0', text)  # 数字

        # カンマ、ピリオド以外の記号をスペースに置換
        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
            else:
                text = text.replace(p, " ")

        return text

    # 分かち書き
    def tokenizer_mecab(text):
        m_t = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        text = m_t.parse(text)  # これでスペースで単語が区切られる
        ret = text.strip().split()  # スペース部分で区切ったリストに変換
        return ret

    # 前処理と分かち書きをまとめた関数を定義
    def tokenizer_with_preprocessing(text):
        text = preprocessing_text(text)  # 前処理の正規化
        ret = tokenizer_mecab(text)  # Janomeの単語分割

        return ret


    # データを読み込んだときに、読み込んだ内容に対して行う処理を定義します
    max_length = 256
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    # フォルダ「data」から各tsvファイルを読み込みます
    train_ds, val_ds = torchtext.data.TabularDataset.splits(
        path='./data/', train='train.tsv',
        validation='test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])

    # torchtextで日本語ベクトルとして日本語学習済みモデルを読み込む
    japanese_fastText_vectors = Vectors(name='./data/model.vec')

    # ベクトル化したバージョンのボキャブラリーを作成します
    TEXT.build_vocab(train_ds, vectors=japanese_fastText_vectors, min_freq=5)

    # DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
    train_dl = torchtext.data.Iterator(
        train_ds, batch_size=batch_size, train=True)
    
    val_dl = torchtext.data.Iterator(
        val_ds, batch_size=batch_size, train=False, sort=False)


    return train_dl, val_dl, TEXT



def preprocessing_text(text):
    
    # 半角・全角の統一
    text = mojimoji.han_to_zen(text) 
    # 改行、半角スペース、全角スペースを削除
    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('　', '', text)
    text = re.sub(' ', '', text)
    # 数字文字の一律「0」化
    text = re.sub(r'[0-9 ０-９]+', '0', text)  # 数字

    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    return text

# 分かち書き
def tokenizer_mecab(text):
    m_t = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    text = m_t.parse(text)  # これでスペースで単語が区切られる
    ret = text.strip().split()  # スペース部分で区切ったリストに変換
    return ret

# 前処理と分かち書きをまとめた関数を定義
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)  # 前処理の正規化
    ret = tokenizer_mecab(text)  # Janomeの単語分割

    return ret
    