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
import pickle
import torch
from torchtext.vocab import Vectors
from app1.transformer import TransformerClassification
from app1.config import *


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
    ret = tokenizer_mecab(text)  # Mecabの単語分割

    return ret


def pickle_load(path):
    with open(path, 'rb') as f:
        TEXT = pickle.load(f)
    return TEXT

def pickle_dump(TEXT, path):
    with open(path, 'wb') as f:
        pickle.dump(TEXT, f)


def create_tensor(sentence,TEXT, max_length):
    #入力文章をTorch Teonsor型にのINDEXデータに変換
    token_ids = torch.ones((max_length)).to(torch.int64)
    ids_list = list(map(lambda x: TEXT.vocab.stoi[x] , sentence))
    print(ids_list)
    for i, index in enumerate(ids_list):
        token_ids[i] = index
    return token_ids


def load_model(model_path, TEXT):
    net_trained = TransformerClassification(text_embedding_vectors=TEXT.vocab.vectors, d_model=300,
                  max_seq_len=256, output_dim=2)
    param = torch.load(model_path)
    net_trained.load_state_dict(param)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_trained.eval()   # モデルを検証モードに
    net_trained.to(device)
    return net_trained


def create_vocab(pkl_path):
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                            use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length,
                            init_token="<cls>", eos_token="<eos>")
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    train_ds, val_ds = torchtext.data.TabularDataset.splits(
        path=data_path, train='train.tsv',validation='test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])
    japanese_fastText_vectors = Vectors(name='/mnt/c/Users/sinfo/Desktop/pytorch/pytorch_advanced-master/django/sample/app1/data/model.vec')
    # ベクトル化したバージョンのボキャブラリーを作成
    TEXT.build_vocab(train_ds, vectors=japanese_fastText_vectors, min_freq=5)
    pickle_dump(TEXT, pkl_path)
    return TEXT


def create_input_data(sentence, TEXT):
    sentence = tokenizer_with_preprocessing(sentence)
    sentence.insert(0, '<cls>')
    sentence.append('<eos>')
    #   '<cls>': 2, '<eos>': 3,
    sentence = create_tensor(sentence,TEXT, 256)
    sentence = sentence.unsqueeze_(0)   #  torch.Size([256])  > torch.Size([1, 256])
    # GPUが使えるならGPUにデータを送る
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = sentence.to(device)
    # mask作成
    input_pad = 1  # 単語のIDにおいて、'<pad>': 1 
    input_mask = (input != input_pad)

    return input, input_mask


def create_tensor(sentence, TEXT, max_length):
    #入力文章をTorch Teonsor型にのINDEXデータに変換
    token_ids = torch.ones((max_length)).to(torch.int64)
    ids_list = list(map(lambda x: TEXT.vocab.stoi[x] , sentence))
    print(ids_list)
    for i, index in enumerate(ids_list):
        token_ids[i] = index
    return token_ids


# HTMLを作成する関数を実装


def highlight(word, attn):
    "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"

    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


def mk_html(input, preds, normlized_weights_1, normlized_weights_2, TEXT):
    "HTMLデータを作成する"

    # indexの結果を抽出
    index = 0
    sentence = input.squeeze_(0) # 文章  #  torch.Size([1, 256])  > torch.Size([256]) 
    pred = preds[0]  # 予測

 
    # indexのAttentionを抽出と規格化
    attens1 = normlized_weights_1[index, 0, :]  # 0番目の<cls>のAttention
    attens1 /= attens1.max()

    attens2 = normlized_weights_2[index, 0, :]  # 0番目の<cls>のAttention
    attens2 /= attens2.max()

    if pred == 0:
        pred_str = "Negative"
        # 表示用のHTMLを作成する
        html = '推論ラベル：<font color=red>{}</font><br><hr>'.format(pred_str)
    else:
        pred_str = "Positive"
        # 表示用のHTMLを作成する
        html = '推論ラベル：<font color=blue>{}</font><br><hr>'.format(pred_str)

  
    # 1段目のAttention
    html += '[TransformerのAttentionを可視化]<br><br>'
    html += '<div class="alert alert-info text-black rounded p-3">'
    for word, attn in zip(sentence, attens1):
        html += highlight(TEXT.vocab.itos[word], attn)
    html += "<br><br>"
    html += "</div>"

    # 2段目のAttention
    #html += '[TransformerBlockの2段目のAttentionを可視化]<br>'
    #for word, attn in zip(sentence, attens2):
    #    html += highlight(TEXT.vocab.itos[word], attn)

    html += "<br><br>"

    return html


