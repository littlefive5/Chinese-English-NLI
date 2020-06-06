import numpy as np
import pandas as pd
import csv
import jsonlines
from pandas.core.frame import DataFrame
import json
import requests
import urllib.parse
import sys
import execjs  # 可通过pip install PyExecJS安装，用来执行js脚本
import time
import http.client    #修改引用的模块
import hashlib        #修改引用的模块
from urllib import parse
import random

sentence1=[]
sentence2=[]
label=[]
with open("multinli_1.0/multinli_1.0_train.jsonl", "r+", encoding="utf-8-sig") as f:
    for item in jsonlines.Reader(f):
        if item['gold_label']=='entailment' or item['gold_label']=='contradiction':
            sentence1.append(item['sentence1'])
            sentence2.append(item['sentence2'])
            label.append(item['gold_label'])



train_data=list(zip(sentence1,sentence2,label))

class BaiduTranslate:
    def __init__(self,fromLang,toLang):
        self.url = "/api/trans/vip/translate"
        self.appid="**********" #申请的账号
        self.secretKey = '*********'#账号密码
        self.fromLang = fromLang
        self.toLang = toLang
        self.salt = random.randint(32768, 65536)

    def BdTrans(self,text):
        sign = self.appid + text + str(self.salt) + self.secretKey
        md = hashlib.md5()
        md.update(sign.encode(encoding='utf-8'))
        sign = md.hexdigest()
        myurl = self.url + \
                '?appid=' + self.appid + \
                '&q=' + parse.quote(text) + \
                '&from=' + self.fromLang + \
                '&to=' + self.toLang + \
                '&salt=' + str(self.salt) + \
                '&sign=' + sign
        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)
            response = httpClient.getresponse()
            html = response.read().decode('utf-8')
            html = json.loads(html)
            dst = html["trans_result"][0]["dst"]
            return  True , dst
        except Exception as e:
            return False , e

def chunks(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]

def main():
    BaiduTranslate_test = BaiduTranslate('en','zh')
    datas = chunks(train_data,1000)
    index=0
    for data in datas:
        corpus=[]
        for item in data:
            item = list(item)
            source = item[0]
            text = BaiduTranslate_test.BdTrans(source)
            text=text[1]
            item.append(text)
            print(source,text)
            data1={}
            data1['sentence1'] = item[3]
            data1['sentence2'] = item[0]
            data1['label'] = 'right'
            data2={}
            data1['sentence1'] = item[3]
            data1['sentence2'] = item[1]
            data1['label'] = item[2]
            corpus.append(data1)
            corpus.append(data2)
        all_data={}
        all_data['data'] = corpus
        file = 'outputs/'+str(index).zfill(4)+'.json' 
        index+=1
        with open(file,'w+',encoding='utf-8') as f:
            json.dump(all_data,f,ensure_ascii=False)


if __name__ == '__main__':
    main()