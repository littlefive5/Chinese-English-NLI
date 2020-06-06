# Chinese-English-NLI
In this program we inference the relation between two different languages.

NLI is an important task in NLP.  In most datasets, the premise and the hypothesis are in the same language.  In this project we generate a dataset with MultiNLI and XNLI in which the premise is Chinese and the hypothesis is English. Then we do the inference between different language.

I use the ESIM([Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf)) to make the predict. Since it is a cross language dataset, I tried two different ways in the encoding part. 

### About the datasets

The XNLI is a NLI datasets with 15 different languages, I choose the Chinese and English part. And I use the **Baidu** translate api to translate the premise in MultiNli to Chinese.

The code is in the file `prepare_data.py` and `prepare the data.ipynb`

The data format is json.

##### An example

```
{
'sentence1': '所以，我一个人上了法庭，告诉他们真相，但这对我没有任何好处。',
'sentence2': So, I went to court by myself and told them the truth, but it didn't do me any good.',
'label': 'right'
}
```



### Two ways to train the model.

First, I use two Bi-LSTM to encode the Chinese and English texts separately. Here I use the BertTokenizer to tokenize the two languages. I use 'Bert-for-Chinese-vocab' and 'Bert-uncased-vocab' to build the vocab.

Then I use the 'bert-multilingual-vocab' and just one Bi-LSTM​ to encode both the Chinese and English texts.

The performance is shown below.

The pretrained model achieves the following performance on the SNLI dataset:

|          | Accuracy (%) |
| :------- | :----------: |
| Two LSTM |     76.0     |
| One LSTM |     78.9     |

Precision for different types.

##### Two LSTM

|               | precision | recall | F1   |
| ------------- | --------- | ------ | ---- |
| right         | 0.88      | 0.92   | 0.90 |
| entailment    | 0.68      | 0.48   | 0.56 |
| contradiction | 0.63      | 0.76   | 0.69 |

##### One LSTM

|               | precision | recall | F1   |
| ------------- | --------- | ------ | ---- |
| right         | 0.85      | 0.97   | 0.90 |
| entailment    | 0.67      | 0.66   | 0.66 |
| contradiction | 0.79      | 0.61   | 0.69 |

### Train with glove and word2vec

In the last part, I just initialize the nn.Embedding randomly. Also I use the BPE to tokenize the text. I try to use the [GloVE](https://nlp.stanford.edu/projects/glove/) and [Word2vec](https://github.com/Embedding/Chinese-Word-Vectors?tdsourcetag=s_pctim_aiomsg) to code the words with two LSTM. The performance is :



|                                   | Accuracy (%) |
| :-------------------------------- | :----------: |
| Two LSTM using BPE                |     76.0     |
| Two LSTM using Pretrained vectors |     77.4     |

##### Two LSTM using Pretrained vectors

|               | precision | recall | F1   |
| ------------- | --------- | ------ | ---- |
| right         | 0.86      | 0.94   | 0.90 |
| entailment    | 0.66      | 0.61   | 0.64 |
| contradiction | 0.71      | 0.65   | 0.68 |





