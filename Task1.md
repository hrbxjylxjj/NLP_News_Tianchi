# 一、赛题理解

## 赛题数据

赛题以匿名处理后的新闻数据为赛题数据，数据集报名后可见并可下载。赛题数据为新闻文本，并按照字符级别进行匿名处理。整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐的文本数据。

赛题数据由以下几个部分构成：训练集20w条样本，测试集A包括5w条样本，测试集B包括5w条样本。为了预防选手人工标注测试集的情况，我们将比赛数据的文本按照字符级别进行了匿名处理。

在数据集中标签的对应的关系如下：{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}

评价标准为类别f1_score的均值，选手提交结果与实际测试集的类别进行对比，结果越大越好。


```python
import pandas as pd
```


```python
df_test=pd.read_csv('/Users/alice/Desktop/test_a.csv',sep='\t')
df_train=pd.read_csv('/Users/alice/Desktop/train_set.csv',sep='\t')
```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2967 6758 339 2021 1854 3731 4109 3792 4149 15...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>4464 486 6352 5619 2465 4802 1452 3137 5778 54...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7346 4068 5074 3747 5681 6093 1777 2226 7354 6...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>7159 948 4866 2109 5520 2490 211 3956 5520 549...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>3646 3055 3055 2490 4659 6065 3370 5814 2465 5...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5399 3117 1070 4321 4568 2621 5466 3772 4516 2...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2491 4109 1757 7539 648 3695 3038 4490 23 7019...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2673 5076 6835 2835 5948 5677 3247 4124 2465 5...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4562 4893 2210 4761 3659 1324 2595 5949 4583 2...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4269 7134 2614 1724 4464 1324 3370 3370 2106 2...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200000 entries, 0 to 199999
    Data columns (total 2 columns):
     #   Column  Non-Null Count   Dtype 
    ---  ------  --------------   ----- 
     0   label   200000 non-null  int64 
     1   text    200000 non-null  object
    dtypes: int64(1), object(1)
    memory usage: 3.1+ MB



```python
df_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50000 entries, 0 to 49999
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   text    50000 non-null  object
    dtypes: object(1)
    memory usage: 390.8+ KB


注意到text属性是object，转换数据类型


```python
df_train['text'] = df_train['text'].apply(lambda x: list(map(lambda y: int(y), x.split())))
df_test['text'] = df_test['text'].apply(lambda x: list(map(lambda y: int(y), x.split())))
```


```python
df_train['label'].value_counts()
```




    0     38918
    1     36945
    2     31425
    3     22133
    4     15016
    5     12232
    6      9985
    7      8841
    8      7847
    9      5878
    10     4920
    11     3131
    12     1821
    13      908
    Name: label, dtype: int64



可以看到科技类最多

看一下文本长度


```python
df_train['text'].map(lambda x: len(x)).describe()
```




    count    200000.000000
    mean        907.207110
    std         996.029036
    min           2.000000
    25%         374.000000
    50%         676.000000
    75%        1131.000000
    max       57921.000000
    Name: text, dtype: float64




```python
df_test['text'].map(lambda x: len(x)).describe()
```




    count    50000.000000
    mean       909.844960
    std       1032.313375
    min         14.000000
    25%        370.000000
    50%        676.000000
    75%       1133.000000
    max      41861.000000
    Name: text, dtype: float64




```python

```
