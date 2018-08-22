# An-Attentive-Neural-Model-for-labeling-Adverse-Drug-Reactions

## Model architecture

![model](https://github.com/Deep1994/An-Attentive-Neural-Model-for-labeling-Adverse-Drug-Reactions/raw/master/img/model.png)

## Requirments

+ Python 3.5.2
+ [Keras](http://keras-cn.readthedocs.io/en/latest/) (deep learning library, verified on 2.1.3)
+ [NLTK](http://www.nltk.org/) (NLP tools, verified on 3.2.1)

## Datasets

We evaluate our model on two ADRs labeling datasets. One is an ADRs-related **Twitter** corpus that includes many informal vocabularies and irregular grammar, and the other is a biomedical text extracted from **PubMed** abstracts with many professional terms and technical descriptions.

### Twitter dataset

The Twitter dataset we used is the same as that used in paper [Deep Learning for Pharmacovigilance: Recurrent Neural Network Architectures for Labeling Adverse Drug Reactions in Twitter Posts](https://academic.oup.com/jamia/article/24/4/813/3041102). This dataset includes two Twitter datasets. The first dataset is **Twitter ADR Dataset(v1.0)**. You can find the dataset download message at  [http://diego.asu.edu/Publications/ADRMine.html](http://diego.asu.edu/Publications/ADRMine.html). _It is against Twitter's Terms of Service to publish the text of tweets, so the original Twitter ADR Dataset v1.0 authors have provided tweet ID's instead._ The second dataset 