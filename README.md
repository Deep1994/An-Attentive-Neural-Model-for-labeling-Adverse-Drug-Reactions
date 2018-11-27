# An-Attentive-Neural-Model-for-labeling-Adverse-Drug-Reactions

This work focuses on extraction of Adverse Drug Reactions（ADRs）from ADRs-related tweets and sentences extracted from PubMed abstracts.

Our paper 《[An Attentive Neural Sequence Labeling Model for Adverse Drug Reactions Mentions Extraction](https://ieeexplore.ieee.org/document/8540859)》 has been accepted by [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?reload=true&punumber=6287639).

You can find more details (in Chinese) about our paper via my blog: [Sequence labeling with embedding-level attention](http://deepon.me/2018/11/18/Sequence-labeling-with-embedding-level-attention/).

![model](https://github.com/Deep1994/An-Attentive-Neural-Model-for-labeling-Adverse-Drug-Reactions/raw/master/img/model.png)

## Requirments

+ Python 3.5.2
+ [Keras](http://keras-cn.readthedocs.io/en/latest/) (deep learning library, verified on 2.1.3)
+ [NLTK](http://www.nltk.org/) (NLP tools, verified on 3.2.1)

## Datasets

We use two datasets in our paper, the first one is a Twitter dataset, which is used in paper [Deep learning for pharmacovigilance: recurrent neural network architectures for labeling adverse drug reactions in Twitter posts](https://academic.oup.com/jamia/article/24/4/813/3041102), another dataset is called ADE-Corpus-V2, which is used in paper [An Attentive Sequence Model for Adverse Drug Event Extraction from Biomedical Text](https://arxiv.org/abs/1801.00625) and availe online: [https://sites.google.com/site/adecorpus/home/document](https://sites.google.com/site/adecorpus/home/document).

Because it is against Twitter's Terms of Service to publish the text of tweets, so we cannot provide the first Twitter dataset, you can obtain this dataset from the author of paper [Deep learning for pharmacovigilance: recurrent neural network architectures for labeling adverse drug reactions in Twitter posts](https://academic.oup.com/jamia/article/24/4/813/3041102) so that you can keep your dataset consistent with that used in our paper.

