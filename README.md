# An-Attentive-Neural-Model-for-labeling-Adverse-Drug-Reactions

This work focuses on extraction of Adverse Drug Reactions（ADRs）from ADRs-related tweets and sentences extracted from PubMed abstracts.

Our paper 《[An Attentive Neural Sequence Labeling Model for Adverse Drug Reactions Mentions Extraction](https://ieeexplore.ieee.org/document/8540859)》 has been accepted by [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?reload=true&punumber=6287639).

You can find more details (in Chinese) about our paper via my blog: [Sequence labeling with embedding-level attention](http://deepon.me/2018/11/18/Sequence-labeling-with-embedding-level-attention/).

![model](https://github.com/Deep1994/An-Attentive-Neural-Model-for-labeling-Adverse-Drug-Reactions/raw/master/img/model.png)

## Requirments

+ Python 3.5.2
+ [Keras](http://keras-cn.readthedocs.io/en/latest/) (deep learning library, verified on 2.1.3)
+ [NLTK](http://www.nltk.org/) (NLP tools, verified on 3.2.1)

## Something you need to prepare before runing the code

### Datasets

We use two datasets in our paper, the first one is a Twitter dataset, which is used in paper [Deep learning for pharmacovigilance: recurrent neural network architectures for labeling adverse drug reactions in Twitter posts](https://academic.oup.com/jamia/article/24/4/813/3041102), another dataset is called ADE-Corpus-V2, which is used in paper [An Attentive Sequence Model for Adverse Drug Event Extraction from Biomedical Text](https://arxiv.org/abs/1801.00625) and availe online: [https://sites.google.com/site/adecorpus/home/document](https://sites.google.com/site/adecorpus/home/document).

Because it is against Twitter's Terms of Service to publish the text of tweets, so we cannot provide the first Twitter dataset, you can obtain this dataset from the author of paper [Deep learning for pharmacovigilance: recurrent neural network architectures for labeling adverse drug reactions in Twitter posts](https://academic.oup.com/jamia/article/24/4/813/3041102) so that you can keep your dataset consistent with that used in our paper.

Please get these two datasets ready and put them to twitter_adr/data and pubmed_adr/data, respectively. I have provided the PubMed dataset in pubmed_adr/data.

### Word embedding

For both datasets, we use the pretrained [GloVe 300d](http://nlp.stanford.edu/data/glove.840B.300d.zip) word embedding, please download it and put it to twitter_adr/embeddings.

## Data processing

We have twitter_adr/data_processing.py and pubmed_adr/data_processing.py to process the two datasets, respectively.

## Model

The twitter_adr/model.py and pubmed_adr/model.py are the model code to generate the predictions, and approximateMatch.py is the script which adopts approximate matching and prints the results of the model.

## Results

![results](https://github.com/Deep1994/An-Attentive-Neural-Model-for-labeling-Adverse-Drug-Reactions/raw/master/results/results.jpg)

This is the result obtained by our model. The F1 on the Twitter dataset is about 0.84, which is more than 10% of the previous SOTA (state-of-the-art), and the F1 on the PubMed dataset is about 0.91, which is more than the previous SOTA about 5%. Due to randomness of the result, it is recommended to run the model several times and average their results.

Because our model is essentially focusing on a sequence labeling task, it can be generalized to any token level classification tasks, such as Named Entity Recognition (NER), Part Of Speech tagging (POS taging). Next I want to validate our model on some larger datasets and explore the magical effects of pre-training.

## Thanks to

I want to sincerely shout out to the following work:

+ [Pharmacovigilance from social media: mining adverse drug reaction mentions using sequence labeling with word embedding cluster features ](https://academic.oup.com/jamia/article/22/3/671/776531)

+ [Deep learning for pharmacovigilance: recurrent neural network architectures for labeling adverse drug reactions in Twitter posts](https://academic.oup.com/jamia/article/24/4/813/3041102)

+ [Attending to Characters in Neural Sequence Labeling Models](https://arxiv.org/abs/1611.04361)

+ [An Attentive Sequence Model for Adverse Drug Event Extraction from Biomedical Text](https://arxiv.org/abs/1801.00625)