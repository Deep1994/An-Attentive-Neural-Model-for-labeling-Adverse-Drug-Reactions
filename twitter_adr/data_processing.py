import os
import re
import csv
from nltk.tokenize import TweetTokenizer
import sys
import numpy as np
import pickle as pkl

# PICKLEFILE:我们会将处理后的数据存储成.pkl格式的文件
# SEQLAB_DATA_DIR:存放数据的文件夹
# LABELSET:标签集，我们用不到Beneficial部分
SETTINGS = {
    'PICKLEFILE': 'H:/twitter_adr/data/processed/ade.full.pkl',
    'SEQLAB_DATA_DIR': 'H:/twitter_adr/data',

    ## The B-I-O labeling scheme
    'LABELSET': {'ADR': {'b': 'I-ADR', 'i': 'I-ADR'},
                 'Indication': {'b': 'I-Indication', 'i': 'I-Indication'},
                 'Beneficial': {'b': 'I-Indication', 'i': 'I-Indication'}}
}

Processed_data_dir = SETTINGS['SEQLAB_DATA_DIR']+'/processed'
Raw_data_dir = SETTINGS['SEQLAB_DATA_DIR']+'/raw'
Out_file = 'adr.full.pkl'

labelset = SETTINGS['LABELSET']
tokset = {'<UNK>'}
raw_headers = ['id', 'start', 'end', 'semantic_type', 'span', 'reldrug', 'tgtdrug', 'text']

# 创建用于存放处理后数据的文件夹
if not os.path.isdir(Processed_data_dir):
    os.makedirs(Processed_data_dir)

trainfiles = 'asu,chop'.split(',')
testfiles = 'asu,chop'.split(',')

# 处理标签的函数
def comp_labels(l1,l2):
    if l1 != 'O':
        return l1
    elif l2 != 'O':
        return l2
    else:
        return 'O'

def combine_labels(seq_old, seq_new):
    if len(seq_old) == 0:
        return seq_new
    seq_combined = []
    for (o,n) in zip(seq_old, seq_new):
        seq_combined.append(comp_labels(o,n))
    return seq_combined

# 替换网址和图片链接，并全部小写处理
def clean_str(string):
    
    string = re.sub(r'http\S+', '<URL>', string)
    string = re.sub(r'pic.twitter\S+', '<PIC>', string)
    
    return string.strip().lower()

# 预处理的主函数，替换@某某某，分词，对句子中的每个词分配对应标签等
def create_adr_dataset(t, files, tokset, labelset):
    
    tokset |= {'<UNK>'}
    atmention = re.compile('@\w+')
    tt = TweetTokenizer()
    
    try:
        os.makedirs(os.path.join(Processed_data_dir, 'train'))
        os.makedirs(os.path.join(Processed_data_dir, 'test'))
    except:
        pass
    
    for f in ['_'.join([d,t]) for d in files]:    
        processed_rows = {}
        fout = open(re.sub(r'\\', r'/', os.path.join(Processed_data_dir, t, f)), 'w', newline='')
        fnames = raw_headers+['tokens','labels','norm_text']
        wrt = csv.DictWriter(fout, fieldnames=fnames)
        wrt.writeheader() 
        fname = re.sub(r"\\", r"/", os.path.join(Raw_data_dir, t, f))
        with open(fname, 'r', errors='ignore') as fin:
            dr = csv.DictReader(fin)
            for row in dr:
                # Pull from processed_rows dir so we can combine multiple annotations in a single tweet
                pr = processed_rows.get(row['id'], {h: row.get(h,[]) for h in fnames})
                
                text = row['text']
                span = row['span']
                
                text = clean_str(text)
                span = clean_str(span)
                
                # Tokenize
                tok_text = tt.tokenize(text)
                tok_span = tt.tokenize(span)
                
                # Add sequence labels to raw data
                labels = ['O'] * len(tok_text)
                if len(row['span']) > 0 and row['semantic_type'] != 'NEG':
                    s = row['semantic_type']
                    for i in range(len(tok_text)):
                        if tok_text[i:i+len(tok_span)] == tok_span:
                            labels[i] = labelset[s]['b']
                            if len(tok_span) > 1:
                                labels[i+1:i+len(tok_span)] = [labelset[s]['i']] * (len(tok_span)-1)

                # Combine spans and labels if duplicate
                pr['labels'] = combine_labels(pr['labels'], labels)
                if pr['span'] != row['span']:
                    pr['span'] = '|'.join([pr['span'], row['span']])
                pr['tokens'] = tok_text

                # Normalize text
                tok_text = [ttw if not atmention.match(ttw) else '<USER>' for ttw in tok_text]  # normalize @user
                lower_text = [w.lower() for w in tok_text]
                pr['norm_text'] = lower_text
                tokset |= set(lower_text)
                processed_rows[row['id']] = pr
        for pr, dct in processed_rows.items():
            wrt.writerow(dct)
        fout.close()
    return tokset

tokset |= create_adr_dataset('train', trainfiles, tokset, labelset)
tokset |= create_adr_dataset('test', testfiles, tokset, labelset)

def flatten(l):
    return [item for sublist in l for item in sublist]

# Build index dictionaries
labels = ['O'] + sorted(list(set(flatten([subdict.values() for subdict in labelset.values()])))) + ['<UNK>']
labels2idx = dict(zip(labels, range(1,len(labels)+1)))
tok2idx = dict(zip(tokset, range(1,len(tokset)+1)))  # leave 0 for padding

train_toks_raw = []
train_lex_raw = []
train_y_raw = []
valid_toks_raw = []
valid_lex_raw = []
valid_y_raw = []
t_toks = []
t_lex = []
t_y = []
t_class = []

def parselist(strlist):
    '''
    Parse list from string representation of list
    :param strlist: string
    :return:list
    '''
    return [w[1:-1] for w in strlist[1:-1].split(', ')]

for dtype in trainfiles:
    with open(re.sub(r"\\", r"/", os.path.join(Processed_data_dir, 'train', dtype+'_train')), 'r') as fin:
        rd = csv.DictReader(fin)
        for row in rd:
            t_toks.append(parselist(row['tokens']))
            t_lex.append(parselist(row['norm_text']))
            t_y.append(parselist(row['labels']))
            if '<UNK>' in parselist(row['labels']):
                sys.stderr.write('<UNK> found in labels for tweet %s' % row['tokens'])
            t_class.append(row['semantic_type'])


def vectorize(listoftoklists, idxdict):
    '''
    Turn each list of tokens or labels in listoftoklists to an equivalent list of indices
    :param listoftoklists: list of lists
    :param idxdict: {tok->int}
    :return: list of np.array
    '''
    res = []
    for toklist in listoftoklists:
        res.append(np.array(list(map(lambda x: idxdict.get(x, idxdict['<UNK>']), toklist))).astype('int32'))
    return res

def load_adefull(fname):
    if not os.path.isfile(fname):
        print('Unable to find file', fname)
        return None
    with open(fname, 'rb') as f:
        train_set, valid_set, test_set, dicts = pkl.load(f)
    return train_set, valid_set, test_set, dicts

train_toks_raw = t_toks
train_lex_raw = t_lex
train_y_raw = t_y
valid_toks_raw = []
valid_lex_raw = []
valid_y_raw = []


test_toks_raw = []
test_lex_raw = []
test_y_raw = []
for dtype in testfiles:
    with open(os.path.join(Processed_data_dir, 'test', dtype+'_test'), 'r') as fin:
        rd = csv.DictReader(fin)
        for row in rd:
            test_toks_raw.append(parselist(row['tokens']))
            test_lex_raw.append(parselist(row['norm_text']))
            test_y_raw.append(parselist(row['labels']))
# Convert each sentence of normalized tokens and labels into arrays of indices
train_lex = vectorize(train_lex_raw, tok2idx)
train_y = vectorize(train_y_raw, labels2idx)
valid_lex = vectorize(valid_lex_raw, tok2idx)
valid_y = vectorize(valid_y_raw, labels2idx)
test_lex = vectorize(test_lex_raw, tok2idx)
test_y = vectorize(test_y_raw, labels2idx)

# Pickle the resulting data set
with open(os.path.join(Processed_data_dir, Out_file),'wb') as fout:
    pkl.dump([[train_toks_raw,train_lex,train_y],[valid_toks_raw,valid_lex,valid_y],[test_toks_raw,test_lex,test_y],
              {'labels2idx':labels2idx, 'words2idx':tok2idx}], fout)

"""
# OOV分析

from gensim.models import KeyedVectors

glove_300d_path = 'H:/twitter_adr/embeddings/glove.840B.300d.txt'
print("Loading embeddings...")
w2v = KeyedVectors.load_word2vec_format(glove_300d_path, binary=False, unicode_errors='ignore')    

IV = []
OOTV = []
OOEV = []
OOBV = []

train_tokens = []

for train_senc in train_toks_raw:
    for train_token in train_senc:
        train_tokens.append(train_token)

unique_train_tokens = list(set(train_tokens))

test_tokens = []

for test_senc in test_toks_raw:
    for test_token in test_senc:
        test_tokens.append(test_token)

unique_test_tokens = list(set(test_tokens))

all_tokens = list(set(unique_train_tokens + unique_test_tokens))

for i in test_tokens:
    if i in train_tokens and i in w2v:
        IV.append(i)
    
    if i in w2v and i not in train_tokens:
        OOTV.append(i)
        
    if i in train_tokens and i not in w2v:
        OOEV.append(i)

    if i not in train_tokens and i not in w2v:
        OOBV.append(i)

# result_file是你预测后的文件的路径，注意需要删掉预测文件底部的F1等指标的显示信息
result_file = list(open('your prediction path', 'r'))

# 输入一个文件地址用来写入OOV分析结果
with open('your new file path', 'w') as fout:
    bos = 'BOS\tO\tO\n'
    eos = 'EOS\tO\tO\n'
    
    for line in bgru_attention:
        line = line.strip()
        line = line.split("\t")
        if line[0] == 'BOS':
            fout.write(bos)
        elif line[0] == 'EOS':
            fout.write(eos)
        # 每次注释掉其余三行来获取对应的未注释的那行的OOV分析结果
        elif line[0] in IV and (line[1] == 'I-ADR' or line[2] == 'I-ADR'):
#        elif line[0] in OOTV and (line[1] == 'I-ADR' or line[2] == 'I-ADR'):
#        elif line[0] in OOEV and (line[1] == 'I-ADR' or line[2] == 'I-ADR'):
#        elif line[0] in OOBV and (line[1] == 'I-ADR' or line[2] == 'I-ADR'):
            fout.write('\t'.join([line[0], line[1], line[2]])+'\n')
        else:
            fout.write('\t'.join([line[0], 'O', 'O'])+'\n')
 
# 调用approximateMatch的get_approx_match方法，计算OOV分析结果        
import approximateMatch  
scores = approximateMatch.get_approx_match('your new file path')
"""