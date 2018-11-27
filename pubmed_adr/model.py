from data_processing import data_processing
from nltk.tokenize import TweetTokenizer
import collections
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import KeyedVectors
from keras.layers import Dense, Input, Lambda, merge, dot, Subtract
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Activation, Reshape
from keras.models import Model
import approximateMatch
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate
import keras.backend as K
import os, re

data_path = "H:/pubmed_adr/data/ADE-Corpus-V2/DRUG-AE.rel"
final_data, idx2word, idx2label, maxlen, vocsize, nclasses, tok_senc_adr, train_lex, test_lex, train_y, test_y = data_processing(data_path)

test_toks = []
test_tok_senc_adr = tok_senc_adr[4372:]
for i in test_tok_senc_adr:
    test_toks.append(i[0])
    
train_toks = []
train_tok_senc_adr = tok_senc_adr[:4372]
for i in train_tok_senc_adr:
    train_toks.append(i[0])

# Char embedding
char_per_word = []
char_word = []
char_senc = []
maxlen_char_word = 0
a = []

for s in (train_toks + test_toks):
    for w in s:
        for c in w.lower():
            char_per_word.append(c)
            
        if len(char_per_word) > 37:
            a.append(char_per_word)
            char_per_word = char_per_word[:37]
        if len(char_per_word) > maxlen_char_word:
            maxlen_char_word = len(char_per_word)

        char_word.append(char_per_word)
        char_per_word = []
        
    char_senc.append(char_word)
    char_word = []
 

charcounts = collections.Counter()
for senc in char_senc:
    for word in senc:
        for charac in word:
            charcounts[charac] += 1
chars = [charcount[0] for charcount in charcounts.most_common()]
char2idx = {c: i+1 for i, c in enumerate(chars)}

char_word_lex = []
char_lex = []
char_word = []
for senc in char_senc:
    for word in senc:
        for charac in word:
            char_word_lex.append([char2idx[charac]])
        
        char_word.append(char_word_lex)
        char_word_lex = []
        
    char_lex.append(char_word)
    char_word = []
    
char_per_word = []  
char_per_senc = [] 
char_senc = []
for s in char_lex:
    for w in s:
        for c in w:
            for e in c:
                char_per_word.append(e)
        char_per_senc.append(char_per_word)
        char_per_word = []
    char_senc.append(char_per_senc)
    char_per_senc = []
    
pad_char_all = []
for senc in char_senc:
    while len(senc) < maxlen:
        senc.insert(0, [])
    pad_senc = pad_sequences(senc, maxlen=maxlen_char_word)
    pad_char_all.append(pad_senc)
    pad_senc = []
    
pad_char_all = np.array(pad_char_all)                       
    
pad_train_lex = pad_char_all[:4372]
pad_test_lex = pad_char_all[4372:]

idx2char  = dict((k,v) for v,k in char2idx.items())
idx2char[0] = 'PAD'
charsize =  max(idx2char.keys()) + 1

# 一些参数的定义
seed = 10
np.random.seed(seed)

glove_300d_path = 'H:/twitter_adr/embeddings/glove.840B.300d.txt'
HIDDEN_DIM = 128
NUM_EPOCHS = 10
BATCH_SIZE = 16

c2v = None
embed_dim = 300
char_embed_dim = 100

def init_embedding_weights(i2w, w2vmodel):
    # Create initial embedding weights matrix
    # Return: np.array with dim [vocabsize, embeddingsize]

    d = 300
    V = len(i2w)
    assert sorted(i2w.keys()) == list(range(V))  # verify indices are sequential

    emb = np.zeros([V,d])
    num_unknownwords = 0
    unknow_words = []
    for i,l in i2w.items():
        if i==0:
            continue
        if l in w2vmodel.vocab:
            emb[i, :] = w2vmodel[l]
        else:
            num_unknownwords += 1
            unknow_words.append(l)
            emb[i] = np.random.uniform(-1, 1, d)
    return emb, num_unknownwords, unknow_words 

def predict_score(model, x, toks, y, pred_dir, i2l, padlen, metafile=0, fileprefix=''):

    pred_probs = model.predict(x, verbose=0)
    test_loss = model.evaluate(x=x, y=y, batch_size=1, verbose=0)
    pred = np.argmax(pred_probs, axis=2)

    N = len(toks)

    # If the name of a metafile is passed, simply write this round of predictions to file
    if metafile > 0:
        meta = open(metafile, 'a')

    fname = re.sub(r'\\', r'/', os.path.join(pred_dir, fileprefix+'approxmatch_test'))
    with open(fname, 'w') as fout:
        for i in range(N):
            bos = 'BOS\tO\tO\n'
            fout.write(bos)
            if metafile > 0:
                meta.write(bos)

            sentlen = len(toks[i])
            startind = padlen - sentlen

            preds = [i2l[j] for j in pred[i][startind:]]
            actuals = [i2l[j] for j in np.argmax(y[i], axis=1)[startind:]]
            for (w, act, p) in zip(toks[i], actuals, preds):
                line = '\t'.join([w, act, p])+'\n'
                fout.write(line)
                if metafile > 0:
                    meta.write(line)

            eos = 'EOS\tO\tO\n'
            fout.write(eos)
            if metafile > 0:
                meta.write(eos)
    scores = approximateMatch.get_approx_match(fname)
    scores['loss'] = test_loss
    if metafile > 0:
        meta.close()

    with open(fname, 'a') as fout:
        fout.write('\nTEST Approximate Matching Results:\n  ADR: Precision '+ str(scores['p'])+ ' Recall ' + str(scores['r']) + ' F1 ' + str(scores['f1']))
    return scores

# 加载词向量
print('Loading word embeddings...')
w2v = KeyedVectors.load_word2vec_format(glove_300d_path, binary=False, unicode_errors='ignore')
print('word embeddings loading done!')

# Build the model
print('Building the model...')

hiddendim = HIDDEN_DIM
main_input = Input(shape=[maxlen], dtype='int32', name='input') # (None, 36)
char_input = Input(shape=[maxlen, maxlen_char_word], dtype='int32', name='char_input') # (None, 36, 25)

embeds, num_unk, unk_words  = init_embedding_weights(idx2word, w2v)

embed = Embedding(input_dim=vocsize, output_dim=embed_dim, input_length=maxlen,
                  weights=[embeds], mask_zero=False, name='embedding', trainable=False)(main_input)

embed = Dropout(0.5, name='embed_dropout')(embed)

"""
双向LSTM 获取Char embedding
"""
char_embed =  Embedding(input_dim=charsize, output_dim=char_embed_dim, embeddings_initializer='lecun_uniform',
                        input_length=maxlen_char_word, mask_zero=False, name='char_embedding')(char_input)
 
s = char_embed.shape
char_embed = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], char_embed_dim)))(char_embed)
    
fwd_state = GRU(150, return_state=True)(char_embed)[-2]
bwd_state = GRU(150, return_state=True, go_backwards=True)(char_embed)[-2]
char_embed = Concatenate(axis=-1)([fwd_state, bwd_state])
char_embed = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * 150]))(char_embed)
char_embed = Dropout(0.5, name='char_embed_dropout')(char_embed)

"""
使用attention将word和character embedding结合起来
"""
W_embed = Dense(300, name='Wembed')(embed)
W_char_embed = Dense(300, name='W_charembed')(char_embed)
merged1 = merge([W_embed, W_char_embed], name='merged1', mode='sum')
tanh = Activation('tanh', name='tanh')(merged1)
W_tanh = Dense(300, name='w_tanh')(tanh)
a = Activation('sigmoid', name='sigmoid')(W_tanh)

t = Lambda(lambda x: K.ones_like(x, dtype='float32'))(a)

merged2 = merge([a, embed], name='merged2', mode='mul')
sub = Subtract()([t, a])
merged3 = merge([sub, char_embed], name='merged3', mode='mul')
x_wave = merge([merged2, merged3], name='final_re', mode='sum')

# 辅助分类器
auxc = Dense(nclasses, name='auxiliary_classifier')(x_wave)
auxc = Activation('softmax')(auxc) # (None, 36, 5) # (None, 36, 5)

# 双向GRU
bi_gru = Bidirectional(GRU(hiddendim, return_sequences=True, name='gru'), merge_mode='concat', name='bigru')(x_wave) # (None, None, 256)
bi_gru = Dropout(0.5, name='bigru_dropout')(bi_gru)

# 主分类器
mainc = TimeDistributed(Dense(nclasses), name='main_classifier')(bi_gru)
mainc = Activation('softmax')(mainc) # (None, 36, 5)

# 将辅助分类器和主分类器相加，作为模型最终输出
final_output = merge([auxc, mainc], mode='sum')

model = Model(inputs=[main_input, char_input], outputs=final_output, name='output')
model.compile(optimizer='adam', loss='categorical_crossentropy')

print('Training...')
history = model.fit([train_lex, pad_train_lex], train_y, batch_size=BATCH_SIZE, validation_split=0.1, epochs=NUM_EPOCHS)

# 预测结果
predir = 'H:/pubmed_adr/model_output/predictions'
fileprefix = 'embedding_level_attention_'

scores = predict_score(model, [test_lex, pad_test_lex], test_toks, test_y, predir, idx2label,
                       maxlen, fileprefix=fileprefix)

"""
# OOV分析

train_tokens = []

for i in tok_senc_adr[:4372]:
    for w in i[0]:
        train_tokens.append(w)
        
test_tokens = []

for i in tok_senc_adr[4372:]:
    for w in i[0]:
        test_tokens.append(w)    

IV = []
OOTV = []
OOEV = []
OOBV = []
      
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