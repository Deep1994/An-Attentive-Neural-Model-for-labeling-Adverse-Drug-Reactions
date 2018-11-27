import numpy as np
from gensim.models import KeyedVectors
import sys, os, re
import pickle as pkl
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Lambda, merge, dot, Subtract
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Activation, Reshape
from keras.models import Model
import approximateMatch
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate
import keras.backend as K
import collections


# 词向量和之前预处理后的数据的路径
glove_300d_path = 'H:/twitter_adr/embeddings/glove.840B.300d.txt'
datapickle_path = 'H:/twitter_adr/data/processed/adr.full.pkl'

# 随机种子
seed = 10
np.random.seed(seed)

# 加载词向量
print("Loading word embeddings...")
w2v = KeyedVectors.load_word2vec_format(glove_300d_path, binary=False, unicode_errors='ignore')
print('word embeddings loading done!')

# 加载数据函数
def load_adefull(fname):
    if not os.path.isfile(fname):
        print('Unable to find file', fname)
        return None
    with open(fname, 'rb') as f:
        train_set, valid_set, test_set, dicts = pkl.load(f)
    return train_set, valid_set, test_set, dicts

# 加载数据
train_set, valid_set, test_set, dic = load_adefull(datapickle_path) # 由于数据量较少，我们没有再对训练集划分验证集
idx2label = dict((k,v) for v,k in dic['labels2idx'].items())
idx2word  = dict((k,v) for v,k in dic['words2idx'].items())

# 0 is used as padding
if 0 in idx2label:
 sys.stderr.write('Index 0 found in labels2idx: data may be lost because 0 used as padding\n')
if 0 in idx2word:
 sys.stderr.write('Index 0 found in words2idx: data may be lost because 0 used as padding\n')
idx2word[0] = 'PAD'
idx2label[0] = 'PAD'
idx2label.pop(4) # 删掉UNK这个标签

train_toks, train_lex, train_y = train_set
test_toks, test_lex,  test_y  = test_set    
    
vocsize =  max(idx2word.keys()) + 1
nclasses = max(idx2label.keys()) + 1  

maxlen = max(max([len(l) for l in train_lex]), max([len(l) for l in test_lex]))

"""
char embedding
"""
char_per_word = []
char_word = []
char_senc = []
maxlen_char_word = 0
a = []

for s in (train_toks + test_toks):
    for w in s:
        for c in w.lower():
            char_per_word.append(c) 
        if len(char_per_word) > 25:
            a.append(char_per_word)
            char_per_word = char_per_word[:25]
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
    while len(senc) < 36:
        senc.insert(0, [])
    pad_senc = pad_sequences(senc, maxlen=maxlen_char_word)
    pad_char_all.append(pad_senc)
    pad_senc = []
    
pad_char_all = np.array(pad_char_all)                       
    
pad_train_lex = pad_char_all[:634]
pad_test_lex = pad_char_all[634:]

idx2char  = dict((k,v) for v,k in char2idx.items())
idx2char[0] = 'PAD'
charsize =  max(idx2char.keys()) + 1

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

def vectorize_set(lexlists, maxlen, V):
    nb_samples = len(lexlists)
    X = np.zeros([nb_samples, maxlen, V])
    for i, lex in enumerate(lexlists):
        for j, tok in enumerate(lex):
            X[i,j,tok] = 1
    return X

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

# Pad inputs to max sequence length and turn into one-hot vectors
train_lex = pad_sequences(train_lex, maxlen=maxlen)
test_lex = pad_sequences(test_lex, maxlen=maxlen)

train_y = pad_sequences(train_y, maxlen=maxlen)
test_y = pad_sequences(test_y, maxlen=maxlen)

train_y = vectorize_set(train_y, maxlen, nclasses)
test_y = vectorize_set(test_y, maxlen, nclasses)

# Build the model
print('Building the model...')

HIDDEN_DIM = 64
BATCH_SIZE = 1
NUM_EPOCHS = 8

hiddendim = HIDDEN_DIM

main_input = Input(shape=[maxlen], dtype='int32', name='input') # (None, 36)
char_input = Input(shape=[maxlen, maxlen_char_word], dtype='int32', name='char_input') # (None, 36, 25)

embeds, num_unk, unk_words  = init_embedding_weights(idx2word, w2v)

embed_dim = 300
char_embed_dim = 100

# 我发现把mask_zero设为False结果并没有变差，甚至有点小提升
embed = Embedding(input_dim=vocsize, output_dim=embed_dim, input_length=maxlen,
                  weights=[embeds], mask_zero=False, name='embedding', trainable=False)(main_input)
embed = Dropout(0.1, name='embed_dropout')(embed)

"""
双向LSTM 获取 Char embedding
"""
char_embed =  Embedding(input_dim=charsize, output_dim=char_embed_dim, embeddings_initializer='lecun_uniform',
                        input_length=maxlen_char_word, mask_zero=False, name='char_embedding')(char_input)
s = char_embed.shape
char_embed = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], char_embed_dim)))(char_embed)
    
fwd_state = GRU(150, return_state=True)(char_embed)[-2]
bwd_state = GRU(150, return_state=True, go_backwards=True)(char_embed)[-2]
char_embed = Concatenate(axis=-1)([fwd_state, bwd_state])
char_embed = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * 150]))(char_embed)
char_embed = Dropout(0.1, name='char_embed_dropout')(char_embed)

"""
使用attention将word embedding和character embedding结合起来
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
auxc = Activation('softmax')(auxc) # (None, 36, 5)

# 双向GRU
bi_gru = Bidirectional(GRU(hiddendim, return_sequences=True, name='gru'), merge_mode='concat', name='bigru')(x_wave) # (None, None, 128)
bi_gru = Dropout(0.1, name='bigru_dropout')(bi_gru)

# 主分类器
mainc = TimeDistributed(Dense(nclasses), name='main_classifier')(bi_gru) # (None, 36, 4)
mainc = Activation('softmax')(mainc)

# 将辅助分类器和主分类器相加，作为模型最终输出
final_output = merge([auxc, mainc], mode='sum')

model = Model(inputs=[main_input, char_input], outputs=final_output, name='output')
model.compile(optimizer='adam', loss='categorical_crossentropy')

print('Training...')
history = model.fit([train_lex, pad_train_lex], train_y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

# 预测结果
predir = 'H:/twitter_adr/model_output/predictions'
fileprefix = 'embedding_level_attention_'

scores = predict_score(model, [test_lex, pad_test_lex], test_toks, test_y, predir, idx2label,
                       maxlen, fileprefix=fileprefix)