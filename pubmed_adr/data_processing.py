def data_processing(data_path):

    from nltk.tokenize import TweetTokenizer
    import collections
    from keras.preprocessing.sequence import pad_sequences
    import numpy as np

    labelset = ['O', 'I-ADR']
    ADE_V2 = list(open(data_path, 'r'))
    
    raw_data = []
    
    for line in ADE_V2:
        line = line.strip()
        line = line.split('|')
        raw_data.append(line) # 原始数据有6821条
    
    senc_drug_ade = []
    new_sample = []
    
    haha = []
    heihei = []
    
    for sample in raw_data:
        
        if sample[5] in sample[2]:
            haha.append([sample[0], sample[3], sample[4], 'I-ADR', sample[2], sample[6], sample[7], sample[5], sample[5], sample[1]])
            if sample[2][:int(sample[7])-int(sample[6])] == sample[5]:
                heihei.append([sample[0], sample[3], sample[4], 'I-ADR', sample[2], sample[6], sample[7], sample[5], sample[5], sample[1]])
    
    b = []
    for i in haha:
        if i not in heihei:
            b.append(i)
    
    nested_idx = [1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 30, 31]  
    for idx in nested_idx:
        heihei.append(b[idx])
    
    for sample in raw_data:
        new_sample.append([sample[0], sample[3], sample[4], 'I-ADR', sample[2], sample[6], sample[7], sample[5], sample[5], sample[1]])
        senc_drug_ade.append(new_sample[0])
        new_sample = []
    
    removed_senc = []
    for sample in senc_drug_ade:
        if sample not in heihei:
            removed_senc.append(sample)
    
    new_data = []
    uni_senc = []
    for sample in removed_senc:
        if sample[9] not in uni_senc:
            uni_senc.append(sample[9])
            new_data.append(sample)
        else:
            if sample[7] != new_data[-1][7] and sample[7] not in new_data[-1][7]:
                if sample[4] not in new_data[-1][4]:
                    new_data[-1][4] += '\t' + sample[4]
                sample[4] = new_data[-1][4]
                sample[7] += ', '+ new_data[-1][7]
                new_data.append(sample)
            else:
                if sample[4] not in new_data[-1][4]:
                    new_data[-1][4] += '\t' + sample[4]
    
    pre_senc = new_data[0][9]  
    pre_drug = new_data[0][7]           
    final_data = []
    final_data.append(new_data[0])
    
    for sample in new_data[1:]:
        if sample[9] != pre_senc:
            final_data.append(sample)
            pre_senc = sample[9]
            pre_drug = sample[7]
        else:
            if len(sample[7].split()) > len(pre_drug.split()):
                final_data[-1][4] = sample[4]
                final_data.append(sample) # 处理完后剩4858条数据
        
    senc_adr = []
    tt = TweetTokenizer()
    
    for i in final_data:
        senc_adr.append([i[9], i[4].split('\t')])
    
    tok_senc_adr = []
    tok_span = []
    sub_tok_span = []
    for i in senc_adr:
        tok_text = tt.tokenize(i[0])
        tok_text = [w.lower() for w in tok_text]
        for j in i[1]:
            sub_tok_span = tt.tokenize(j)
            sub_tok_span = [w.lower() for w in sub_tok_span]
            tok_span.append(sub_tok_span) 
        tok_senc_adr.append([tok_text, tok_span])
        tok_span = []
        sub_tok_span = []
    
    all_labels = []
    
    for i in tok_senc_adr:
        labels = ['O'] * len(i[0])
        for j in i[1]:
            for k in range(len(i[0])):
                if i[0][k:k+len(j)] == j:
                    labels[k] = 'I-ADR'
                    if len(j) > 1:
                        labels[k+1:k+len(j)] = ['I-ADR'] * (len(j)-1)
        all_labels.append(labels)
    
    wordcounts = collections.Counter()

    for i in tok_senc_adr:
        for word in i[0]:
            wordcounts[word] += 1
    
    words = [wordcount[0] for wordcount in wordcounts.most_common()]
    word2idx = {w: i+1 for i, w in enumerate(words)}
    
    labelcounts = collections.Counter()
    for l in labelset:
        labelcounts[l] += 1
    
    labels = [labelcount[0] for labelcount in labelcounts.most_common()]
    label2idx = {l: i+1 for i, l in enumerate(labels)}
    
    idx2label = dict((k,v) for v,k in label2idx.items())
    idx2word  = dict((k,v) for v,k in word2idx.items())
    
    idx2label[0] = 'PAD'
    idx2word[0] = 'PAD'
    
    vec_senc_adr = []
    vec_senc = []
    vec_adr = []
    
    for i, j in zip(tok_senc_adr, all_labels):
        vec_senc_adr.append([[word2idx[word] for word in i[0]], [label2idx[l] for l in j]])
        vec_senc.append([word2idx[word] for word in i[0]])
        vec_adr.append([label2idx[l] for l in j])
            
    
    maxlen = max([len(l) for l in vec_senc]) # 93
    
    vocsize =  max(idx2word.keys()) + 1
    nclasses = max(idx2label.keys()) + 1 
    
    pad_senc = pad_sequences(vec_senc, maxlen=maxlen)
    pad_adr = pad_sequences(vec_adr, maxlen=maxlen)
    
    def vectorize_set(lexlists, maxlen, V):
        nb_samples = len(lexlists)
        X = np.zeros([nb_samples, maxlen, V])
        for i, lex in enumerate(lexlists):
            for j, tok in enumerate(lex):
                X[i,j,tok] = 1
        return X

    pad_adr = vectorize_set(pad_adr, maxlen, nclasses)
    
    train_lex = pad_senc[:4372] # 4858 * 0.9 = 4372
    test_lex = pad_senc[4372:]

    train_y = pad_adr[:4372]
    test_y = pad_adr[4372:]
            
    return final_data, idx2word, maxlen, vocsize, nclasses, tok_senc_adr, train_lex, test_lex, train_y, test_y