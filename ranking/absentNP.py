import nltk
import spacy

from spacy.symbols import NOUN, PROPN, PRON

import json
import os
from tqdm import tqdm



stemmer = nltk.stem.porter.PorterStemmer()
spacy_nlp = spacy.load('en_core_web_sm')
doc = spacy_nlp(u'keyphrase generator')

# hypers
dataset_names = ['kp20k', 'inspec', 'krapivin', 'semeval', 'duc', 'nus']
root_path = "/zf18/yw9fm/KPG_Project"
data_path = os.path.join(root_path,"data")

def summary_recall(name, test_data,groundtruth,beam_size):
    dec_dir = os.path.join(root_path, 'log', name, 'decode_model_495000_1587512303')
    gen_dir = os.path.join(dec_dir,'rouge_dec_dir')
    gen = os.listdir(gen_dir)
    gen = sorted(gen)
    result = []
    assert len(gen)==len(test_data)
    for idx, data in tqdm(enumerate(test_data)):
        if len(groundtruth[idx])>=1:
            ground = [' '.join([stemmer.stem(a) for a in ls]) for ls in groundtruth[idx]]
            read_path = os.path.join(gen_dir,gen[idx])
            assert os.path.exists(read_path)
            with open(read_path,'r') as f_out:
                tmp_sum = f_out.readlines()
            summ = [tmp.strip() for tmp in tmp_sum]
            assert len(summ)<=beam_size
            
            tmp=0
            text = ' '.join(summ)
            text = ' '.join([stemmer.stem(w) for w in text.split()])
            for gold in ground:
                if gold in text:
                    print(gold)
                    tmp += 1
            result.append(tmp*1.0/len(ground))
            
    print(sum(result)*1.0/len(result))
    return result

# for name in ['kp20k']:
#     test_path = os.path.join(data_path,name,name+'_test_spacynp.json')
#     test_data = [json.loads(line) for line in open(test_path, 'r')]
    
# #     present = [data['present_tgt_phrases'] for data in test_data]
# #     print(name,'present')
# #     summary_recall(name,test_data,present,100)
#     print('absent')
#     absent = [data['absent_tgt_phrases'] for data in test_data]
#     result = summary_recall(name,test_data,absent,100)
def get_noun_chunks(text, trim_punct=True, remove_stopword=True):
    spacy_doc = spacy_nlp(text, disable=["textcat"])
    np_chunks = list(spacy_doc.noun_chunks)
    np_str_list = []
    for chunk in np_chunks:
        np = []
        for w in chunk:
            w = w.text
            if trim_punct:
                w = w.strip(r"""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~""")
            if remove_stopword:
                if w.lower() in stopword_set:
                    continue
            np.append(w)
        if len(np) > 0:
            np_str_list.append(' '.join(np))

    return np_str_list

def noun_chunks(doc, remove_duplicate=True):
    """
    Detect base noun phrases from a dependency parse. Works on both Doc and Span.
    """
    noun_chunk_list = []
    labels = ['nsubj', 'dobj', 'nsubjpass', 'pcomp', 'pobj', 'dative', 'appos',
              'attr', 'ROOT']
    id2name = {tid: t for tid, t in enumerate(spacy.symbols.NAMES)}
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add('conj')
    np_set = set()

    for i, word in enumerate(doc):
        # print(i, word.text, id2name[word.pos], id2name[word.dep] if word.dep in id2name else 'np_dep')
        if word.pos not in (NOUN, PROPN, PRON):
            continue
        # Prevent nested chunks from being produced
        if word.dep in np_deps:
            # print(doc[word.left_edge.i: word.i+1])
            # print([id2name[t.pos] for t in doc[word.left_edge.i: word.i+1]])
            if remove_duplicate:
                for np in all_nested_NPs(doc[word.left_edge.i: word.i+1]):
                    if np.text not in np_set:
                        noun_chunk_list.append(np)
                        np_set.add(np.text)
            else:
                noun_chunk_list.extend(all_nested_NPs(doc[word.left_edge.i: word.i+1]))
        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.dep in np_deps:
                # print(doc[word.left_edge.i: word.i + 1])
                # print([id2name[t.pos] for t in doc[word.left_edge.i: word.i+1]])
                if remove_duplicate:
                    for np in all_nested_NPs(doc[word.left_edge.i: word.i+1]):
                        if np.text not in np_set:
                            noun_chunk_list.append(np)
                            np_set.add(np.text)
                else:
                    noun_chunk_list.extend(all_nested_NPs(doc[word.left_edge.i: word.i+1]))

    return noun_chunk_list
def all_nested_NPs(span):
    i = 0
    for i, word in enumerate(span):
        if word.pos != 89: # not a DET
            break

    span = span[i: ]
    nested_nps = []

    # a two-layer loop to get all possible nested phrases
    for k in range(1, len(span) + 1):
        for i in range(len(span) - k + 1):
            # print(span[i: i + k])
            np = span[i: i + k]
            nested_nps.append(np)

    return nested_nps
def stem_word_list(word_list):
    return [stemmer.stem(w.strip()) for w in word_list]

# text = "A feedback vertex set of a graph G is a set S  of its vertices such that the subgraph induced by V(G)?S is a forest. The cardinality of a minimum feedback vertex set of G  is denoted by ?(G). A graph G is 2-degenerate  if each subgraph G? of G has a vertex v  such that dG?(v)?2. In this paper, we prove that ?(G)?2n/5 for any 2-degenerate n-vertex graph G and moreover, we show that this bound is tight. As a consequence, we derive a polynomial time algorithm, which for a given 2-degenerate n-vertex graph returns its feedback vertex set of cardinality at most 2n/5."
# spacy_doc = spacy_nlp(text, disable=["textcat"])
# spacy_nps = noun_chunks(spacy_doc, remove_duplicate=True)
# nps = [[t.text.lower() for t in np] for np in spacy_nps]
# stemmed_nps = [' '.join(stem_word_list(p)) for p in nps]
# np_set = set(stemmed_nps)

# match_np = [p for p in np_set if p in present_tgts_set]
# recall = len(match_np) / len(stemmed_present_tgts) if len(stemmed_present_tgts) > 0 else -1.0
def np_summary_recall(name, test_data,groundtruth,beam_size):
    dec_dir = os.path.join(root_path, 'log', name, 'decode_model_495000_1587512303')
    gen_dir = os.path.join(dec_dir,'rouge_dec_dir')
    gen = os.listdir(gen_dir)
    gen = sorted(gen)
    result = []
    assert len(gen)==len(test_data)
    for idx, data in tqdm(enumerate(test_data)):
        if len(groundtruth[idx])>=1:
            ground = [' '.join([stemmer.stem(a) for a in ls]) for ls in groundtruth[idx]]
            read_path = os.path.join(gen_dir,gen[idx])
            assert os.path.exists(read_path)
            with open(read_path,'r') as f_out:
                tmp_sum = f_out.readlines()
            summ = [tmp.strip() for tmp in tmp_sum[:beam_size]]
            
            tmp=0
            text = ' '.join(summ)
            text = ' '.join([stemmer.stem(w) for w in text.split()])
            
            spacy_doc = spacy_nlp(text, disable=["textcat"])
            spacy_nps = noun_chunks(spacy_doc, remove_duplicate=True)
            nps = [[t.text.lower() for t in np] for np in spacy_nps]
            stemmed_nps = [' '.join(stem_word_list(p)) for p in nps]
            np_set = set(stemmed_nps)
            
            for gold in ground:
                if gold in np_set:
                    print(gold)
                    tmp += 1
            result.append(tmp*1.0/len(ground))        
    print(sum(result)*1.0/len(result))
    return result

for name in ['kp20k']:
    test_path = os.path.join(data_path,name,name+'_test_spacynp.json')
    test_data = [json.loads(line) for line in open(test_path, 'r')]
    
#     present = [data['present_tgt_phrases'] for data in test_data]
#     print(name,'present')
#     summary_recall(name,test_data,present,100)
    print('absent')
    absent = [data['absent_tgt_phrases'] for data in test_data]
    result = np_summary_recall(name,test_data,absent,100)