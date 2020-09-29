import nltk
import spacy
import json
import os
import re
from tqdm import tqdm


# hypers
dataset_names = ['kp20k', 'inspec', 'krapivin', 'semeval', 'duc', 'nus']
root_path = "/zf18/yw9fm/KPG_Project"
data_path = os.path.join(root_path,"data")
DEC_MODEL = 'decode_model_495000_1587512303'

stemmer = nltk.stem.porter.PorterStemmer()
spacy_nlp = spacy.load('en_core_web_sm')

def write_aug_test(name, test_data):
    dec_dir = os.path.join(root_path, 'log', name, DEC_MODEL)
    gen_dir = os.path.join(dec_dir,'rouge_dec_dir')
    gen = os.listdir(gen_dir)
    gen = sorted(gen)
    target_txt_path = "../data/EN/"+name+".test.onlyaug.txt"
    with open(target_txt_path,'w+') as f_in:
        for idx, data in enumerate(test_data):
            read_path = os.path.join(gen_dir,gen[idx])
            assert os.path.exists(read_path)
            with open(read_path,'r') as f_out:
                tmp_sum = f_out.readlines()
            summ = ' '.join([tmp.strip() for tmp in tmp_sum])
            f_in.write(summ + '\n')
# write datasets to AutoPhrase framework
# for name in ['kp20k']:
#     test_path = os.path.join(data_path,name,name+'_test_spacynp.json')
#     test_data = [json.loads(line) for line in open(test_path, 'r')]
#     write_aug_test(name, test_data)
#     print(name,os.path.exists(test_path),len(test_data))
def stem_process(text):
    doc = spacy_nlp(text)
    return ' '.join([stemmer.stem(w) for w in doc.text.split()])
def summary_recall(name, test_data,groundtruth,beam_size, model_name='kp20k'):
    seg_path = "../models/" + model_name +"/segmentation."+name+".onlyaug.txt"
    data_size = len(test_data)
    result = []
    assert os.path.exists(seg_path)
    with open(seg_path,'r') as f:
        lines = [line.lower().rstrip() for line in f]
    assert len(lines) == data_size
        
    for idx, data in tqdm(enumerate(test_data)):
        if len(groundtruth[idx])>=1:
            ground = [' '.join([stemmer.stem(a) for a in ls]) for ls in groundtruth[idx]]
            
            aug_k = re.findall('<phrase>(.+?)</phrase>', lines[idx])
            aug_k = list(set(aug_k))
            aug_k = set([stem_process(kw) for kw in aug_k[:beam_size]])
            tmp = 0
            for gold in ground:
                if gold in aug_k:
                    print(gold)
                    tmp += 1
            result.append(tmp*1.0/len(ground))
    with open('ap_result_150.log','w') as f:
        f.write(sum(result)*1.0/len(result))        
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
    result = summary_recall(name,test_data,absent,200)
 