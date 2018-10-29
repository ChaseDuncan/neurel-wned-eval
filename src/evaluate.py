''' For evaluating the performance of Neurel EL system
on the WNED datasets.
'''
import logging

from ccg_nlpy.core import view
from ccg_nlpy import core, local_pipeline

from os import listdir
from os.path import isfile, join
from collections import defaultdict

def bracket_map():
    brack_map = {}
    brackets = set()
    with open("data/wned-datasets/wikipedia/wikipedia-name2bracket.tsv", "r") as f:
        for line in f.readlines():
            spline = line.strip().split("\t")
            doc_id, bracket = spline[0], spline[1]
            brack_map[doc_id] = bracket
            brackets.add(bracket)
    return brack_map, brackets


def get_ta_dir(directory):
    """
        Returns a list of TextAnnotation objects which are instatiated
        using the serialized json data in the directory parameter.

        @param directory path to directory with serialized TAs
        @return tas a list of TextAnnotations
    """
    pipeline = local_pipeline.LocalPipeline()
    serialized_tas = [join(directory+"/",f) \
        for f in listdir(directory) if isfile(join(directory+"/",f))]
    tas = []

    for ser_ta in serialized_tas:
        with open(ser_ta, mode='r', encoding='utf-8') as f:
            tas.append(core.text_annotation.TextAnnotation(f.read(),pipeline))
    return tas


#EL_VIEW = "NEUREL"
#CAND_MAP_NAME="jointScoreMap"
EL_VIEW = "English_WIKIFIERVIEW"
CAND_MAP_NAME="labelScoreMap"
GOLD_VIEW = "GOLD_WIKI_LABELS"
#COH_VIEW = "COHERENCE"
COH_VIEW = "NGDVIEW"
def score(ta):
    el_view = ta.view_dictionary[EL_VIEW].as_json
    gold_view = ta.view_dictionary[GOLD_VIEW].as_json
    coh_view = ta.view_dictionary[COH_VIEW].as_json
    gold_labels = {}
    coh_labels = {}
    corr, cap_corr, num_cap, unk_wids, in_cands, \
        in_cands_cap, coh_correct = 0, 0, 0, 0, 0, 0, 0

    for constituent in gold_view['viewData'][0]['constituents']:
        gold_labels[constituent['start']]=constituent['label']
    for constituent in coh_view['viewData'][0]['constituents']:
        coh_labels[constituent['start']]=constituent['label']

    num_mentions=len(gold_labels.keys())
    for constituent in el_view['viewData'][0]['constituents']:
        label = constituent['label']
        gold_label = gold_labels[constituent['start']]  
        coh_label = coh_labels[constituent['start']]  
        if coh_label != label:
            print(coh_label, label, gold_label)
        if label == "<unk_wid>":
            unk_wids+=1
        elif label  == gold_label:
            corr+=1
        if label != "<unk_wid>":
            num_cap+=1
            if label == gold_label:
                cap_corr+=1

        cands = constituent[CAND_MAP_NAME].keys()
        for cand in cands:
            if cand == gold_label:
                in_cands+=1
                if gold_label[0].isupper():
                    in_cands_cap+=1

    coh_view = ta.view_dictionary[COH_VIEW].as_json
    for constituent in coh_view['viewData'][0]['constituents']:
        gold_label = gold_labels[constituent['start']]  

        label = constituent['label']
        if label == gold_label:
            coh_correct+=1

    return num_mentions, corr, in_cands, in_cands_cap, cap_corr, num_cap, unk_wids, \
        coh_correct

    
BRACKET_MAP, BRACKETS = bracket_map()

def avg_acc_brackets(bracket_dicts):
    acc_sum = 0.0
    for bracket in BRACKETS:
        bracket_dict = bracket_dicts[bracket]
        acc_sum+=(bracket_dict['correct_labels']/bracket_dict['total_mentions'])
    return acc_sum / len(BRACKETS)

def avg_cap_acc_brackets(bracket_dicts):
    acc_sum = 0.0
    for bracket in BRACKETS:
        bracket_dict = bracket_dicts[bracket]
        acc_sum+=(bracket_dict['correct_cap_labels']/bracket_dict['total_cap_labels'])
    return acc_sum / len(BRACKETS)

def avg_coh_acc_brackets(bracket_dicts):
    acc_sum = 0.0
    for bracket in BRACKETS:
        bracket_dict = bracket_dicts[bracket]
        acc_sum+=(bracket_dict['correct_coh']/bracket_dict['total_mentions'])
    return acc_sum / len(BRACKETS)

def evaluate(annotated_ta_dir):
    bracket_dicts = defaultdict(lambda: defaultdict(float)) 
    tas = get_ta_dir(annotated_ta_dir)

    total_mentions = 0.0
    correct_labels = 0.0
    recall = 0.0
    recall_cap = 0.0
    # Using first letter capitalized as a proxy for NE detection
    correct_cap_labels = 0.0
    correct_coh = 0.0
    total_cap_labels = 0.0
    total_unk_wids = 0.0

    for ta in tas:
        num_mentions, corr, in_cands, in_cands_cap,cap_corr,\
            num_cap, unk_wids, corr_coh = score(ta)

        bracket_dict = bracket_dicts[BRACKET_MAP[ta.id]]
        bracket_dict['total_mentions']+=num_mentions
        bracket_dict['correct_labels']+=corr
        bracket_dict['recall']+=in_cands
        bracket_dict['recall_cap']+=in_cands_cap
        bracket_dict['correct_cap_labels']+=cap_corr
        bracket_dict['total_cap_labels']+=num_cap
        bracket_dict['total_unk_wids']+=unk_wids
        bracket_dict['correct_coh']+= corr_coh
    print("average acc across brackets: %f"%(avg_acc_brackets(bracket_dicts)))
    #print("average cap acc across brackets: %f"%())
    #print("accuracy: %f"%(correct_labels/total_mentions))
    #print("recall: %f"%(recall/total_mentions))
    #print("recall cap: %f"%(recall_cap/total_cap_labels))
    #print("unk_wids: %f"%(total_unk_wids))
    #print("accuracy of known: %f"%(correct_labels/(total_mentions-total_unk_wids)))
    #print("accuracy on capital mentions: %f"%(correct_cap_labels/total_cap_labels))

    print("coherence accuracy: %f"%(avg_coh_acc_brackets(bracket_dicts)))
    

if __name__=="__main__":
    import sys
    annotated_ta_dir = sys.argv[1]
    evaluate(annotated_ta_dir)
