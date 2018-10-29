''' Implementation of realtional coherence my thesis
basically hardcoded to work on the output from Nitish's
model.
'''
from collections import defaultdict
from create_wned_tas import init_view, init_constituent, serialize_tas

from ccg_nlpy import core, local_pipeline
from ccg_nlpy.core import view

import numpy as np
import pdb
from os import listdir
from os.path import isfile, join

RELATIONS_FILE = \
    "/shared/preprocessed/cddunca2/thesis/fb15k_237_relations.tsv" 

#CAND_MAP="jointScoreMap"
CAND_MAP="labelScoreMap"

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


def init_relations_dict():
    ''' Returns a dictionary which maps a Wikipedia title t1 to a set 
    of Wikipedia titles T such that (t1, r, tk) or (tk, r, t1) for tk in T
    is a relation in the FB15k-237 dataset.
    '''
    rel_dict = defaultdict(set)
    with open(RELATIONS_FILE, "r") as f:
        for line in f.readlines():
            spline = line.strip().split("\t")
            rel_dict[spline[0]].add(spline[2])
            rel_dict[spline[2]].add(spline[0])
    return rel_dict


def init_coherence_constituent(el_con, label, score):
    return {'tokens':el_con['tokens'], 'score':score, 'label': label,
            'start':el_con['start'], 'end':el_con['end']}


def get_disambiguation_context(constituent, constituents, strategy):
    ''' The disambiguation context is the set of all titles which are not a
    candidate title for the constituent under consideration.

    Question: how does this work with mentions which corefer?
    '''

    disambiguation_context = set()
    for con in constituents:
        if strategy == "cucerzan":
            disambiguation_context.update(list(con[CAND_MAP].keys()))
        if strategy == "vinculum":
            disambiguation_context.update(con['label'])

    return disambiguation_context.difference(set(constituent[CAND_MAP].keys()))


def compute_confidence(constituent):
    ''' Compute the confidence of the label as measured by the distance
    between the top two scoring titles. If there is only one title then
    the confidence is the score of that as given by the joint.
    '''
    scores = constituent[CAND_MAP].values()
    if(len(scores) == 1):
        return list(scores)[0]
    sorted_scores = sorted(scores, reverse=True)
    return sorted_scores[0]-sorted_scores[1]



REL_DICT = init_relations_dict()

def score_cand(candidate, disambiguation_context):
    sum_of_scores = 0.0
    for disambiguation_cand in disambiguation_context:
        if disambiguation_cand in REL_DICT[candidate]:
            sum_of_scores+=1
    return sum_of_scores


def coherence_cand(constituent, disambiguation_context): 
    best_score = compute_confidence(constituent)
    best_cand = constituent['label']
    candidates = constituent[CAND_MAP].keys()
    if(len(candidates) == 1):
        return best_score, best_cand
    
    Z = len(disambiguation_context) - 1
    for candidate in candidates:
        norm_score = score_cand(candidate, disambiguation_context) / Z
        coh_score = norm_score + constituent[CAND_MAP][candidate]
        if coh_score > best_score:
            best_score = coh_score
            best_cand = candidate
     
    return best_score, best_cand


def coherence_view(view):
    #strategy = "vinculum"
    strategy = "cucerzan"
    constituents = view["viewData"][0]["constituents"]
    coherence_constituents = [] 
    for constituent in constituents:
        disambiguation_context = get_disambiguation_context(constituent, constituents,
                                                            strategy)
        score, label = coherence_cand(constituent, disambiguation_context)
        coherence_constituents.append(init_coherence_constituent(constituent,
                                                       label,score))                                                       
    coh_view = init_view("COHERENCE")
    coh_view["viewData"][0]["constituents"] = coherence_constituents
    return coh_view

EL_VIEW="NEUREL"
EL_VIEW="English_WIKIFIERVIEW"
def add_coherence_view_ta_dir(ta_dir_in, ta_dir_out=None):
    tas = get_ta_dir(ta_dir_in)
    for ta in tas:
        el_view = ta.view_dictionary[EL_VIEW].as_json     
        ta.view_dictionary['COHERENCE'] = view.View(coherence_view(el_view), ta.tokens)

    if ta_dir_out == None:
        ta_dir_out = ta_dir_in
        
    serialize_tas(tas, ta_dir_out)


if __name__=="__main__":
    import sys
    ta_dir_in = sys.argv[1]
    if len(sys.argv) > 2:
        ta_dir_out = sys.argv[2]
    add_coherence_view_ta_dir(ta_dir_in, ta_dir_out)


