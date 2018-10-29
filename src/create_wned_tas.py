''' Generates a directory of JSON serialized text annotations (TA)
from the WNED-Wikipedia and WNED-Clueweb datasets. Among the standard
sentence and token views the TAs include an "NER_CONLL" view which
is simply the mentions which are provided by the dataset.
'''
DATA_DIR="../data/wned-datasets/"
#DATASETS=["clueweb12","wikipedia"] 
DATASETS=["wikipedia"] 
SPANLABELVIEW=\
'edu.illinois.cs.cogcomp.core.datastructures.textannotation.SpanLabelView'

import xml.etree.ElementTree as ET
import os
import json

from ccg_nlpy import local_pipeline
from ccg_nlpy.core import view


pipeline = local_pipeline.LocalPipeline() 

def init_view(name):
    return {'viewName':name,
            'viewData': [{'viewName':name,
                          'constituents':[],
                          'score': 1,
                          'generator':'gold_annotation',
                          'viewType':SPANLABELVIEW}]}


def init_offset_dicts(token_offsets):
    start_offsets = {}
    end_offsets = {}
    for i, token in enumerate(token_offsets):
        start_offsets[token[0]] = i
        end_offsets[token[1]] = i
    return start_offsets, end_offsets


def parse_annotation(anno):
    mention = anno[0].text
    wiki_name = anno[1].text.strip().replace(" ", "_")
    start_offset = int(anno[2].text)
    char_length = int(anno[3].text)
    end_offset = start_offset+char_length

    return(mention, wiki_name, start_offset, end_offset)

def get_end_offset(end_offsets, end_offset):
    ''' End offsets are a little more complicated since some of the mentions are stemmed
    and the given character length is according to the stemmed token and not the original.
    For instance, 'Austrian' in text will be given as the mention 'Austria' of length 7.
    This results in a key error since the token 'Austrian' in the text ends at start+8
    not start+7. In the event of a KeyError we search for the next smallest end offset.
    '''

    try:
        return end_offsets[end_offset]+1
    except KeyError:
        ''' There are more elegant ways to do this but the lists are small and my time
        is short...'''
        found_offset=False
        while not found_offset:
            end_offset+=1
            if end_offset in end_offsets.keys():
                found_offset = True
        return end_offsets[end_offset]+1
        

def get_start_offset(start_offsets, start_offset):
    ''' See note in get_end_offset. The same issue arises in the start offsets,
    e.g. 'non-Test' is given as 'Test'
    '''
    try:
        return start_offsets[start_offset]
    except KeyError:
        ''' There are more elegant ways to do this but the lists are small and my time
        is short...'''
        found_offset=False
        while not found_offset:
            start_offset-=1
            if start_offset in start_offsets.keys():
                found_offset = True
        return start_offsets[start_offset]

    
def init_constituent(mention, label, start_offset, end_offset, start_offsets, 
                    end_offsets):
    return {'tokens':mention, 'score':1.0, 'label': label,
            'start':get_start_offset(start_offsets, start_offset), 
            'end':get_end_offset(end_offsets,end_offset)}

def serialize_tas(tas, directory):
    """
        Serialize list of TextAnnotations to a given directory.

        @param directory the path to where the serialized TAs should be written.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    for ta in tas:
        filename = directory + "/" + ta.id
        print(filename)
        print(ta.id)
        with open(filename, mode="w", encoding="utf-8") as f:
            print("Writing %s to file."%filename)
            json.dump(ta.as_json, f, indent=4, ensure_ascii=False)

def generate_json(dataset, out_dir):
    dataset_dir = DATA_DIR+dataset+"/"
    tree = ET.parse(dataset_dir+dataset+".xml")
    root = tree.getroot()
    tas = []
    doc_ct = 0
    for child in root:
        doc_ct+=1
        doc_name = child.attrib['docName']
        raw_doc = dataset_dir+"RawText/"+doc_name
        gold_annotations = [] 
        for anno in child:
            gold_annotations.append(parse_annotation(anno))
        ner_view = init_view('NER')
        gold_view = init_view('GOLD_WIKI_LABELS')

        ner_constituents = []
        gold_constituents = []

        print("Working on document %s, %d / %d."%(doc_name, doc_ct, len(root)))
        with open(raw_doc, "r") as f:
            ta = pipeline.doc(f.read())
            ta.id = doc_name
            start_offsets, end_offsets = init_offset_dicts(ta.get_token_char_offsets)
            for mention, wiki_name, start_offset, end_offset in gold_annotations:
                    ner_constituents.append(init_constituent(mention, 'UNK',
                                                        start_offset, end_offset,
                                                        start_offsets, end_offsets))
                    gold_constituents.append(init_constituent(mention, wiki_name,
                                                          start_offset, end_offset,
                                                          start_offsets, end_offsets))

        ner_view['viewData'][0]['constituents'] = ner_constituents
        gold_view['viewData'][0]['constituents'] = gold_constituents

        ta.view_dictionary['NER'] = view.View(ner_view,ta.get_tokens)
        ta.view_dictionary['GOLD_WIKI_LABELS'] = view.View(gold_view,ta.get_tokens)

        tas.append(ta)
    
    serialize_tas(tas, out_dir)

if __name__=="__main__":
    for ds in DATASETS:
        generate_json(ds, "../data/wned-datasets-textannotations/"+ds)
