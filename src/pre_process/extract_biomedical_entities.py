"""
Extracting Biomedical Entities as pre-processing stage for biomedical evaluation datasets: Relish, TRECCOVID-RF
Using SciSpacy:
en_ner_bc5cdr_md: A spaCy NER model trained on the BC5CDR corpus.
https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
"""
import argparse
from typing import List, Dict

import spacy
import codecs
import json
import os
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from tqdm import tqdm
from pathlib import Path

def load_entity_model(entity_model_name: str="en_ner_bc5cdr_md", entity_linker: bool=False, abbreviations_detector: bool=True):
    """
    :param entity_model_dir: path to dir where PURE's entity berty mode was downloaded.
    e.g. /aspire/PURE/scierc_models/ent-scib-ctx0
    :return: loaded entity model
    """
    nlp = spacy.load(entity_model_name)
    if abbreviations_detector:
        nlp.add_pipe("abbreviation_detector")
    if not entity_linker:
        return nlp
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    linker = nlp.get_pipe("scispacy_linker")
    return nlp, linker


def extract_biomed_ners(sentences: List[str], model, resolve_abbrevations=True) -> Dict[str, str]:
    """
    Extracts NER entities from sentences using the entity model provided.
    :param sentences: List[str]
    :param model: SciSpacy Biomed NER model
    :return: dict of entities and their labels.
    """
    ents2labels = dict()
    for j, sentence in enumerate(sentences):
        sent = model(sentence)
        sent_ents = sent.ents
        for ent in sent_ents:
            normalized_ent = ent.text.lower()
            if resolve_abbrevations:
                sent_abbrevs = sent._.abbreviations
                for abrev in sent_abbrevs:
                    if normalized_ent == abrev.text.lower():
                        normalized_ent = abrev._.long_form.text.lower()
            if normalized_ent in ents2labels.keys():
                if ent.label_ != ents2labels[normalized_ent]:
                    ents2labels[normalized_ent] = ent.label_
            else:
                ents2labels[normalized_ent] = ent.label_
    return ents2labels

def load_dataset(fname: str):
    """
    :param fname: filename for evaluation data
    :return: dict of {pid: data}
    """
    dataset = dict()
    with codecs.open(fname, 'r', 'utf-8') as f:
        for jsonline in tqdm(f):
            data = json.loads(jsonline.strip())
            pid = data['paper_id']
            ret_dict = {
                'TITLE': data['title'],
                'ABSTRACT': data['abstract'],
            }
            dataset[pid] = ret_dict
    return dataset


def main(dataset_dir, dataset_name, output_dir):
    """
    :param dataset_dir: Data path where biomedical evaluation datasets are located.
    :param dataset_name: name of dataset (relish or treccovid)
    :return:
    """

    # load entity model and dataset
    print("Loading model and dataset")
    model = load_entity_model()
    dataset = load_dataset(os.path.join(dataset_dir, f'abstracts-{dataset_name}.jsonl'))

    # find entities for each paper
    print("Extracting entities from abstracts")
    entities = dict()
    for (doc_id, doc) in tqdm(list(dataset.items())):
        doc_entities = extract_biomed_ners(doc['ABSTRACT'], model)
        entities[doc_id] = doc_entities

    # save results
    output_filename = os.path.join(output_dir, f'{dataset_name}-ner.jsonl')
    print(f"Writing output to: {output_filename}")
    with codecs.open(output_filename, 'w', 'utf-8') as fp:
        json.dump(entities, fp)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, help='Name of dataset to extract entities on')
    parser.add_argument('--dataset_dir', required=True, help='Dataset dir. abstracts-{dataset_name}.jsonl should be inside')
    parser.add_argument('--output_dir', required=True, help='Output directory where extracted entities should be written')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(dataset_dir=args.dataset_dir, dataset_name=args.dataset_name, output_dir=args.output_dir)

''''


papers = [{"title": "MicroRNAs in autoimmune disease.",
         "abstract": ["MicroRNAs (miRNAs) are non-coding, single-stranded small RNAs, usually 18-25 nucleotides long, have ability to regulate gene expression post-transcriptionally.",
                      "miRNAs are highly homologous, conserved and are found in various living organisms including plants and animals.",
                      "Present studies show that these small RNAs anticipate and are directly involved in many important physiological and pathological processes including growth, proliferation, maturation, metabolism, and inflammation among others.",
                      "Evidences are accumulating that miRNAs play active role in directing immune responses and, therefore, might be involved in pathogenesis of autoimmune diseases.",
                      "Recent studies have found that miRNAs are critical in proliferation, maturation and differentiation of T cells, B cells and, therefore, may affect the outcome of an immune response.",
                      "In light of such understanding, this review briefly introduces miRNAs and discusses its role in the pathogenesis of various autoimmune diseases, as well as its potential as a biomarker and therapeutic target in the management of autoimmune diseases."],
            "paper_id": "26000120"},
          # {"title": "A Novel Model for Evaluating the Flow of Endodontic Materials Using Micro-computed Tomography.",
          #  "abstract": ["INTRODUCTION: Flow and filling ability are important properties of endodontic materials.",
          #               "The aim of this study was to propose a new technique for evaluating flow using micro-computed tomographic (\u03bcCT) imaging.",
          #               "METHODS: A glass plate was manufactured with a central cavity and 4 grooves extending out horizontally and vertically.",
          #               "The flow of MTA-Angelus (Angelus, Londrina, PR, Brazil), zinc oxide eugenol (ZOE), and Biodentine (BIO) (Septodont, Saint Maur des Foss\u00e9s, France) was evaluated using International Standards Organization (ISO) 6876/2002 and a new technique as follows: 0.05\u00a0\u00b1\u00a00.005\u00a0mL of each material was placed in the central cavity, and another glass plate and metal weight with a total mass of 120\u00a0g were placed over the material.",
          #               "The plate/material set was scanned using \u03bcCT imaging.",
          #               "The flow was calculated by linear measurement (mm) of the material in the grooves.",
          #               "Central cavity filling was calculated in mm(3) in the central cavity.", "Lateral cavity filling (LCF) was measured by LCF mean values up to 2\u00a0mm from the central cavity.",
          #               "Data were analyzed statistically using analysis of variance and Tukey tests with a 5% significance level.",
          #               "RESULTS: ZOE showed the highest flow rate determined by ISO methodology (P\u00a0<\u00a0.05).",
          #               "Analysis performed using \u03bcCT imaging showed MTA-Angelus and ZOE had higher linear flow rates in the grooves.",
          #               "Central cavity filling was similar for the materials.", "However, LCF was higher for BIO versus ZOE.",
          #               "CONCLUSIONS: Although ZOE presented better flow determined by ISO methodology, BIO showed the best filling ability.",
          #               "The model of the technique proposed for evaluating flow using \u03bcCT imaging showed proper and reproducible results and could improve flow analysis."],
          #  "paper_id": "28268019"}
          ]
          
          
          

for i, paper in enumerate(papers):
    print(f"Paper {i}, title: {paper['title']}")
    for j, sentence in enumerate(paper['abstract']):
        doc = nlp(sentence)
        if len(doc.ents) > 0:
            print(f"{j}. count: {len(doc.ents)}")
            for ent in doc.ents:
                print(f"{ent.text} \t ({ent.start_char}, {ent.end_char}) \t {ent.label_}\n")
                for umls_ent in ent._.kb_ents:
                    print(linker.kb.cui_to_entity[umls_ent[0]])
            print("------------")
            print("Abbreviation", "\t","Span","\t", "Definition")
            for abrv in doc._.abbreviations:
                print(f"{abrv} \t\t ({abrv.start}, {abrv.end}) \t\t {abrv._.long_form}")
    print("#########################################################################")
'''