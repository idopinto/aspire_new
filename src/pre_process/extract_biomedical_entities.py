"""
Extracting Biomedical Entities as pre-processing stage for biomedical evaluation datasets: Relish, TRECCOVID-RF
Using SciSpacy:
en_ner_bc5cdr_md: A spaCy NER model trained on the BC5CDR corpus.
Entity Types: ['DISEASES', 'CHEMICALS']
https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

en_ner_craft_md: A spaCy NER model trained on the CRAFT corpus.
Entity Types: ['CHEBI', 'CL', 'GGP', 'GO', 'SO', 'TAXON']
https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_craft_md-0.5.4.tar.gz

en_ner_jnlpba_md: A spaCy NER model trained on the JNLPBA corpus.
Entity Types: ['CELL_LINE', 'CELL_TYPE', 'DNA', 'PROTEIN', 'RNA']
https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz

en_ner_bionlp13cg_md: A spaCy NER model trained on the BIONLP13CG corpus.
Entity Types: ['AMINO_ACID','ANATOMICAL_SYSTEM', 'CANCER','CELL','CELLULAR_COMPONENT,'DEVELOPING_ANATOMICAL_STRUCTURE,'GENE_OR_GENE_PRODUCT'
                ,'IMMATERIAL_ANATOMICAL_STRUCTURE, 'MULTI_TISSUE_STRUCTURE,'ORGAN','ORGANISM','ORGANISM_SUBDIVISION,'ORGANISM_SUBSTANCE','PATHOLOGICAL_FORMATION','SIMPLE_CHEMICAL','TISSUE']
https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
"""
import argparse
from typing import List, Dict

import spacy
import codecs
import json
import os

from pandas import DataFrame
from spacy import displacy

from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

NER_MODELS =[
             "en_ner_bc5cdr_md",
             "en_ner_craft_md",
             "en_ner_jnlpba_md",
             "en_ner_bionlp13cg_md",
             # "en_core_sci_scibert"
             ]

def get_row(model, document, doc_abbreviation_dict, paper_id, sent_i):
    """
    Function to return extracted entities along with their UMLS definitions.
    Args:
        model: SciSpaCy model with a UMLS linker
        document: Text to analyze
        doc_abbreviation_dict: Dictionary to track abbreviations
        paper_id: Identifier for the paper
        sent_i: Sentence index
    Returns:
        doc: Processed spaCy document
        row: Set of unique entities with their labels and UMLS definitions
        doc_abbreviation_dict: Updated abbreviation dictionary
    """
    doc = model(document)
    linker = model.get_pipe("scispacy_linker")
    model_name = model.meta['name']
    # Create a dictionary of detected abbreviations
    doc_abbreviation_dict.update({abrv.text.lower(): abrv._.long_form.text.lower() for abrv in doc._.abbreviations})

    expanded_entities = []
    for entity in doc.ents:
        umls_definitions = []
        # Retrieve UMLS definitions
        for umls_ent in entity._.kb_ents:
            cui = umls_ent[0]
            umls_entity = linker.kb.cui_to_entity[cui]
            umls_definitions.append((cui, umls_entity.definition))
        if not umls_definitions:
            continue
        if entity.text.lower() in doc_abbreviation_dict:
            long_form = doc_abbreviation_dict[entity.text.lower()]
        else:
            long_form = "-"
        cui_list = [cui for cui, _ in umls_definitions]
        definition_list = [definition for _, definition in umls_definitions if definition]

        if definition_list:
            combined_cui = "; ".join(cui_list)
            combined_definition = " | ".join(definition_list)  # Separate multiple definitions with "|"
            expanded_entities.append([entity.text, long_form,entity.label_, combined_cui, combined_definition])

    row = set((paper_id, entity_text,long_form,entity_label, sent_i, model_name, cuds,definitions) for entity_text,long_form, entity_label, cuds, definitions in expanded_entities)

    return doc, row, doc_abbreviation_dict


def load_entity_model(entity_model_name: str="en_ner_bc5cdr_md", resolve_abbreviations:bool=False):
    """
    :param resolve_abbreviations:
    :param entity_model_name: name of spaCy or SciSpaCy model
    :return: loaded entity model
    """
    nlp = spacy.load(entity_model_name)
    if resolve_abbreviations:
        nlp.add_pipe("abbreviation_detector", before="ner")  # Ensure it's before NER
    return nlp


def extract_biomed_ners(paper_id, sentences: List[str], model, entities_df: pd.DataFrame) -> DataFrame:
    """
    Extracts NER entities from sentences using the entity model provided.
    :param paper_id:
    :param resolve_abbrevations:
    :param entities_df:
    :param sentences: List[str]
    :param model: SciSpacy Biomed NER model
    :return: dict of entities and their labels.
    """
    doc_abbreviation_dict = {}
    for j, sentence in enumerate(sentences):
        sent, row_to_append, doc_abbreviation_dict = get_row(model, sentence, doc_abbreviation_dict, paper_id, j)
        if len(row_to_append) > 0:
            to_append = pd.DataFrame(row_to_append, columns=["paper_id","entity","long_form", "label", "sent_index",'ner_model','cuds','definitions'])
            entities_df["entity_lower"] = entities_df["entity"].str.lower()
            entities_df = pd.concat([entities_df, to_append], ignore_index=True).drop_duplicates(
                subset=["paper_id", "entity_lower"]
            ).drop(columns=["entity_lower"])

    return entities_df

def load_dataset(fname: str):
    """
    :param fname: filename for evaluation data
    :return: dict of {pid: data}
    """
    dataset = dict()
    # i = 0
    with codecs.open(fname, 'r', 'utf-8') as f:
        for jsonline in tqdm(f):
            # if i >= 100:
            #     break
            data = json.loads(jsonline.strip())
            pid = data['paper_id']
            ret_dict = {
                'TITLE': data['title'],
                'ABSTRACT': data['abstract'],
            }
            dataset[pid] = ret_dict
            # i += 1

    print(len(dataset))
    return dataset


def main(dataset_dir, dataset_name, output_dir):
    """
    :param output_dir: Where to save the NER output file
    :param dataset_dir: Data path where biomedical evaluation datasets are located.
    :param dataset_name: name of dataset (relish or treccovid)
    :return:
    """
    # load entity model and dataset
    dataset = load_dataset(os.path.join(dataset_dir, f'abstracts-{dataset_name}.jsonl'))
    print(f"{dataset_name} Loaded.")
    entities_df = pd.DataFrame(columns=["paper_id","entity",'long_form', "label", "sent_index",'ner_model','cuds','definitions'])
    for model_name in NER_MODELS:
        model = load_entity_model(entity_model_name=model_name, resolve_abbreviations=True)
        model.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        print(f"{model_name} Loaded.")
        # find entities for each paper
        print("Extracting entities from abstracts")
        for (doc_id, doc) in tqdm(list(dataset.items())):
            entities_df = extract_biomed_ners(doc_id, doc['ABSTRACT'], model,entities_df)

    output_filename = os.path.join(output_dir, f'{dataset_name}-ner.csv')
    entities_df.to_csv(output_filename, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, help='Name of dataset to extract entities on')
    parser.add_argument('--dataset_dir', required=True, help='Dataset dir. abstracts-{dataset_name}.jsonl should be inside')
    parser.add_argument('--output_dir', required=True, help='Output directory where extracted entities should be written')
    return parser.parse_args()

def analayze():
    file_path = "/cs/labs/tomhope/idopinto12/aspire_new/datasets/eval/TRECCOVID-RF/treccovid-ner.csv"
    df = pd.read_csv(file_path)
    # Load the SPECTER tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

    # Count unique paper_id values
    unique_paper_ids_count = df["paper_id"].nunique()

    print(unique_paper_ids_count)

    # Calculate the average length of a single definition across all definitions
    df["definitions"] = df["definitions"].astype(str)
    all_definitions = df["definitions"].dropna().str.split(r"\s*\|\s*").explode()
    average_single_definition_length = all_definitions.str.len().mean()
    token_counts = all_definitions.apply(lambda x: len(tokenizer.tokenize(x)))
    average_tokens_per_definition = token_counts.mean()
    print(average_single_definition_length, average_tokens_per_definition)


if __name__ == '__main__':
    args = parse_args()
    analayze()


    # main(dataset_dir=args.dataset_dir, dataset_name=args.dataset_name, output_dir=args.output_dir)


# {'Shape': (89126, 8),
#  'Columns': ['paper_id',
#   'entity',
#   'long_form',
#   'label',
#   'sent_index',
#   'ner_model',
#   'cuds',
#   'definitions']}