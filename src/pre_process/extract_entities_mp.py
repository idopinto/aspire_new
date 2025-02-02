import pandas as pd
import polars as pl
import spacy
import scispacy
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from tqdm import tqdm
import warnings
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ner_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
NER_MODELS = [
    "en_ner_bc5cdr_md",
    "en_ner_craft_md",
    "en_ner_jnlpba_md",
    "en_ner_bionlp13cg_md"
]
BATCH_SIZE = 100  # Increased for better performance
MAX_WORKERS = max(1, cpu_count() - 1)


class BiomedicalNERExtractor:
    def __init__(self, models: List[str] = NER_MODELS):
        """
        Initialize NER extractor with multiple biomedical NER models

        Args:
            models: List of spaCy biomedical NER models to use
        """
        self.models = models
        self.shared_kb = None
        self._load_shared_kb()

    def _load_shared_kb(self):
        """
        Load shared knowledge base from the first model
        """
        try:
            _, self.shared_kb = self._init_spacy_model(self.models[0])
        except Exception as e:
            logger.error(f"Error loading shared KB: {e}")
            self.shared_kb = None

    def _init_spacy_model(self, model_name: str):
        """
        Initialize a spaCy model with UMLS linker

        Args:
            model_name: Name of the NER model to use

        Returns:
            Tuple of (initialized spaCy model, knowledge base)
        """
        # Load the model
        nlp = spacy.load(model_name)

        # Add abbreviation detector
        if "abbreviation_detector" not in nlp.pipe_names:
            nlp.add_pipe("abbreviation_detector", before="ner")

        # Remove existing linker if present
        if "scispacy_linker" in nlp.pipe_names:
            nlp.remove_pipe("scispacy_linker")

        # Add linker with shared KB
        linker = nlp.add_pipe("scispacy_linker",
                              config={
                                  "resolve_abbreviations": True,
                                  "linker_name": "umls"
                              })

        return nlp, linker.kb

    def _extract_entities_batch(self,
                                abstracts: pl.DataFrame,
                                model,
                                linker_kb,
                                model_name: str) -> List[Dict[str, Any]]:
        """
        Extract entities from abstracts using a specific NER model

        Args:
            abstracts: Polars DataFrame of abstracts
            model: spaCy NER model
            linker_kb: UMLS knowledge base
            model_name: Name of the NER model

        Returns:
            List of extracted entities
        """
        extracted_entities = []

        # Iterate through abstracts
        for row in abstracts.rows(named=True):
            doc_id = row['paper_id']
            abbreviation_dict = {}
            # Process each sentence
            for sent_index, sentence in enumerate(row['abstract']):
                # Skip empty sentences
                if not sentence:
                    continue
                # Process sentence with spaCy
                try:
                    doc = model(sentence)
                except Exception as e:
                    logger.warning(f"Error processing sentence for {doc_id}: {e}")
                    continue

                # Extract abbreviations for this sentence
                abbreviation_dict = {}
                try:
                    abbreviation_dict.update({
                        abrv.text.lower(): abrv._.long_form.text.lower()
                        for abrv in doc._.abbreviations
                    })
                except Exception as e:
                    logger.warning(f"Error extracting abbreviations: {e}")

                # Process entities in the sentence
                for entity in doc.ents:
                    # Extract UMLS information
                    umls_definitions = []
                    for umls_ent in entity._.kb_ents:
                        cui = umls_ent[0]
                        try:
                            # Try to get UMLS entity definition
                            umls_entity = linker_kb.cui_to_entity[cui]
                            umls_definitions.append((cui, umls_entity.definition))
                        except KeyError:
                            # Skip if CUI not found in knowledge base
                            continue

                    # Skip if no UMLS definitions found
                    if not umls_definitions:
                        continue

                    # Resolve long form for abbreviations
                    long_form = abbreviation_dict.get(entity.text.lower(), "-")

                    # Combine CUIs and definitions
                    combined_cui = "; ".join([cui for cui, _ in umls_definitions])
                    combined_definition = " | ".join([
                        definition for _, definition in umls_definitions
                        if definition
                    ])

                    # Store extracted entity information
                    extracted_entities.append({
                        "paper_id": doc_id,
                        "entity": entity.text,
                        "long_form": long_form,
                        "label": entity.label_,
                        "sent_index": sent_index,
                        "ner_model": model_name,
                        "cuds": combined_cui,
                        "definitions": combined_definition
                    })

        return extracted_entities
    def process_model_parallel(self,
                               abstracts: pl.DataFrame,
                               model_name: str) -> pl.DataFrame:
        """
        Process abstracts for a specific NER model

        Args:
            abstracts: Polars DataFrame of abstracts
            model_name: Name of the NER model

        Returns:
            DataFrame with extracted entities
        """
        start_time = time.time()
        logger.info(f"Starting processing for model: {model_name}")

        # Initialize model and knowledge base
        model, linker_kb = self._init_spacy_model(model_name)

        # Process abstracts for this model
        model_entities = self._extract_entities_batch(
            abstracts, model, linker_kb, model_name
        )

        # Convert to Polars DataFrame
        if model_entities:
            df = pl.DataFrame(
                model_entities,
                schema={
                    "paper_id": pl.Utf8,
                    "entity": pl.Utf8,
                    "long_form": pl.Utf8,
                    "label": pl.Utf8,
                    "sent_index": pl.Int64,
                    "ner_model": pl.Utf8,
                    "cuds": pl.Utf8,
                    "definitions": pl.Utf8
                }
            )
        else:
            df = pl.DataFrame()

        # Log processing time and results
        end_time = time.time()
        logger.info(f"Model {model_name} processing complete. "
                    f"Time taken: {end_time - start_time:.2f} seconds. "
                    f"Entities extracted: {len(df)}")

        return df

    def process_abstracts_parallel(self, abstracts: pl.DataFrame) -> pl.DataFrame:
        """
        Process abstracts using multiple NER models in parallel

        Args:
            abstracts: Polars DataFrame of abstracts

        Returns:
            Processed DataFrame with extracted entities from all models
        """
        # Parallel processing using ProcessPoolExecutor
        all_entities = []

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit processing tasks for each model
            future_to_model = {
                executor.submit(self.process_model_parallel, abstracts, model_name): model_name
                for model_name in self.models
            }

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result_df = future.result()
                    all_entities.append(result_df)
                except Exception as e:
                    logger.error(f"Error processing model {model_name}: {e}")

        # Combine results from all models
        if all_entities:
            return pl.concat(all_entities)
        return pl.DataFrame()


def load_abstracts(dataset_path: Path,
                   dataset_name: str,
                   limit: Optional[int] = None) -> pl.DataFrame:
    """
    Load abstracts using Polars with optional limit

    Args:
        dataset_path: Path to dataset directory
        dataset_name: Name of the dataset
        limit: Optional limit on number of documents to load

    Returns:
        Polars DataFrame of abstracts
    """
    # Log start of loading
    logger.info(f"Loading abstracts for dataset: {dataset_name}")

    # # Read JSONL file
    # df = pl.read_json(
    #     dataset_path / f"abstracts-{dataset_name}.jsonl"
    # )
    df = pl.read_ndjson(dataset_path / f"abstracts-{dataset_name}.jsonl")  # Use read_ndjson for line-delimited JSON

    # Filter out rows without abstracts
    df = df.filter(pl.col("abstract").is_not_null())

    # Select and rename columns
    df = df.select([
        pl.col("paper_id"),
        pl.col("title"),
        pl.col("abstract")
    ])

    # Apply optional limit
    if limit:
        df = df.head(limit)

    # Log number of loaded abstracts
    logger.info(f"Loaded {len(df)} abstracts")

    return df


def main():
    """
    Main processing function for biomedical NER
    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="Optimized Biomedical Named Entity Recognition")
    parser.add_argument('--dataset_name', required=True, help='Name of dataset to extract entities')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on number of documents')
    args = parser.parse_args()

    # Overall start time
    total_start_time = time.time()

    # Dataset path configuration
    root_path = Path("/cs/labs/tomhope/idopinto12/aspire_new")
    dataset_dir = {
        "relish": "Relish",
        "treccovid": "TRECCOVID-RF"
    }

    try:
        dataset_path = root_path / "datasets" / "eval" / dataset_dir[args.dataset_name]
    except KeyError:
        raise KeyError(f"Dataset {args.dataset_name} not found.")

    # Load abstracts
    abstracts_df = load_abstracts(dataset_path, args.dataset_name, args.limit)

    # Initialize NER extractor
    ner_extractor = BiomedicalNERExtractor()

    # Process abstracts in parallel
    logger.info("Starting parallel entity extraction...")
    extracted_entities_df = ner_extractor.process_abstracts_parallel(abstracts_df)
    n_rows_before = extracted_entities_df.shape[0]
    # Drop duplicates that have the same paper_id and normalized entity_name.
    extracted_entities_df = extracted_entities_df.with_columns(
        pl.col("entity").str.to_lowercase().alias("entity_lower")
    ).unique(
        subset=["paper_id", "entity_lower"]
    ).drop("entity_lower")
    n_rows_after = extracted_entities_df.shape[0]

    # Save results
    parquet_output = dataset_path / f"{args.dataset_name}_ner.parquet"
    csv_output = dataset_path / f"{args.dataset_name}_ner.csv"

    # Save as Parquet (more efficient for large datasets)
    extracted_entities_df.write_parquet(parquet_output)
    logger.info(f"Saved {len(extracted_entities_df)} extracted entities to {parquet_output}")

    # Save as CSV (for easier human readability)
    extracted_entities_df.write_csv(csv_output)
    logger.info(f"Saved {len(extracted_entities_df)} extracted entities to {csv_output}")

    # Log total processing time
    total_end_time = time.time()
    logger.info(f"Total processing time: {total_end_time - total_start_time:.2f} seconds")

    # log stats
    unique_paper_ids_count = extracted_entities_df["paper_id"].n_unique()
    logger.info(f"Number of unique paper_ids: {unique_paper_ids_count}, {unique_paper_ids_count/len(abstracts_df)*100:.2f}% papers have entities with definitions")
    logger.info(f"Number of entities: {extracted_entities_df.shape}, {n_rows_before - n_rows_after} rows dropped. (Duplications)")
    average_single_definition_length = (
        extracted_entities_df.with_columns(
            pl.col("definitions").cast(pl.Utf8)  # Convert to string
        ).select(
            pl.col("definitions")
            .str.split_exact("|")  # Split on pipe
            .flatten()  # Explode the arrays
            .str.strip()  # Remove whitespace
            .str.lengths()  # Get lengths
            .mean()  # Calculate mean
            .alias("avg_length")
        )
    )
    logger.info(f"Average Single-Definition Length: {average_single_definition_length}")

def post_process(dataset_path: Path, dataset_name:str,  file_path: Path):
    orig_df = pl.read_ndjson(dataset_path / f"abstracts-{dataset_name}.jsonl")  # Use read_ndjson for line-delimited JSON
    print(f"{orig_df.shape} original number of documents")

    df = pl.read_csv(file_path)
    print(df.columns)
    print(f"{dataset_name} loaded.")
    n_rows_before = df.shape[0]
    unique_paper_ids_count = df["paper_id"].n_unique()
    print(f"Number of entities before: {df.shape}")
    print(f"{unique_paper_ids_count} unique papers")

    # Drop duplicates that have the same paper_id and normalized entity_name.
    df = df.with_columns(
        pl.col("entity").str.to_lowercase().alias("entity_lower")
    ).unique(
        subset=["paper_id", "entity_lower"]
    ).drop("entity_lower")
    n_rows_after = df.shape[0]
    unique_paper_ids_count = df["paper_id"].n_unique()
    print(f"Number of entities after: {df.shape}, {n_rows_before - n_rows_after} rows dropped. (Duplications)")
    print(f"{unique_paper_ids_count} unique papers")

    parquet_output = dataset_path / f"{dataset_name}_ner.parquet"
    csv_output = dataset_path / f"{dataset_name}_ner.csv"
    # Save as Parquet (more efficient for large datasets)
    df.write_parquet(parquet_output)
    logger.info(f"Saved {len(df)} extracted entities to {parquet_output}")

    # Save as CSV (for easier human readability)
    df.write_csv(csv_output)
    logger.info(f"Saved {len(df)} extracted entities to {csv_output}")
    '''
    Relish:
        Number of entities before: (5679189, 8)
        161291 unique papers.
        Number of entities after: (2553087, 8), 3126102 rows dropped. (Duplications -paper_id, entity_lower)
        1879 / 163170 = 0.0115156 = 1.15% of Relish don't have entities with definitions.
    '''

if __name__ == "__main__":
    # file_path = Path("/cs/labs/tomhope/idopinto12/aspire_new/datasets/eval/Relish/relish_ner_entities.csv")
    # post_process(dataset_path=Path("/cs/labs/tomhope/idopinto12/aspire_new/datasets/eval/Relish"), dataset_name="relish", file_path=file_path)
    main()
