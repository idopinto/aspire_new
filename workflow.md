

I need to check if papers from relish exist in the training data. how much
# Connecting remotely
1. Start tmux session
```bash
tmux new -s my_session
```
2. Connect to phoenix via jump server and authenticate with 1password
```angular2html
expect ~/scripts/phoenix_ssh.exp -> alias to 'ph'
```
3. Change directory to project aspire
```angular2html
cd /cs/labs/tomhope/idopinto -> alias to 'golab'
cd aspire
```
4. Allocate resources
```angular2html
e.g
srun --mem=100gb -c4 --gres=gpu:1,vmem:45g --time=2-0 --pty $SHELL
srun --mem=30gb -c8 --time=2-0 --pty $SHELL
srun --mem=128gb -c8 --time=2-0 --pty $SHELL
srun --mem=150gb -c4 --time=2-0 --pty $SHELL

```
5. activate virutalenv
```
source /aspire_env/bin/activate
```

6. run script from terminal or pycharm (debugging also possible)
```angular2html
python3 -m  src.evaluation.evaluate --model_name aspire_biomed --dataset_name treccovid --dataset_dir datasets/eval/TRECCOVID-RF --results_dir results --log_fname logs/tc_log_file --cache
python3 -m  src.evaluation.evaluate --model_name aspire_biomed --dataset_name relish --dataset_dir datasets/eval/Relish --results_dir results --log_fname logs/relish_log_file --cache
python3 -m  src.evaluation.evaluate --model_name aspire_compsci --dataset_name csfcube --dataset_dir datasets/eval/CSFCube-1.1 --results_dir results --facet all --log_fname logs/csf_log_file --cache 
python script_name.py filt_cocit_papers --run_path /path/to/data --dataset s2orcbiomed
```

Pre-process S2ORC for training BERT-E
```
python pre_proc_gorc.py filter_by_hostserv -i <raw_meta_path> -o <filt_meta_path> -d gorcfulltext
python pre_proc_gorc.py filter_by_hostserv -i shared/0_S2ORC_metadata -o  -d gorcfulltext


python pre_proc_gorc.py gather_by_hostserv -i <in_meta_path> -o <raw_data_path> -d gorcfulltext
python pre_proc_gorc.py get_batch_pids -i <in_path> -o <out_path>
python pre_proc_gorc.py gather_from_citationnw -r <root_path> -d gorcfulltext
python pre_proc_gorc.py filter_area_citcontexts --root_path <root_path> --area compsci
python pre_proc_gorc.py gather_area_cocits --root_path <root_path> --area compsci
python pre_proc_gorc.py gather_filtcocit_corpus --root_path <root_path> --in_meta_path <in_meta_path> --raw_data_path <raw_data_path> --out_path <out_path> --dataset s2orccompsci
```

```
python3 -m src.pre_process.pre_proc_gorc gather_filtcocis_corpus --root_path <pid2batch_path> --in_meth_path <filtered_metadata.tsv> --raw_data_path <raw_data_path> --outpath <out_path> --dataset s2orcbiomed
python3 -m src.pre_process.pre_proc_cocits filt_cocit_papers --run_path /cs/labs/tomhope/idopinto12/aspire/datasets/train --dataset s2orcbiomed
python3 -m src.pre_process.pre_proc_cocits write_examples --in_path <in_path> --out_path <out_path> --model_name cosentbert --dataset s2orcbiomed --experiment sbalisentbienc
python3 -m src.learning.main_fsim train_model --model_name sbalisentbienc --dataset s2rcbiomed --num_gpus 8 --datapath <data_path> --run_path <run_path> --config_path config/models_config/s2orcbiomed/hparam_opt/sbalisentbienc-sup-best.json
python3 -m src.evaluation.evaluate --model_name


multiprocessing.pool.MaybeEncodingError: Error sending result:

Reason: 'error("'i' format requires -2147483648 <= number <= 2147483647")'
```

```angular2html
geomloss==0.2.4
joblib==1.0.1
matplotlib==3.3.4
nltk==3.6.2
numpy==1.20.3
pandas==1.3.1
POT==0.7.0
scikit_learn==1.0.1
scipy==1.6.2
seaborn==0.11.1
sentence_transformers==1.2.0
spacy==3.2.1
torch==1.8.1
transformers==4.5.1
```


    # batch_num = 1
    # metadata_fp = f'/cs/labs/tomhope/shared/0_S2ORC_metadata/metadata_{batch_num}.jsonl.gz'
    # out_fname = f'/cs/labs/tomhope/idopinto12/aspire/datasets/train/metadata_{batch_num}.tsv'
    # pdf_fp = f'/cs/labs/tomhope/shared/0_S2ORC_pdf_parses/pdf_parses_{batch_num}.jsonl.gz'
    # print(f"Number of CPU cores: {os.cpu_count()}")
    # print("Loading metadata...")
    # df = open_gzip_jsonl_file_as_pd(metadata_fp, out_fname)
    # print(f"TSV file saved to {out_fname}")
    # print(f"Batch {batch_num} shape:", df.shape)
    # print(f"Batch {batch_num} columns: {df.columns}")
    # print(f"example of row 0 :{df.iloc[0]}")
    # print("Loading pdf...")
    # pdf_df = open_gzip_jsonl_file_as_pd(pdf_fp)
    # print("Done")
    # print(f"Batch {batch_num} shape:", pdf_df.shape)
    # print(f"Batch {batch_num} columns: {pdf_df.columns}")
    # print(f"example of row 0 :{pdf_df.iloc[0]}")
    '''
    Batch metadata_0 shape: (1366661, 25)
    Batch metadata_0 columns: 
    Index(
       ['paper_id', 'title', 'authors', 'abstract', 'year', 'arxiv_id',
       'acl_id', 'pmc_id', 'pubmed_id', 'doi', 'venue', 'journal', 'mag_id',
       'mag_field_of_study', 'outbound_citations', 'inbound_citations',
       'has_outbound_citations', 'has_inbound_citations', 'has_pdf_parse',
       's2_url', 'has_pdf_body_text', 'has_pdf_parsed_abstract',
       'has_pdf_parsed_body_text', 'has_pdf_parsed_bib_entries',
       'has_pdf_parsed_ref_entries'],
      dtype='object')
      
      
    Batch 1 shape: (1365929, 25)
    Batch 1 columns: Index(['paper_id', 'title', 'authors', 'abstract', 'year', 'arxiv_id',
       'acl_id', 'pmc_id', 'pubmed_id', 'doi', 'venue', 'journal',
       'has_pdf_body_text', 'mag_id', 'mag_field_of_study',
       'outbound_citations', 'inbound_citations', 'has_outbound_citations',
       'has_inbound_citations', 'has_pdf_parse', 'has_pdf_parsed_abstract',
       'has_pdf_parsed_body_text', 'has_pdf_parsed_bib_entries',
       'has_pdf_parsed_ref_entries', 's2_url'],
      dtype='object')
      
      example of row 0 :
        paper_id                                                              77490118
        title                         LVIII Aerotitis Program in the Fifteenth Air F...
        authors                       [{'first': 'Tony', 'middle': ['J.'], 'last': '...
        abstract                                                                   None
        year                                                                     1945.0
        arxiv_id                                                                   None
        acl_id                                                                     None
        pmc_id                                                                     None
        pubmed_id                                                                  None
        doi                                                  10.1177/000348944505400408
        venue                                                                      None
        journal                              Annals of Otology, Rhinology & Laryngology
        has_pdf_body_text                                                         False
        mag_id                                                               2425011621
        mag_field_of_study                                                   [Medicine]
        outbound_citations                                                           []
        inbound_citations                                                            []
        has_outbound_citations                                                    False
        has_inbound_citations                                                     False
        has_pdf_parse                                                              True
        has_pdf_parsed_abstract                                                   False
        has_pdf_parsed_body_text                                                  False
        has_pdf_parsed_bib_entries                                                False
        has_pdf_parsed_ref_entries                                                False
        s2_url                        https://api.semanticscholar.org/CorpusID:77490118
        Name: 0, dtype: object
        
        Batch 1 shape: (310316, 6)
        Batch 1 columns: Index(['paper_id', '_pdf_hash', 'abstract', 'body_text', 'bib_entries',
               'ref_entries'],
              dtype='object')
        example of row 0 :paper_id                                       77490118
        _pdf_hash      f448565e65ce3dadb7f0622f93c0eb48a8005698
        abstract                                             []
        body_text                                            []
        bib_entries                                          {}
        ref_entries                                          {}
        
        
        Filtering 2 batches...
        Filtering ['metadata_0.jsonl.gz', 'metadata_1.jsonl.gz']
        Filtering metadata in: /cs/labs/tomhope/shared/0_S2ORC_metadata/
        Filtering by columns: None
        Reading /cs/labs/tomhope/shared/0_S2ORC_metadata/metadata_0.jsonl.gz
        df shape (1366661, 25)
        filtered shape (310736, 25)
        Reading /cs/labs/tomhope/shared/0_S2ORC_metadata/metadata_1.jsonl.gz
        df shape (1365929, 25)
        filtered shape (310316, 25)
        meta_df: 1366661; valid: (310736, 25)
        meta_df: 2732590; valid: (310316, 25)
        Total rows: 2732590; filtered rows: (621052, 25)
        Wrote: /cs/labs/tomhope/idopinto12/aspire/datasets/s2orc_filtered_metadata/metadata-s2orcfulltext.tsv
        Took: 299.8918s