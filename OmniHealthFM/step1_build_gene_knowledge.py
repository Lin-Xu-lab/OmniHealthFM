"""
step1_build_gene_knowledge.py

What this script does
---------------------
This script builds gene-level text knowledge by combining:
1) GO annotations from a GOA Human GAF file (goa_human.gaf.gz)
2) protein/gene interaction partners queried from STRING (online API)
and exports multiple JSONL files for downstream usage (e.g., LLaMA fine-tuning).

Inputs (user must provide)
--------------------------
- go.obo
  Download: http://purl.obolibrary.org/obo/go.obo
- goa_human.gaf.gz
  Download: http://current.geneontology.org/products/pages/downloads.html

Outputs
-------
- gene_string.jsonl
- gene_string_GO.jsonl
- gene_string_GO_modeling.jsonl

Reproducibility notes
---------------------
- Gene Ontology (GO) files are external resources; users should download them separately and record the file versions and download dates.
- STRING interaction data are retrieved via the STRING API and therefore depend on network availability and the STRING database state at runtime.
- For full reproducibility, we recommend storing the exact input files and the generated JSONL outputs together with the Zenodo release.


Example
-------
python step1_build_gene_knowledge.py \
  --go-obo ./go.obo \
  --go-gaf ./goa_human.gaf.gz \
  --outdir ./out

Dependencies
------------
pandas, goatools, requests, tqdm

"""

import os
import sys
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Build gene text knowledge from GO + STRING.")
    p.add_argument("--go-obo", default="go.obo", help="Path to go.obo")
    p.add_argument("--go-gaf", default="goa_human.gaf.gz", help="Path to goa_human.gaf.gz")
    p.add_argument("--outdir", default=".", help="Output directory (default: current directory)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.go_obo):
        raise FileNotFoundError(
            f"Cannot find GO OBO file: {args.go_obo}\n"
            f"Download from: http://purl.obolibrary.org/obo/go.obo"
        )
    if not os.path.exists(args.go_gaf):
        raise FileNotFoundError(
            f"Cannot find GO GAF file: {args.go_gaf}\n"
            f"Download from: http://current.geneontology.org/products/pages/downloads.html"
        )

    os.makedirs(args.outdir, exist_ok=True)
    os.chdir(args.outdir)

    print("=== Running gene GO + STRING pipeline ===")
    print(f"GO OBO: {os.path.abspath(args.go_obo)}")
    print(f"GO GAF: {os.path.abspath(args.go_gaf)}")
    print(f"OUTDIR: {os.path.abspath(args.outdir)}")
    print("========================================")

 

    import pandas as pd
    from goatools.obo_parser import GODag
    from collections import defaultdict
    import gzip
    import requests
    import json
    from tqdm import tqdm

    # I downloaded from http://purl.obolibrary.org/obo/go.obo
    GO_OBO_FILE = args.go_obo
    # I downloaded from http://current.geneontology.org/products/pages/downloads.html
    GO_GAF_FILE = args.go_gaf

    go_dag = GODag(GO_OBO_FILE)
    gene_go = defaultdict(set)

    with gzip.open(GO_GAF_FILE, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split('\t')
            db_object_symbol = parts[2]
            go_term = parts[4]
            evidence = parts[6]
            if evidence != "IEA":  # Filter weak evidence
                gene_go[db_object_symbol].add(go_term)

    genelist = []
    for gene, go_terms in gene_go.items():
        texts = []
        genelist.append(gene)

    def get_string_interactions(gene, species=9606, limit=5):
        url = "https://string-db.org/api/json/network"
        params = {
            "identifiers": gene,
            "species": species,
            "limit": limit,
            "caller_identity": "gene-rag-demo"
        }
        try:
            r = requests.get(url, params=params)
            if r.status_code != 200:
                return []
            data = r.json()
            partners = set()
            for entry in data:
                if entry["preferredName_A"] == gene:
                    partners.add(entry["preferredName_B"])
                elif entry["preferredName_B"] == gene:
                    partners.add(entry["preferredName_A"])
            return list(partners)
        except:
            return []

    output_file = "gene_string.jsonl"
    gene_set = set(genelist)
    with open(output_file, "w") as fout:
        for gene in tqdm(gene_set):
            partners = get_string_interactions(gene)
            text = f"{gene} interacts with {', '.join(partners)}." if partners else f"{gene} has no known interactions."
            entry = {"gene": gene, "text": text}
            fout.write(json.dumps(entry) + "\n")

    go_dag = GODag(GO_OBO_FILE)

    gene2go = defaultdict(list)
    with gzip.open("goa_human.gaf.gz", "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) > 4:
                gene_symbol = parts[2]
                go_id = parts[4]
                gene2go[gene_symbol].append(go_id)

    with open("gene_string.jsonl", "r") as f:
        gene_entries = [json.loads(line) for line in f]

    enriched_entries = []
    for entry in gene_entries:
        gene = entry["gene"]
        text = entry["text"]
        go_ids = gene2go.get(gene.upper(), [])

        go_terms = []
        for go_id in go_ids:
            term = go_dag.get(go_id)
            if term and term.name:
                go_terms.append(term.name)

        if go_terms:
            go_text = " Associated GO terms: " + ", ".join(sorted(set(go_terms))) + "."
            text += go_text

        enriched_entries.append({"gene": gene, "text": text})

    with open("gene_string_GO.jsonl", "w") as f:
        for entry in enriched_entries:
            f.write(json.dumps(entry) + "\n")

    go_dag = GODag(GO_OBO_FILE)

    with open("gene_string_GO.jsonl", "r") as f:
        enriched_entries = [json.loads(line) for line in f]

    train_data = []

    for entry in enriched_entries:
        gene = entry["gene"]
        text = entry["text"]

        go_terms = []
        for go_id in entry.get("go_terms", []):
            term = go_dag.get(go_id)
            if term and term.name:
                go_terms.append(term.name)

        go_text = " Associated GO terms: " + ", ".join(sorted(set(go_terms))) + "."
        text += go_text

        train_data.append({"text": f"gene: {gene} {text}"})

    # final jsonl file for Llama fine tuning
    with open("gene_string_GO_modeling.jsonl", "w") as f:
        for example in train_data:
            f.write(json.dumps(example) + "\n")


    print("\n=== Done ===")
    print("Generated files in:", os.path.abspath(args.outdir))
    print(" - gene_string.jsonl")
    print(" - gene_string_GO.jsonl")
    print(" - gene_string_GO_modeling.jsonl")


if __name__ == "__main__":
    main()