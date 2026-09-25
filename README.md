# CM3060 Final Project: research methodology classifier

- `CM3060_research_methodology_classifier.ipynb`: full pipeline, executed (HTML copy alongside)
- `report/CM3060_Final_Report_Draft.docx`: report draft (update the table of contents in Word: right click > Update field)
- `data/corpus.csv`: 7,325 Crossref abstracts labelled by venue; `data/arxiv_ood.csv`: 374 arXiv abstracts
- `data/gold_methodology.csv`: 200 hand-labelled abstracts (items 0-99 dev, 100-199 test)
- `src/`: fetch scripts, text cleaning, methodology lexicon, zero-shot hypotheses
- `cache/`: MiniLM embeddings and zero-shot scores (delete to recompute)
- `figures/`: all plots and result tables

Install: `pip install pandas scikit-learn nltk matplotlib seaborn sentence-transformers transformers torch pymupdf requests jupyter`
