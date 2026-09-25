# Classifying Computing Publications by Discipline, Field and Research Methodology

This repository contains the code and data for my University of London BSc Computer Science final project (CM3070). The project follows the CM3060 Natural Language Processing template, using the project idea *"Identifying research methodologies that are used in research in the computing disciplines"*.

## Overview

Anyone who has written a literature review knows how long it takes to screen papers. Before reading a paper properly, you usually want to know three things about it: what area of computing it belongs to, what specific field it covers, and what kind of research it is. Is it an experiment on benchmarks, a new tool, a user study, a review of other work, or a set of proofs? The first two can often be guessed from the venue or the keywords. The methodology is harder, because authors rarely state it in one sentence. It has to be pieced together from phrases like *"we conducted semi-structured interviews"*, *"we evaluate on three benchmarks"* or *"we prove a lower bound"*.

This project builds an NLP pipeline that reads the **title and abstract** of a computing paper and predicts:

| Output | Classes |
|---|---|
| **Discipline** | Intelligent Systems, Software and Security, Human-Centred Computing |
| **Field** | Natural Language Processing, Artificial Intelligence, Machine Learning, Cybersecurity, Software Engineering, Human-Computer Interaction |
| **Research methodology** | Experimental, Design Science, Theoretical, Case Study, Review, Qualitative, Survey |
| **Methodology (grouped)** | Experimental, Design Science, Theory and Review, Human and Field Studies |

The system is designed as a screening aid, not a replacement for reading. Every prediction comes with a **confidence value** and the **words that pushed the model towards that answer**, so the user can see why a label was given and decide whether to trust it.

## Example output

These rows come from the explainable output table in Section 12 of the notebook. They are papers from the held-out test set.

| arXiv ID | Predicted discipline / field | Predicted methodology (grouped) | Confidence | Supporting terms | True field / methodology |
|---|---|---|---|---|---|
| 2608.30910 | Intelligent Systems / NLP | Experimental | 0.90 | llm, model, guided, language model | Machine Learning / Experimental |
| 2609.00043 | Software and Security / Software Engineering | Theory and Review | 0.39 | theory, structure, definition, aspect | Software Engineering / Theoretical |
| 2608.31007 | Intelligent Systems / NLP | Human and Field Studies | 0.46 | interview, language, interaction | HCI / Experimental |
| 2609.00309 | Software and Security / Cybersecurity | Experimental | 0.65 | inference, strategy, adversarial, attack | Cybersecurity / Experimental |

The third row shows why the explanations matter. The word *interview* pushed the prediction towards a human study, but the paper actually analyses interview transcripts with machine learning models. The supporting terms and the low confidence make this easy to spot.

## How it works

1. **Data collection.** Paper metadata is downloaded from the arXiv API for six computer science categories (cs.CL, cs.AI, cs.LG, cs.CR, cs.SE, cs.HC). Only papers whose *primary* category matches are kept, giving 50 papers per field and 300 in total.
2. **Annotation.** Each paper was given one methodology label by hand, following a written annotation guide (in the notebook, Section 4). A short note records the reason for every label.
3. **Preprocessing.** Text is lowercased, LaTeX, URLs, numbers and punctuation are removed, stop words are dropped and words are lemmatised with NLTK. Words such as *we* and *our* are kept on purpose, because phrases like *we prove* or *we interviewed* are strong methodology clues.
4. **Features.** TF-IDF with unigrams and bigrams.
5. **Classifiers.**
   * a transparent keyword baseline with weighted rules
   * Multinomial Naive Bayes, Decision Tree and Logistic Regression (scikit-learn pipelines)
   * a hybrid that lets keyword rules override Logistic Regression, where the rules it trusts are learned from the training data
6. **Evaluation.** Stratified 5-fold cross-validation with macro-averaged precision, recall and F1, plus a held-out 80/20 split for confusion matrices and error analysis. A majority class predictor is reported as a floor.
7. **Explainable output.** For Logistic Regression, each term's contribution is its TF-IDF weight multiplied by the class coefficient. This gives an exact explanation of every prediction, not an approximation.
8. **PDF input.** A PDF can be classified by extracting its abstract, introduction, method and conclusion sections with pypdf.

## Dataset

| Property | Value |
|---|---|
| Source | arXiv API, primary category only |
| Papers | 300 (50 per field) |
| Submission dates | 27 to 31 August 2026 |
| Average abstract length | about 190 words |
| Methodology labels | Experimental 193, Design Science 58, Theoretical 19, Case Study 13, Review 9, Qualitative 6, Survey 2 |

The label distribution is very uneven. Almost two thirds of the papers are Experimental, and the mix changes a lot between fields. Only 12% of NLP papers are not Experimental, compared with 66% of HCI papers.

### Methodology definitions used for annotation

* **Experimental**: tests a model, algorithm or technique through experiments, benchmarks or datasets.
* **Design Science**: builds an artefact (tool, system, framework, protocol) for a stated problem and demonstrates or evaluates it. The artefact is the main contribution.
* **Theoretical**: proofs, formal analysis, analytical models or conceptual theory with no empirical evaluation.
* **Case Study**: an in-depth look at one specific organisation, deployment or real-world setting.
* **Review**: synthesises existing research (systematic review, tutorial, perspective, taxonomy).
* **Qualitative**: interviews, observation, design probes or thematic analysis.
* **Survey**: data collected mainly through questionnaires. A literature "survey" is counted as a Review.

## Results

Macro F1 from 5-fold cross-validation (accuracy in brackets):

| Target | Majority class | Keyword baseline | Naive Bayes | Decision Tree | Logistic Regression |
|---|---|---|---|---|---|
| Discipline (3) | 0.222 (0.500) | 0.690 (0.723) | 0.751 (0.797) | 0.524 (0.580) | **0.772 (0.800)** |
| Field (6) | 0.048 (0.167) | 0.503 (0.497) | 0.601 (0.603) | 0.397 (0.407) | **0.617 (0.620)** |
| Methodology (7) | 0.112 (0.643) | **0.358 (0.493)** | 0.143 (0.653) | 0.239 (0.527) | 0.216 (0.683) |
| Methodology grouped (4) | 0.196 (0.643) | **0.452 (0.517)** | 0.225 (0.650) | 0.308 (0.487) | 0.392 (0.693) |

**Key findings**

* Field and discipline can be predicted well from an abstract. Logistic Regression is the best model, and most of its mistakes are between NLP, AI and Machine Learning, which overlap heavily in recent papers about language models. Cybersecurity was classified perfectly on the test set.
* Methodology is much harder. The trained models get high accuracy mainly by predicting "Experimental", which already gives 64% accuracy. On macro F1, the simple keyword baseline beats every trained model.
* Methodology accuracy drops as a field becomes more varied in its research methods: 0.90 in NLP, 0.44 in HCI.
* Cue words are real but ambiguous. The Review rule finds 7 of the 9 reviews, but it also fires on 43 papers in total, because words like *review* and *perspective* appear in many experimental abstracts.

The notebook also includes n-gram and preprocessing experiments, the hybrid classifier comparison and a full error analysis.

## Repository structure

```
.
├── FYP-FinalSub.ipynb                 # full implementation with outputs (16 sections)
├── raw_arxiv_primary_category.csv     # 300 papers collected from arXiv
├── arxiv_methodology_annotation.csv   # same papers with methodology label and annotation note
├── dataset_with_predictions.csv       # derived labels, cleaned text, baseline and CV predictions
├── results_cross_validation.csv       # model comparison (all targets)
├── results_hybrid_comparison.csv      # hybrid classifier results
├── results_experiments.csv            # n-gram, title only and preprocessing experiments
├── results_sample_output.csv          # sample of the explainable output table
├── models/                            # trained Logistic Regression pipelines (joblib)
│   ├── logreg_field.joblib
│   ├── logreg_methodology.joblib
│   └── logreg_methodology_group.joblib
├── figures/                           # charts saved by the notebook
├── requirements.txt
└── README.md
```

## Getting started

**Requirements:** Python 3.10 or later.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook FYP-FinalSub.ipynb
```

Then run all cells. The notebook downloads the NLTK stop word and WordNet data on the first run.

`USE_CACHED_DATA` is set to `True` by default, so the notebook uses the saved CSV files and reproduces the saved outputs exactly. Setting it to `False` downloads a fresh set of papers from arXiv, but those papers will need methodology labels before the models can be retrained. Section 13 downloads one PDF from arXiv to show how full papers are classified.

### Using a saved model on a new abstract

The saved models expect text that has gone through the same `preprocess` function defined in Section 6 of the notebook. After running that section:

```python
import joblib

field_model = joblib.load("models/logreg_field.joblib")
method_model = joblib.load("models/logreg_methodology_group.joblib")

text = preprocess("Your paper title. Your paper abstract ...")
print(field_model.predict([text])[0])
print(method_model.predict([text])[0])
```

For the full output with discipline, confidence and supporting terms, use `classify_paper(title, abstract)` from Section 12.

### Tests

Section 15 of the notebook contains 14 unit and integration tests covering metadata cleaning, preprocessing, the keyword rules, arXiv response parsing and the full classification pipeline. All of them pass.

## Limitations

* All methodology labels were assigned by one annotator, so there is no agreement measure yet.
* The dataset is small (300 papers) and covers only five days of arXiv submissions.
* The discipline grouping is a design choice for this project, not an external standard.
* The models are trained on titles and abstracts only. The PDF path is a demonstration and has not been evaluated at scale.

## Future work

* Collect more examples of rare methodologies and add a second annotator to measure agreement.
* Train on method sections from full papers, where methodology is usually stated more clearly.
* Try pretrained scientific language models such as SciBERT or SPECTER embeddings for field classification.
* Add a simple web interface where users can paste an abstract or upload a PDF and correct the predicted label.

## Acknowledgements

Thank you to arXiv for use of its open access interoperability. Paper metadata was retrieved through the [arXiv API](https://info.arxiv.org/help/api/index.html). The methodology labels and annotation notes were created for this project.

Built with pandas, NumPy, scikit-learn, NLTK, matplotlib, requests and pypdf.
