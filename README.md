# Explorative analysis of precursors of physical aggression in a health care institute: a Text Mining approach
This project is commissioned by GGZ Rivierduinen. Their goal was to do more with data. This is achieved by analyzing their incident reports for precursors. The goal of this thesis project was to detect causes of physical aggression towards people or material. 

# Overview
This system preprocesses incident reports into clauses, predicts aggressive clauses by default using the machine-learning approach. This approach trains a Support Vector Machine Model and yields a precision of 87.9 per cent, a recall of 63.6 per cent and an F1 score of 84.6 per cent. Another implemented approach is rule-based which yields a precision score of 81.4 per cent, a recall of 65.4 per cent and a F1 score of 70.0 per cent. Furthermore the 6 clauses preceding the predictions are clustered by BERTopic. This results in groups of precursors of aggressive incidents. 

# Project structure

```
hamersma-agression-causes
└───input
│       │   sample_input.xls
|       |   agressie_ww.txt
|       |   sample_devset.txt
|       |   sample_testset.txt
└───output
│       │   predictions_rule-based.csv
│       │   predictions_rule-based_error_analysis.csv
│       │   predictions_SVM.csv
│       │   predictions_SVM_error_analysis.csv
|       └───clusters
└───figures
└───models
└───utils
│       │   bert_embeddings.py
│       │   cluster_analysis.py
│       │   data_analysis.py
│       │   evaluate_annotation.py
│       │   quality_vectors.py
│       │   transform_annotations.py
│   │   main.py
│   │   preprocessing.py
│   │   approach1_rulebased.py
│   │   approach2_machine_learning.py
│   │   evaluation.py
│   │   bertopic_clustering.py
│   .gitignore
│   LICENSE
│   README.md
│   requirements.tx
```

## Thesis report
The dissertation is uploaded in Canvas: https://canvas.vu.nl/courses/47033/assignments/104476

## Data
Since the aggression incidents are highly sensitive, the data cannot be shared. For this reason, sample data is provided.

Data needed in inputfolder:
- txt file with aggressive verbs, see agressie_ww
- txt file with development vim ids, see sample_devset
- txt file with test vim ids, see sample_testset
- excel file with aggression incidents, see sample_input

Data needed in annotations:
- excel file with annotations, see sample annotations_ann1

## README
Download or install:
- BERTopic: https://github.com/MaartenGr/BERTopic/
- Spacy, large Dutch language model: https://spacy.io/models/nl
- requirements.txt

## Utils
The scripts in this folder are used for data visualization or transformation of the annotations during development. Each script runs by calling it without arguments.

- bert_embeddings.py: Generates BERT embeddings by pretrained BERTje embedding model
- transform_annotations.py: First, the data was annotated on token level, this script transforms these annotations to clauses.
- data_analysis.py: Visualizes the annotations for data understanding: shows position of cause and aggression in sentence/clause, distribution of labels, amount of clauses labelled aggressive or as cause in one incident report
- cluster_analysis.py: Visualizes distribution of clauses over clusters
- evaluate_annotation.py: Calculates Inter-Annotator agreement score and corrects for chance
- quality_vectors.py: Visualizes similarity scores of word- and clause pairs for different embedding models

## Main
When running the main script, no arguments need to be provided. The system yields rule-based and machine learning predictions for the sample_input.xls file in the input folder. The generated files, such as the preprocessed file, the predictions and files for error analysis are placed in the output folder. Within the output folder there is a folder 'clusters'. In here, the label, keywords and size files of the clustering model are placed. Models and visualizations are saved in the folder 'models' and 'figures'.

- preprocessing.py: Preprocesses incident report into clauses and removes abbreviations, saves new file with a clause per row and its identifiers
- approach_rulebased.py: Generates predictions following 5 rules, saves new file with predictions and error analysis containing the true labels in output folder
- approach2_machine_learning.py:	Generates predictions by training a SVM, saves new file with predictions and error analysis containing the true labels in output folder
- bertopic_clustering.py: Clusters clauses making use of BERTopic, results in output/cluster
- evaluation.py: Selects predictions of test set and gold labels, calculates performance and prints confusion matrix




