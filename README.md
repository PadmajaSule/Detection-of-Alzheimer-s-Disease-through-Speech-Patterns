# Detection-of-Alzheimer-s-Disease-through-Speech-Patterns
## Problem Definition
The project aims to develop an automatic model for early Alzheimer's disease (AD) detection using speech analysis. The critical challenge addressed is the need for accessible and timely methods of identifying AD onset, considering the significant impact of delayed diagnosis on patient outcomes. By leveraging advanced deep learning algorithms, the model seeks to analyze subtle linguistic changes in speech patterns that may signify the early stages of AD, thus enabling accurate diagnosis and timely intervention. This initiative responds to the urgent need for innovative approaches to AD detection, aiming to revolutionize early diagnosis practices and ultimately improve patient care and quality of life.

## Data Processing Pipeline for Dementia Research
### Overview
This repository contains scripts and data used for analyzing transcripts of conversations with individuals affected by various forms of dementia. We obtained these transcripts in CHAT Format (Codes for the Human Analysis of Transcripts) from DementiaBank’s Pitt Corpus. Our analysis involved extracting specific features from these transcripts using CLAN software.
### Data Extraction
The following commands were employed for extracting features:
*	IPSYN: Used for measuring syntactic complexity.
*	EVAL: Applied for determining word-type ratios, conducting grammatical analysis, and counting utterances.
### Data Preparation
All extracted features were consolidated into a single CSV file containing over 100+ features. After processing, the dataset was refined to include 44 relevant features, which were subsequently used for training our models.
### Acknowledgments
*	DementiaBank’s Pitt Corpus for providing the original transcripts.
*	CLAN software for facilitating the extraction of linguistic features.

