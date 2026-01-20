# Bayesian Profile Regression using Variational Inference to Identify Clusters of Multiple Long-Term Conditions Conditioning on Mortality in Population-Scale Data

James Rafferty<sup>1*</sup>,  Keith R. Abrams<sup>2</sup>, Munir Pirmohamed<sup>3</sup>, Mark Davies<sup>4</sup> and Rhiannon K. Owen<sup>1</sup>

<sup>1</sup> Health Data Research UK, Swansea University Medical School, Swansea University, Singleton Park, Swansea, SA1 8PP, Wales, UK,

<sup>2</sup> Department of Statistics, University of Warwick, Coventry, CV4 7AL, England, UK, 

<sup>3</sup> Department of Pharmacology and Therapeutics, University of Liverpool, Liverpool, L3 5TR, England, UK and 

<sup>4</sup> Division of Cancer and Genetics, Cardiff University, Heath Park, Cardiff CF14 4XN, UK

<sup>*</sup> Corresponding author. j.m.rafferty@swansea.ac.uk

This repository contains code used to do the analysis in the paper "Bayesian Profile Regression using Variational Inference to Identify Clusters of Multiple Long-Term Conditions Conditioning on Mortality in Population-Scale Data"

Code defining the model and fitting can be found in the `bayesian_regression_models/` folder. Examples of how the model is used are found in `run_PR_model.ipynb`. Generation of plots and tables found in the paper was performed in `process_results.ipynb`. The script used to perform the simulation study is `iterative_simulation_study.py`. Raw data containing results can be found in the `data/` folder. Analysis of Electronic Health Record data was performed in the Secure Anonymised Information Linkage (SAIL) Databank. Results have been approved out of the gateway by SAIL analyst review. Raw EHR data is not publicly available, but is available via [SAIL](https://saildatabank.com/).

Jim Rafferty.

