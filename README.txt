    This repo contains the entire Alzheimer's disease capstone project.
There are in depth ipynb files for each section of the project that culminate
in a milestone report and a final report. Presentation of the results are included
in the AD_Final_Presentation.pptx file. Custom modules (.py files) were created
and used for this analysis.
    Classical statistical techniques were used to identify threshold values
for predicting high-risk patients for Alzheimer's disease. Supervised machine
learning algorithms were used to generate predictive models capable of identifying
high-risk patients for Alzheimer's disease.

Word Documents:
Capstone1_Ideas                    Initial ideas for the capstone project
Capstone1_Proposal                 Proposal for capstone project

IPython Notebooks:

1-Data_Import_and_Clean.ipynb            data wrangling and cleaning steps
2-Data_Storytelling.ipynb                exploratory data analysis
3-Statistical_Data_Analysis.ipynb        statistical analysis
4-Capstone_Milestone_Report.ipynb        summary milestone report
5-Machine_Learning.ipynb                 in depth analysis, machine learning, and predictive modeling
6-Alzheimers_Final_Report.ipynb          final report

Additional IPython Notebook:
zz2-Data_Storytelling_Raw-Copy1.ipynb    additional exploratory analyses omitted from final product 

Custom modules:
adnidatawrangling.py         code to wrangle and clean the data (derived from 1-Data_Import_and_Clean.ipynb)
eda.py                       code to produce exploratory data analysis results and visualizations
                                    (derived from 2-Data_Storytelling.ipynb)
ml.py                        code to process and summarize machine learning algorithms
                                    (derived from 5-Machine_Learning.ipynb)
sda.py                       code for statistical analysis (derived from 3-Statistical_Data_Analysis.ipynb)

Borrowed modules:
feature_selector.py          feature selection tool obtained from https://github.com/WillKoehrsen/feature-selector

Data files:
ADNIMERGE.csv           comma separated values file containing the data used for this analysis
ADNIMERGE_DICT.csv      comma separated values file containing a data dictionary for ADNIMERGE.csv

Additional files from exploratory data analysis:
pairplot.png                image of pairplot to explore possible correlations
pairplot_clin.png           image of pairplot to explore possible correlations
pairplot_scans.png          image of pairplot to explore possible correlations
