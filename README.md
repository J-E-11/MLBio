# Linear and Non-Linear Models to Classify Single-Cell Transcriptomics of Chronic Myeloid Leukemia Patients During Diagnosis and Remission Therapy Development

##### Group 4: Ana Helena Valdeira Caetano, Olivia Huang, Rucha Narkhede, Sukhleen Kaur 

This repository consists of the data used and the code used for the Machine Learning in Bioinformatics (CS4260) course at TU Delft. The purpose of this study was to classify HSC, BCR-ABL<sup>+</sup> Diagnosis and BCR-ABL<sup>+</sup> Remission.

This repository consists of 4 directories:
* [**preprocess**](https://github.com/J-E-11/MLBio/tree/master/preprocess): contains the R script to remove batch effect.
* [**data**](https://github.com/J-E-11/MLBio/tree/master/data): contains the original extracted dataset as well as the dataset with the batch effect removed.
* [**three_class_problem**](https://github.com/J-E-11/MLBio/tree/master/three_class_problem): contains the scripts used for feature selection and classifications for the three class problem.
* [**two_class_problem**](https://github.com/J-E-11/MLBio/tree/master/two_class_problem): contains 2 further directories for the two-class problem, i.e classification into BCR-ABL<sup>+</sup> Diagnosis and BCR-ABL<sup>+</sup> Remission:
    * [**src**](https://github.com/J-E-11/MLBio/tree/master/binary_class_problem/src): containing the scripts used for feature selection and classification as well as the .csv files containing the selected features. There are 4 .csv files. Two for features selected from the original extracted dataset and two for features selected from dataset with the batch effect removed.
    * [**img**](https://github.com/J-E-11/MLBio/tree/master/binary_class_problem/img): containing plots generated by the source codes. Again there are 2 more directories. One for plots generated on features selected from the original extracted dataset and the other for features selected from the dataset with batch effect removed. Since two feature selection models were used, they both contain further 2 directories, one for each model.
* [**extras**](https://github.com/J-E-11/MLBio/tree/master/extras): contains extra code that was not included for the report but included in the appendix.


To remove batch effect, [**Limma Package**](https://rdrr.io/bioc/limma/) in **R 3.6.2** was used. The feature selection, classification and plotting was done in **Python 3.7.5** using the following libararies:
* [**Pandas**](https://pandas.pydata.org/pandas-docs/version/0.24.2/index.html)
* [**NumPy**](https://numpy.org/devdocs/release/1.16.2-notes.html)
* [**Scikit-Learn**](https://scikit-learn.org/stable/index.html)
* [**XGBoost**](https://xgboost.readthedocs.io/en/stable/python/index.html)
* [**Seaborn**](https://seaborn.pydata.org/)
* [**Matplotlib**](https://matplotlib.org/3.0.3/index.html)

