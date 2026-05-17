# AI Intrusion Detection

This is a machine learning project I recently started to better understand how ML can be used in cybersecurity. The goal of the project is to classify network traffic as either benign or malicious using the CICIDS2017 dataset.

Right now, the project is still in development. I am focusing on the main steps of a machine learning workflow, including cleaning the data, choosing useful features, training different models, and evaluating the results with classification metrics.

## Dataset

This project uses the CICIDS2017 dataset, which was created by the Canadian Institute for Cybersecurity at the University of New Brunswick.

For this project, I am currently using the Friday datasets, which include examples of normal network traffic as well as attacks such as:

- Botnet activity
- PortScan
- DDoS
- Benign traffic

The dataset also contains flow-based network features, such as packet counts, byte counts, flow duration, ports, and traffic labels. These features help the model learn patterns that may separate normal network behavior from suspicious or malicious activity.

For example, a high number of flow bytes or packets in a short duration going to the same destination could potentially indicate flooding behavior, which is often associated with DDoS attacks.

## Why I Am Building This

I wanted to work on a project that combines machine learning with cybersecurity because both areas interest me. Intrusion detection is a good problem to study because it is not just about getting a high accuracy score. The model also needs to correctly identify harmful traffic.

A model could appear accurate if most of the dataset is benign, but still perform poorly at detecting attacks. Because of that, I plan to look at metrics such as precision, recall, F1-score, and confusion matrices.

## Tools and Technologies

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Jupyter Notebook
- Git/GitHub

## Current Progress

So far, I have started working on:

- Loading the CICIDS2017 Friday datasets
- Looking through the columns and labels
- Understanding the different types of network traffic in the data
- Cleaning the dataset
- Preparing the project for model training and evaluation

## Next Steps

The next steps for this project are:

- Split the data into training and testing sets
- Train baseline machine learning models
- Compare model results using classification metrics
- Create visualizations such as confusion matrices
- Write a short summary of the results and limitations

## Dataset Source

The CICIDS2017 dataset was created by the Canadian Institute for Cybersecurity at the University of New Brunswick.

Dataset link: https://www.unb.ca/cic/datasets/ids-2017.html

## Author

Sameer Shrivastava  
Computer Science/Data Science Student
