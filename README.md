# AI Intrusion Detection

This is a machine learning project I built to better understand how ML can be used in cybersecurity. The goal is to classify network traffic as benign or as a specific type of attack using the CICIDS2017 dataset.

The project covers the main steps of a machine learning workflow: cleaning the data, choosing useful features, training different models, and evaluating the results with classification metrics.

## Dataset

This project uses the CICIDS2017 dataset, created by the Canadian Institute for Cybersecurity at the University of New Brunswick.

I am using the Friday files, which contain normal network traffic along with these attack types:

- Botnet activity
- PortScan
- DDoS

The dataset contains flow-based network features, such as packet counts, byte counts, flow duration, and ports. These features help the model learn patterns that separate normal behavior from malicious activity.

For example, a high number of flow bytes or packets in a short duration going to the same destination could indicate flooding behavior, which is often associated with DDoS attacks.

## Why I Am Building This

I wanted a project that combines machine learning with cybersecurity because both areas interest me. Intrusion detection is a good problem to study because it is not just about getting a high accuracy score. The model also needs to correctly catch harmful traffic, even when attacks are rare compared to normal traffic.

## Tools and Technologies

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- streamlit
- Git/GitHub

## Interactive App

I built a Streamlit app (`app.py`) that turns the trained models into something you can interact with. I also use Claude, Anthropic's AI model, to explain the results in plain English.

In the app, you can:

- Pick a network flow from the test set, or enter feature values manually]
- See the Random Forest and Logistic Regression predictions side by side]
- Ask Claude to explain why a flow was classified as an attack, based on its most important features]

The classifiers make the actual predictions. Claude does not classify traffic, but it interprets the model output and feature values and explains them in language a non-expert could follow, similar to how a security analyst might summarize an alert.

### Class Distribution

![Class distribution](artifacts/class_distribution.png)

This shows how many flows belong to each label. The classes are not balanced: [ex: "Bot has far fewer examples than BENIGN, DDoS, and PortScan"]. This matters because a model can score high accuracy just by predicting the common classes, so I focus on per-class precision and recall instead of accuracy alone.

### Feature Correlation

![Correlation matrix](artifacts/correlation_matrix.png)

This heatmap shows how strongly the numeric features are correlated with each other. Many features are highly correlated  [ex: two related features, like forward packet counts and total packet counts], which means some of them carry redundant information.

### Feature Importance

![Feature importance](artifacts/feature_importance.png)

These are the features the Random Forest relied on most. The top features were average packet size, packet length mean, and BWD Packets. This makes sense because attacks tend to produce very uniform, distinctive packet sizes: port scans send tiny probe packets that carry almost no data, DDoS floods repeat the same kind of request over and over, and in both cases the target sends back very little, which shows up in the backward packet counts. Normal traffic, like web browsing and downloads, has a much wider mix of packet sizes and more two-way communication.

### Confusion Matrices

![Confusion matrices](artifacts/confusion_matrices.png)

A confusion matrix shows, for each true label, what the model predicted. Correct predictions fall on the diagonal, and mistakes fall everywhere else.

Both models catch nearly every attack, but they differ a lot in false alarms. Logistic Regression labeled 13,287 benign flows as Bot and misclassified about 19% of all benign traffic as some kind of attack, while Random Forest misclassified less than 1%. Logistic Regression also mistook 330 DDoS flows for Bot.

Random Forest's main remaining weakness is the Bot class. It caught 381 of the 389 real Bot flows, but it also flagged 627 benign flows as Bot, so most of its Bot alerts are still false alarms. Bot is the rarest class in the data, and bot traffic is designed to look like normal traffic, which makes it the hardest class to separate cleanly.

### Normalized Confusion Matrix

![Normalized confusion matrix](artifacts/confusion_matrix_normalized.png)

This version shows percentages instead of raw counts, so each row sums to 100%. It makes the smaller classes easier to judge, since their mistakes don't get hidden by the large classes. The diagonal values are each class's recall: the percentage of that traffic type the model correctly caught.

### Precision, Recall, and F1 Comparison

![Precision, recall, and F1 comparison](artifacts/precision_recall_f1_comparison.png)

This compares both models on three metrics for each class:

- **Precision:** when the model predicts a class, how often it is right
- **Recall:** out of all real examples of a class, how many the model caught
- **F1 score:** a single score that balances precision and recall

Both models score close to 1.0 on PortScan and DDoS, so those attacks are easy to detect with flow features. The biggest differences are on BENIGN and Bot.

Logistic Regression's benign recall drops to about 0.81 because it labels roughly one in five normal flows as an attack. Furthermore, its Bot precision is only about 0.03: it catches almost every real Bot flow, but fewer than 3% of its Bot alerts are correct, giving it a Bot F1 score near 0.05.

Random Forest does much better, with a Bot precision of about 0.38 and an F1 score of about 0.54 while keeping Bot recall near 0.98. Even so, Bot is clearly its weakest class. Accuracy alone would be misleading here because Bot is so rare, a model could be wrong about most of its Bot alerts and still have an overall accuracy above 99%.

## How to Run

1. Download the Friday CSV files from the [CICIDS2017 dataset page](https://www.unb.ca/cic/datasets/ids-2017.html) and place them in the project folder.
2. Install the dependencies:
```
   pip install -r requirements.txt
```
3. Train the models:
```
   python ml_start.py
```
4. Generate the plots:
```
   python plots.py
```
5. Launch the demo app with
```
    streamlit run app.py`
```

The raw data and trained model files are not included in this repo because of their size. Running the scripts above recreates them.

## Author

Sameer Shrivastava  
Computer Science & Data Science Student, Rutgers University–New Brunswick
