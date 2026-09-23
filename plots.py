import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
sns.set(style='darkgrid')


# Load everything the training script saved

rf_model = joblib.load('artifacts/rf_model.pkl')
lr_model = joblib.load('artifacts/lr_model.pkl')
X_test = joblib.load('artifacts/X_test.pkl')
y_test = joblib.load('artifacts/y_test.pkl')
rf_preds = joblib.load('artifacts/rf_preds.pkl')
lr_preds = joblib.load('artifacts/lr_preds.pkl')
feature_cols = joblib.load('artifacts/feature_cols.pkl')
corr = joblib.load('artifacts/corr.pkl')
label_counts = joblib.load('artifacts/label_counts.pkl')

CLASS_ORDER = ['BENIGN', 'Bot', 'PortScan', 'DDoS']


def plot_class_distribution():
    """
    Bar chart showing how many rows belong to each traffic type.
    Uses a log scale because BENIGN traffic massively outnumbers
    the attack classes, especially Bot, so a normal scale would
    make the smaller bars invisible.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=label_counts.index, y=label_counts.values,
        order=CLASS_ORDER, palette='viridis'
    )
    ax.set_yscale('log')
    plt.title('Class Distribution (log scale)', fontsize=14)
    plt.xlabel('Traffic Type')
    plt.ylabel('Count (log scale)')
    for i, v in enumerate(label_counts.reindex(CLASS_ORDER).values):
        ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('artifacts/class_distribution.png', dpi=150)
    plt.show()


def plot_correlation_heatmap():
    """
    Heatmap showing how every numeric feature relates to every other
    feature. Useful for spotting which features are redundant or useless.
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, cmap='coolwarm', annot=False, linewidth=0.5)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.title('Correlation Matrix', fontsize=18)
    plt.tight_layout()
    plt.savefig('artifacts/correlation_matrix.png', dpi=150)
    plt.show()


def plot_confusion_matrices():
    """
    Shows the Random Forest and Logistic Regression confusion matrices
    side by side. Each cell is a raw count of how many rows with a
    given true label were predicted as each class, so you can see
    exactly where each model gets confused.
    """
    preds_list = [rf_preds, lr_preds]
    titles = ['Confusion Matrix (Random Forest)', 'Confusion Matrix (Logistic Regression)']

    conf_matrices = [confusion_matrix(y_test, p, labels=CLASS_ORDER) for p in preds_list]

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    for i, cm in enumerate(conf_matrices):
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axs[i],
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER
        )
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Predicted label')
        axs[i].set_ylabel('True label')

    fig.tight_layout()
    plt.savefig('artifacts/confusion_matrices.png', dpi=150)
    plt.show()


def plot_feature_importance():
    """
    Horizontal bar chart of the features the Random Forest relies
    on most. Good for explaining why the model makes the decisions
    it makes, not just how accurate it is.
    """
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(9, 8))
    importances.plot(kind='barh', color='teal')
    plt.title('Random Forest Feature Importances', fontsize=14)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('artifacts/feature_importance.png', dpi=150)
    plt.show()


def plot_normalized_confusion_matrix():
    """
    This is the same concept as the regular confusion matrix, but each row is
    converted to a percentage of that true class instead of a raw
    count. Raw counts make rare classes like Bot look fine just
    because BENIGN has so many more rows overall, normalizing by
    row fixes that and shows the real misclassification rate.
    """
    cm = confusion_matrix(y_test, rf_preds, labels=CLASS_ORDER, normalize='true')

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER
    )
    plt.title('Random Forest, Normalized Confusion Matrix\n(% of each true class)', fontsize=13)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig('artifacts/confusion_matrix_normalized.png', dpi=150)
    plt.show()


def plot_precision_recall_f1():
    """
    Grouped bar chart comparing precision, recall, and F1 score for
    both models across every class. Looking at these three metrics
    together matters because a model can look great on one metric,
    while performing poorly on another for the same class.
    """
    rf_report = classification_report(y_test, rf_preds, output_dict=True)
    lr_report = classification_report(y_test, lr_preds, output_dict=True)

    metrics = ['precision', 'recall', 'f1-score']
    rows = []
    for cls in CLASS_ORDER:
        for metric in metrics:
            rows.append({'class': cls, 'metric': metric, 'model': 'Random Forest',
                         'value': rf_report[cls][metric]})
            rows.append({'class': cls, 'metric': metric, 'model': 'Logistic Regression',
                         'value': lr_report[cls][metric]})
    df = pd.DataFrame(rows)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, metric in enumerate(metrics):
        subset = df[df['metric'] == metric]
        sns.barplot(
            data=subset, x='class', y='value', hue='model',
            order=CLASS_ORDER, ax=axs[i]
        )
        axs[i].set_title(metric.capitalize())
        axs[i].set_ylim(0, 1.05)
        axs[i].set_xlabel('')
        axs[i].set_ylabel('Score' if i == 0 else '')
        axs[i].legend(loc='lower left', fontsize=8)

    fig.suptitle('Per-Class Metrics: Random Forest vs Logistic Regression', fontsize=14)
    fig.tight_layout()
    plt.savefig('artifacts/precision_recall_f1_comparison.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    plot_class_distribution()
    plot_correlation_heatmap()
    plot_confusion_matrices()
    plot_feature_importance()
    plot_normalized_confusion_matrix()
    plot_precision_recall_f1()
    print("\nAll plots saved to ./artifacts/ as PNGs.")