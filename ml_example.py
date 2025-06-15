import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def generate_synthetic_data(n_samples=10000):
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)
    
    # Create two feature vectors: age and income
    data = {
        'age': [fake.random_int(min=18, max=80) for _ in range(n_samples)],
        'income': [fake.random_int(min=20000, max=150000) for _ in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Simple threshold-based classification with noise
    # Over 40 years old AND Make more than $60,000 annually for class 1.
    base_target = (df['age'] > 55) & (df['income'] > 60000)

    # 10% of the data points are randomly flipped to the opposite class
    noise = np.random.random(n_samples) < 0.05
    df['target'] = (base_target ^ noise).astype(int) 
    
    return df

def plot_distribution(df):
    plt.figure(figsize=(10, 6))
    for target in [0, 1]:
        mask = df['target'] == target
        plt.scatter(df[mask]['age'], df[mask]['income'], 
                   alpha=0.5, 
                   label=f'Class {target}')
    
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Distribution of Classes by Age and Income')
    plt.legend()
    plt.show()

def evaluate_model(y_true, y_pred, y_pred_proba):
    print("\nLog Loss:", round(log_loss(y_true, y_pred_proba), 4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def main():
    df = generate_synthetic_data()
    
    X = df[['age', 'income']]
    y = df['target']

    print(y.value_counts(normalize=True))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    print("\nAverage predicted probabilities:")
    print(f"Class 0: {y_pred_proba[:, 0].mean():.3f}")
    print(f"Class 1: {y_pred_proba[:, 1].mean():.3f}")
    
    print("Data Distribution:")
    plot_distribution(df)
    
    evaluate_model(y_test, y_pred, y_pred_proba)

if __name__ == "__main__":
    main()
