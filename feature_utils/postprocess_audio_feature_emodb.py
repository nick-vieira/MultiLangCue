import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import numpy as np

def plot_feature_distributions(df, features, thresholds, title, output_filename, num_classes):
    fig, axes = plt.subplots(4, 2, figsize=(20, 25))
    fig.suptitle(title, fontsize=16)
    
    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        sns.histplot(df[feature].replace(0, np.nan).dropna(), kde=True, ax=ax)
        ax.set_title(feature)
        
        colors = ['r', 'g', 'b', 'y', 'm']
        for threshold, color in zip(thresholds[feature].keys(), colors):
            ax.axvline(thresholds[feature][threshold], color=color, linestyle='--')
            ax.text(thresholds[feature][threshold], ax.get_ylim()[1], threshold,
                    ha='center', va='bottom', color=color, rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def extract_thresholds_and_stats(df, num_classes):
    features = ["duration", "avg_intensity", "intensity_variation", "avg_pitch", 
                "pitch_std", "pitch_range", "articulation_rate", "mean_hnr"]
    
    thresholds = {}
    stats = {}
    
    for feature in features:
        feature_data = df[feature].replace(0, np.nan).dropna()
        percentiles = feature_data.quantile(np.linspace(0.1, 0.9, num_classes - 1)).to_dict()
        thresholds[feature] = percentiles
        stats[feature] = {'mean': feature_data.mean(), 'std': feature_data.std()}
    
    plot_feature_distributions(df, features, thresholds, 
                               "Distribution of EmoDB Audio Features", 
                               '../distributions/feature_distributions_emodb.png', num_classes)
    
    return thresholds, stats

def standardize_and_categorize(df, thresholds, stats, num_classes):
    for feature in thresholds.keys():
        df[f'{feature}_standardized'] = df[feature].apply(lambda x: (x - stats[feature]['mean']) / stats[feature]['std'] if pd.notna(x) else np.nan)
        df[f'{feature}_category'] = df[feature].apply(lambda x: categorize(x, thresholds[feature], num_classes))
    return df

def categorize(value, thresholds, num_classes):
    if pd.isna(value) or value == 0:
        return 'none'
    
    threshold_keys = list(thresholds.keys())
    for i, key in enumerate(threshold_keys):
        if value <= thresholds[key]:
            return f'class_{i+1}'
    return f'class_{num_classes}'

def main():
    df = pd.read_csv('../speech_features/emodb_audio_features.csv')
    num_classes = 5  # Modify for different category levels
    thresholds, stats = extract_thresholds_and_stats(df, num_classes)
    df = standardize_and_categorize(df, thresholds, stats, num_classes)
    df.to_csv('../speech_features/processed_emodb_audio_features.csv', index=False)
    print("Processed EmoDB audio features saved.")

if __name__ == "__main__":
    main()