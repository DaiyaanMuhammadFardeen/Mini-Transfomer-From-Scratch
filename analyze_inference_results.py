import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import os
import traceback  # Add this for error handling

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create pictures directory if it doesn't exist
pictures_dir = "pictures"
if not os.path.exists(pictures_dir):
    os.makedirs(pictures_dir)

try:
    # Load the CSV data
    df = pd.read_csv('inference_results.csv')
    print(f"Successfully loaded data with shape: {df.shape}")

    # Clean and preprocess the data
    def extract_commit_types(diff_text):
        """Extract types of changes from diff text"""
        # Handle NaN values
        if pd.isna(diff_text):
            return ['other']
        
        types = []
        if '<ADD>' in str(diff_text) or '<MODIFY>' in str(diff_text) or '<REMOVE>' in str(diff_text):
            types.append('code_change')
        if '<COMMENT_ADD>' in str(diff_text) or '<COMMENT_REMOVE>' in str(diff_text) or '<COMMENT_MODIFY>' in str(diff_text):
            types.append('comment_change')
        if any(tag in str(diff_text) for tag in ['<ADD>', '<REMOVE>', '<MODIFY>']):
            types.append('structural_change')
        return types if types else ['other']

    def calculate_diff_lengths(diff_text):
        """Calculate various lengths of the diff text"""
        # Handle NaN values
        if pd.isna(diff_text):
            return {
                'additions': 0,
                'removals': 0,
                'modifications': 0,
                'comments': 0,
                'total_changes': 0
            }
        
        add_count = str(diff_text).count('<ADD>')
        remove_count = str(diff_text).count('<REMOVE>')
        modify_count = str(diff_text).count('<MODIFY>')
        comment_count = str(diff_text).count('<COMMENT')
        
        return {
            'additions': add_count,
            'removals': remove_count,
            'modifications': modify_count,
            'comments': comment_count,
            'total_changes': add_count + remove_count + modify_count + comment_count
        }

    # Apply preprocessing
    df['commit_type'] = df['diff_text'].apply(lambda x: extract_commit_types(x)[0])
    df['diff_length'] = df['diff_text'].apply(lambda x: len(str(x)) if not pd.isna(x) else 0)
    df['message_length'] = df['message'].apply(lambda x: len(str(x)) if not pd.isna(x) else 0)
    df['predicted_length'] = df['predicted_message'].apply(lambda x: len(str(x)) if not pd.isna(x) else 0)

    # Calculate change metrics
    change_metrics = df['diff_text'].apply(calculate_diff_lengths)
    df = pd.concat([df, pd.DataFrame(change_metrics.tolist())], axis=1)

    # Extract keywords from commit messages
    def extract_keywords(text):
        """Extract meaningful keywords from text"""
        # Handle NaN values
        if pd.isna(text):
            return []
        
        # Remove special characters and split
        words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'use', 'have', 'been', 'will', 'with', 'from', 'have', 'that', 'this', 'was', 'had', 'were', 'can', 'use', 'more', 'than', 'well', 'same', 'bit', 'list', 'file', 'value', 'default', 'test', 'bug', 'fix', 'add', 'remove', 'modify', 'change', 'update'}
        return [word for word in words if word not in stop_words]

    df['keywords'] = df['message'].apply(extract_keywords)

    # Flatten all keywords for analysis
    all_keywords = [word for sublist in df['keywords'] for word in sublist if sublist]
    keyword_counts = Counter(all_keywords)

    print(f"Dataset shape: {df.shape}")
    print(f"Commit types distribution:\n{df['commit_type'].value_counts()}")

    # 1. Commit Type Distribution
    plt.figure(figsize=(10, 6))
    type_counts = df['commit_type'].value_counts()
    colors = sns.color_palette("viridis", len(type_counts))
    plt.bar(type_counts.index, type_counts.values, color=colors)
    plt.title('Distribution of Commit Types', fontsize=16, fontweight='bold')
    plt.xlabel('Commit Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, 'commit_type_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: commit_type_distribution.png")

    # 2. Length Analysis - Original Message vs Predicted Message
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Message length comparison
    axes[0,0].hist(df['message_length'], bins=50, alpha=0.7, label='Original Messages', color='skyblue', edgecolor='black')
    axes[0,0].hist(df['predicted_length'], bins=50, alpha=0.7, label='Predicted Messages', color='lightcoral', edgecolor='black')
    axes[0,0].set_title('Distribution of Message Lengths', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Length (characters)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()

    # Scatter plot of original vs predicted length
    axes[0,1].scatter(df['message_length'], df['predicted_length'], alpha=0.6, color='purple')
    axes[0,1].plot([0, max(df['message_length'])], [0, max(df['message_length'])], 'r--', lw=2)
    axes[0,1].set_title('Original vs Predicted Message Length', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Original Message Length')
    axes[0,1].set_ylabel('Predicted Message Length')

    # Difference in lengths
    length_diff = df['predicted_length'] - df['message_length']
    axes[1,0].hist(length_diff, bins=50, color='orange', edgecolor='black')
    axes[1,0].set_title('Difference in Message Lengths (Predicted - Original)', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Length Difference')
    axes[1,0].set_ylabel('Frequency')

    # Box plot comparison
    data_to_plot = [df['message_length'], df['predicted_length']]
    axes[1,1].boxplot(data_to_plot, labels=['Original', 'Predicted'])
    axes[1,1].set_title('Box Plot: Message Length Comparison', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Length')

    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, 'message_length_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: message_length_analysis.png")

    # 3. Changes Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Distribution of different types of changes
    change_cols = ['additions', 'removals', 'modifications', 'comments']
    change_sums = [df[col].sum() for col in change_cols]

    axes[0,0].pie(change_sums, labels=change_cols, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
    axes[0,0].set_title('Distribution of Change Types', fontsize=14, fontweight='bold')

    # Histogram of total changes
    axes[0,1].hist(df['total_changes'], bins=50, color='teal', edgecolor='black', alpha=0.7)
    axes[0,1].set_title('Distribution of Total Changes per Commit', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Total Number of Changes')
    axes[0,1].set_ylabel('Frequency')

    # Average changes by commit type
    avg_changes_by_type = df.groupby('commit_type')[change_cols].mean()
    avg_changes_by_type.plot(kind='bar', ax=axes[1,0], width=0.8)
    axes[1,0].set_title('Average Changes by Commit Type', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Commit Type')
    axes[1,0].set_ylabel('Average Count')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Correlation between different types of changes
    corr_matrix = df[change_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Correlation Between Different Change Types', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, 'changes_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: changes_analysis.png")

    # 4. Keyword Analysis - Word Cloud
    all_keywords_str = ' '.join(all_keywords)

    plt.figure(figsize=(12, 8))
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(all_keywords_str)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Keywords in Commit Messages', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, 'keyword_wordcloud.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: keyword_wordcloud.png")

    # 5. Top Keywords Bar Chart
    top_keywords = dict(keyword_counts.most_common(20))
    plt.figure(figsize=(12, 8))
    keywords, counts = zip(*top_keywords.items())
    colors = sns.color_palette("plasma", len(keywords))
    bars = plt.bar(range(len(keywords)), counts, color=colors)
    plt.title('Top 20 Keywords in Commit Messages', fontsize=16, fontweight='bold')
    plt.xlabel('Keywords', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(range(len(keywords)), keywords, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, 'top_keywords.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: top_keywords.png")

    # 6. Detailed Analysis of Message Quality
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Length of original messages vs their complexity (number of changes)
    axes[0,0].scatter(df['total_changes'], df['message_length'], alpha=0.6, color='green')
    axes[0,0].set_title('Message Length vs Total Changes', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Total Changes')
    axes[0,0].set_ylabel('Original Message Length')

    # Relationship between diff length and message length
    axes[0,1].scatter(df['diff_length'], df['message_length'], alpha=0.6, color='red')
    axes[0,1].set_title('Diff Length vs Original Message Length', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Diff Text Length')
    axes[0,1].set_ylabel('Original Message Length')

    # Relationship between diff length and predicted message length
    axes[1,0].scatter(df['diff_length'], df['predicted_length'], alpha=0.6, color='blue')
    axes[1,0].set_title('Diff Length vs Predicted Message Length', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Diff Text Length')
    axes[1,0].set_ylabel('Predicted Message Length')

    # Ratio of predicted to original message length
    length_ratio = df['predicted_length'] / (df['message_length'] + 1)  # +1 to avoid division by zero
    axes[1,1].hist(length_ratio, bins=50, color='gold', edgecolor='black', alpha=0.7)
    axes[1,1].set_title('Ratio of Predicted to Original Message Length', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Predicted Length / Original Length')
    axes[1,1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: detailed_analysis.png")

    # 7. Box Plot: Message Length by Commit Type
    plt.figure(figsize=(12, 8))
    df.boxplot(column='message_length', by='commit_type', ax=plt.gca())
    plt.title('Distribution of Message Length by Commit Type', fontsize=16, fontweight='bold')
    plt.suptitle('')  # Remove auto-generated title
    plt.xlabel('Commit Type', fontsize=12)
    plt.ylabel('Message Length', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(pictures_dir, 'message_length_by_type.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: message_length_by_type.png")

    # 8. Pairplot of numerical features
    numerical_cols = ['message_length', 'predicted_length', 'diff_length', 'additions', 'removals', 'modifications', 'comments', 'total_changes']
    sample_df = df[numerical_cols].sample(min(1000, len(df)))  # Sample for performance

    pair_plot = sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha': 0.6})
    pair_plot.fig.suptitle('Pairwise Relationships Between Features', y=1.02, fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(pictures_dir, 'pairplot_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: pairplot_features.png")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average original message length: {df['message_length'].mean():.2f}")
    print(f"Average predicted message length: {df['predicted_length'].mean():.2f}")
    print(f"Average diff length: {df['diff_length'].mean():.2f}")
    print(f"Average total changes: {df['total_changes'].mean():.2f}")
    print(f"Top 10 most frequent keywords: {dict(keyword_counts.most_common(10))}")

    print(f"\nAll charts have been saved to the '{pictures_dir}' directory.")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("Full traceback:")
    traceback.print_exc()