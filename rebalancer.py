#!/usr/bin/env python3
"""
Dataset Rebalancer Script

This script rebalances datasets based on various strategies to reduce redundancy
and improve model training effectiveness.
"""

import argparse
import gc
import sys
import time
import warnings
from collections import Counter, defaultdict
import heapq
import math
import os
import re
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import hashlib

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global debug flag
DEBUG = True

def debug_print(msg):
    """Print debug messages if DEBUG is enabled."""
    if DEBUG:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")
        sys.stdout.flush()

def aggressive_gc():
    """Perform aggressive garbage collection."""
    debug_print("Performing aggressive garbage collection...")
    collected = gc.collect()
    debug_print(f"Garbage collector: collected {collected} objects")

def calculate_ngrams(text, n=3):
    """Calculate n-grams for a text."""
    debug_print(f"Calculating {n}-grams for text: {text[:50]}...")
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(' '.join(words[i:i+n]))
    return ngrams

def hash_text(text):
    """Create a hash for a text for fast comparison."""
    return hashlib.md5(text.encode()).hexdigest()

def is_vague_message(message):
    """Check if a message is vague (e.g., 'Feature: Add new functionality ###')."""
    # Pattern to match "Feature: Add new functionality ###" where ### is any number of digits
    pattern = r'^Feature:\s*Add\s*new\s*functionality\s*\d+$'
    return bool(re.match(pattern, message.strip(), re.IGNORECASE))

def process_chunk_similarity(args):
    """Process a chunk of similarity calculations for multiprocessing."""
    chunk_data, tfidf_matrix, threshold = args
    selected_indices = []
    
    for i, similarity_row in enumerate(chunk_data):
        # Find similar messages
        similar_indices = np.where(similarity_row >= threshold)[0]
        
        # Select the first one as representative
        if len(similar_indices) > 0:
            selected_indices.append(similar_indices[0])
            
    return selected_indices

class DatasetRebalancer:
    """Main class for rebalancing datasets."""
    
    def __init__(self, input_file, output_file, strategy='clustering', sample_ratio=0.5, 
                 max_samples=None, random_state=42, remove_vague=True, use_multiprocessing=True):
        self.input_file = input_file
        self.output_file = output_file
        self.strategy = strategy
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples
        self.random_state = random_state
        self.remove_vague = remove_vague
        self.use_multiprocessing = use_multiprocessing
        self.df_original = None
        self.df_rebalanced = None
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        debug_print(f"Initialized DatasetRebalancer with:")
        debug_print(f"  Input file: {input_file}")
        debug_print(f"  Output file: {output_file}")
        debug_print(f"  Strategy: {strategy}")
        debug_print(f"  Sample ratio: {sample_ratio}")
        debug_print(f"  Max samples: {max_samples}")
        debug_print(f"  Random state: {random_state}")
        debug_print(f"  Remove vague messages: {remove_vague}")
        debug_print(f"  Use multiprocessing: {use_multiprocessing}")
        
    def load_data(self):
        """Load the dataset from parquet file."""
        debug_print(f"Loading data from {self.input_file}...")
        self.df_original = pd.read_parquet(self.input_file)
        debug_print(f"Loaded {len(self.df_original)} rows from parquet file")
        
        # Fill NaN values ONLY for messages (never for diff_text)
        debug_print("Filling NaN values for messages only...")
        # Create a working copy to avoid modifying original data
        self.df_working = self.df_original.copy()
        self.df_working['message'] = self.df_working['message'].fillna('')
        # NEVER modify diff_text - keep it exactly as is
        
        aggressive_gc()
        
    def filter_vague_messages(self):
        """Remove vague commit messages."""
        if not self.remove_vague:
            return
            
        debug_print("Filtering out vague commit messages...")
        original_count = len(self.df_working)
        
        # Create a mask for non-vague messages with progress bar
        if TQDM_AVAILABLE:
            message_iter = tqdm(self.df_working['message'], desc="Checking messages")
        else:
            message_iter = self.df_working['message']
            debug_print("Checking messages for vagueness...")
            
        vague_mask = [is_vague_message(msg) for msg in message_iter]
        # Invert mask to keep non-vague messages
        mask = ~np.array(vague_mask)
        self.df_working = self.df_working[mask].copy()
        
        filtered_count = original_count - len(self.df_working)
        debug_print(f"Filtered out {filtered_count} vague messages. Remaining: {len(self.df_working)}")
        
    def visualize_before(self):
        """Create visualizations of the original dataset."""
        debug_print("Creating visualizations of original dataset...")
        
        # Message length distribution
        debug_print("Calculating message lengths...")
        message_lengths = self.df_original['message'].str.len()
        
        # Plot message length distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(message_lengths, bins=50, alpha=0.7, color='blue')
        plt.xlabel('Message Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Original Dataset: Message Length Distribution')
        plt.yscale('log')
        
        # Top frequent messages
        debug_print("Finding top frequent messages...")
        message_counts = self.df_original['message'].value_counts()
        top_messages = message_counts.head(20)
        
        plt.subplot(1, 2, 2)
        top_messages.plot(kind='barh', color='green', alpha=0.7)
        plt.xlabel('Frequency')
        plt.title('Top 20 Most Frequent Messages')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('dataset_before_rebalancing.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        debug_print("Saved visualization as dataset_before_rebalancing.png")
        
    def rebalance(self):
        """Apply the selected rebalancing strategy."""
        debug_print(f"Starting rebalancing with strategy: {self.strategy}")
        
        # First filter vague messages if requested
        self.filter_vague_messages()
        
        start_time = time.time()
        if self.strategy == 'clustering':
            self._clustering_sampling()
        elif self.strategy == 'near_duplicate':
            self._near_duplicate_detection()
        elif self.strategy == 'diversity':
            self._diversity_sampling()
        elif self.strategy == 'frequency_downscale':
            self._frequency_downscaling()
        elif self.strategy == 'quality_scoring':
            self._quality_scoring()
        elif self.strategy == 'information_value':
            self._information_value_scoring()
        elif self.strategy == 'topic_modeling':
            self._topic_modeling_approach()
        elif self.strategy == 'ngram_grouping':
            self._ngram_grouping()
        elif self.strategy == 'multi_criteria':
            self._multi_criteria_balancing()
        elif self.strategy == 'remove_duplicates':
            self._remove_duplicates_only()
        else:
            debug_print(f"Unknown strategy: {self.strategy}. Using remove_duplicates as default.")
            self._remove_duplicates_only()
            
        end_time = time.time()
        debug_print(f"Rebalancing completed in {end_time - start_time:.2f} seconds")
        aggressive_gc()
        
    def _remove_duplicates_only(self):
        """Only remove exact duplicates, keep all other rows."""
        debug_print("Removing only exact duplicates...")
        
        # Hash ONLY messages for quick duplicate detection (never process diff_text)
        debug_print("Hashing messages for duplicate detection...")
        if TQDM_AVAILABLE:
            hashes = [hash_text(msg) for msg in tqdm(self.df_working['message'], desc="Hashing messages")]
        else:
            debug_print("Hashing messages...")
            hashes = [hash_text(msg) for msg in self.df_working['message']]
            
        df_with_hash = self.df_working.copy()
        df_with_hash['message_hash'] = hashes
        
        # Remove exact duplicates, keeping the first occurrence
        debug_print("Removing exact duplicates...")
        self.df_rebalanced = df_with_hash.drop_duplicates(subset=['message_hash'], keep='first').copy()
        # Remove the temporary hash column - KEEP ALL OTHER COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = self.df_rebalanced.drop('message_hash', axis=1)
        
        debug_print(f"Removed duplicates. Original: {len(self.df_working)}, Rebalanced: {len(self.df_rebalanced)}")
        
    def _clustering_sampling(self):
        """Use text clustering algorithms on message embeddings."""
        debug_print("Applying clustering-based sampling...")
        
        # Work with the full dataset but in a memory-efficient way
        debug_print(f"Processing {len(self.df_working)} rows...")
        
        # Vectorize ONLY messages using TF-IDF with a reasonable max_features
        debug_print("Vectorizing messages with TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
        
        if TQDM_AVAILABLE:
            tfidf_matrix = vectorizer.fit_transform(
                tqdm(self.df_working['message'], desc="Vectorizing messages")
            )
        else:
            debug_print("Vectorizing messages...")
            tfidf_matrix = vectorizer.fit_transform(self.df_working['message'])
            
        debug_print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Reduce dimensionality if needed
        if tfidf_matrix.shape[1] > 2000:
            debug_print("Reducing dimensionality with Truncated SVD...")
            n_components = min(2000, tfidf_matrix.shape[0] // 2, tfidf_matrix.shape[1] // 2)
            svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            tfidf_matrix = svd.fit_transform(tfidf_matrix)
            debug_print(f"Reduced matrix shape: {tfidf_matrix.shape}")
        
        # Perform clustering
        n_clusters = max(50, int(len(self.df_working) * 0.001))  # At least 50 clusters
        debug_print(f"Performing K-means clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        
        if TQDM_AVAILABLE:
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            # Add progress bar description
            pbar = tqdm(total=1, desc="Clustering")
            pbar.update(1)
            pbar.close()
        else:
            debug_print("Performing clustering...")
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
        df_with_cluster = self.df_working.copy()
        df_with_cluster['cluster'] = cluster_labels
        
        # Sample from each cluster to maintain diversity
        debug_print("Sampling from each cluster...")
        samples_per_cluster = max(1, int(len(df_with_cluster) * self.sample_ratio / n_clusters))
        sampled_indices = []
        
        cluster_iter = range(n_clusters)
        if TQDM_AVAILABLE:
            cluster_iter = tqdm(cluster_iter, desc="Sampling from clusters")
        else:
            debug_print(f"Sampling from {n_clusters} clusters...")
            
        for cluster_id in cluster_iter:
            cluster_data = df_with_cluster[df_with_cluster['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                n_samples = min(samples_per_cluster, len(cluster_data))
                sampled_indices.extend(
                    cluster_data.sample(n=n_samples, random_state=self.random_state).index.tolist()
                )
                
        # Get the rebalanced dataframe - KEEP ALL COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = df_with_cluster.loc[sampled_indices].copy()
        # Remove the temporary cluster column - KEEP diff_text INTACT
        self.df_rebalanced = self.df_rebalanced.drop('cluster', axis=1)
        
        debug_print(f"Clustering sampling completed. Selected {len(self.df_rebalanced)} samples.")
        
    def _near_duplicate_detection(self):
        """Detect and remove near-duplicate messages."""
        debug_print("Applying near-duplicate detection...")
        
        # For large datasets, process in chunks
        if len(self.df_working) > 50000:
            debug_print("Large dataset detected, using chunked processing...")
            self._near_duplicate_detection_chunked()
            return
            
        # Hash ONLY messages for quick duplicate detection (never process diff_text)
        debug_print("Hashing messages for duplicate detection...")
        if TQDM_AVAILABLE:
            hashes = [hash_text(msg) for msg in tqdm(self.df_working['message'], desc="Hashing messages")]
        else:
            debug_print("Hashing messages...")
            hashes = [hash_text(msg) for msg in self.df_working['message']]
            
        df_with_hash = self.df_working.copy()
        df_with_hash['message_hash'] = hashes
        
        # Remove exact duplicates first
        debug_print("Removing exact duplicates...")
        df_unique = df_with_hash.drop_duplicates(subset=['message_hash'])
        debug_print(f"Removed {len(df_with_hash) - len(df_unique)} exact duplicates")
        
        # Vectorize ONLY messages
        debug_print("Vectorizing messages...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        if TQDM_AVAILABLE:
            tfidf_matrix = vectorizer.fit_transform(
                tqdm(df_unique['message'], desc="Vectorizing messages")
            )
        else:
            debug_print("Vectorizing messages...")
            tfidf_matrix = vectorizer.fit_transform(df_unique['message'])
            
        debug_print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Calculate similarity matrix
        debug_print("Calculating cosine similarities...")
        # Process in smaller batches to manage memory
        batch_size = min(1000, len(df_unique))
        selected_indices = []
        processed_count = 0
        
        batch_iter = range(0, len(df_unique), batch_size)
        if TQDM_AVAILABLE:
            batch_iter = tqdm(batch_iter, desc="Processing batches", total=len(df_unique)//batch_size+1)
            
        for i in batch_iter:
            end_idx = min(i + batch_size, len(df_unique))
            batch_matrix = tfidf_matrix[i:end_idx]
            
            # Calculate similarities with all messages
            similarities = cosine_similarity(batch_matrix, tfidf_matrix)
            
            # Group similar messages and select representatives
            threshold = 0.9  # High threshold to only remove very similar messages
            for j, similarity_row in enumerate(similarities):
                # Find similar messages
                similar_indices = np.where(similarity_row >= threshold)[0]
                
                # Select the first one as representative
                if len(similar_indices) > 0:
                    representative_idx = df_unique.index[similar_indices[0]]
                    if representative_idx not in selected_indices:
                        selected_indices.append(representative_idx)
            
            processed_count += batch_size
            if not TQDM_AVAILABLE:
                debug_print(f"Processed {min(processed_count, len(df_unique))}/{len(df_unique)} messages")
        
        # Get the rebalanced dataframe - KEEP ALL COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = df_with_hash.loc[selected_indices].copy()
        # Remove the temporary hash column - KEEP diff_text INTACT
        self.df_rebalanced = self.df_rebalanced.drop('message_hash', axis=1)
        
        debug_print(f"Near-duplicate detection completed. Selected {len(self.df_rebalanced)} samples.")
        
    def _near_duplicate_detection_chunked(self):
        """Near-duplicate detection for large datasets using chunked processing."""
        debug_print("Using chunked processing for large dataset...")
        
        # Hash ONLY messages for quick duplicate detection (never process diff_text)
        debug_print("Hashing messages for duplicate detection...")
        if TQDM_AVAILABLE:
            hashes = [hash_text(msg) for msg in tqdm(self.df_working['message'], desc="Hashing messages")]
        else:
            debug_print("Hashing messages...")
            hashes = [hash_text(msg) for msg in self.df_working['message']]
            
        df_with_hash = self.df_working.copy()
        df_with_hash['message_hash'] = hashes
        
        # Remove exact duplicates first
        debug_print("Removing exact duplicates...")
        df_unique = df_with_hash.drop_duplicates(subset=['message_hash'])
        debug_print(f"Removed {len(df_with_hash) - len(df_unique)} exact duplicates")
        
        # For very large datasets, sample for similarity detection
        sample_size = min(20000, len(df_unique))
        debug_print(f"Sampling {sample_size} unique messages for similarity detection...")
        df_sample = df_unique.sample(n=sample_size, random_state=self.random_state)
        
        # Vectorize ONLY sampled messages
        debug_print("Vectorizing sampled messages...")
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        
        if TQDM_AVAILABLE:
            tfidf_matrix = vectorizer.fit_transform(
                tqdm(df_sample['message'], desc="Vectorizing messages")
            )
        else:
            debug_print("Vectorizing messages...")
            tfidf_matrix = vectorizer.fit_transform(df_sample['message'])
            
        debug_print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Calculate similarity within sample to identify clusters
        debug_print("Calculating cosine similarities within sample...")
        if TQDM_AVAILABLE:
            # For large matrices, we still need to process without progress bar
            similarity_matrix = cosine_similarity(tfidf_matrix)
        else:
            debug_print("Computing similarity matrix...")
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
        debug_print(f"Similarity matrix shape: {similarity_matrix.shape}")
        
        # Group similar messages and select representatives
        visited = set()
        selected_indices = []
        
        threshold = 0.9  # High threshold to only remove very similar messages
        debug_print(f"Grouping messages with similarity threshold {threshold}...")
        
        row_iter = range(len(similarity_matrix))
        if TQDM_AVAILABLE:
            row_iter = tqdm(row_iter, desc="Grouping similar messages")
            
        for i in row_iter:
            if i not in visited:
                # Find similar messages
                similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
                
                # Mark all similar messages as visited
                for idx in similar_indices:
                    visited.add(idx)
                
                # Select one representative (the first one)
                selected_indices.append(df_sample.index[similar_indices[0]])
        
        # Add remaining messages that weren't in the sample
        sampled_indices = set(df_sample.index)
        remaining_indices = [idx for idx in df_unique.index if idx not in sampled_indices]
        selected_indices.extend(remaining_indices)
        
        # Get the rebalanced dataframe - KEEP ALL COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = df_with_hash.loc[selected_indices].copy()
        # Remove the temporary hash column - KEEP diff_text INTACT
        self.df_rebalanced = self.df_rebalanced.drop('message_hash', axis=1)
        
        debug_print(f"Near-duplicate detection completed. Selected {len(self.df_rebalanced)} samples.")
        
    def _diversity_sampling(self):
        """Implement farthest-point sampling to maximize textual diversity."""
        debug_print("Applying diversity sampling...")
        
        # For large datasets, sample a reasonable subset for processing
        max_process_size = 100000
        if len(self.df_working) > max_process_size:
            debug_print(f"Large dataset detected, sampling {max_process_size} rows for processing...")
            df_process = self.df_working.sample(n=max_process_size, random_state=self.random_state)
        else:
            df_process = self.df_working
            
        # Vectorize ONLY messages
        debug_print("Vectorizing messages...")
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
        
        if TQDM_AVAILABLE:
            tfidf_matrix = vectorizer.fit_transform(
                tqdm(df_process['message'], desc="Vectorizing messages")
            )
        else:
            debug_print("Vectorizing messages...")
            tfidf_matrix = vectorizer.fit_transform(df_process['message'])
            
        debug_print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Farthest point sampling
        debug_print("Performing farthest-point sampling...")
        n_samples = int(len(df_process) * self.sample_ratio)
        n_samples = min(n_samples, len(df_process))
        
        if n_samples >= len(self.df_working):
            # If we're keeping most of the data, just return the original
            self.df_rebalanced = self.df_working.copy()
            debug_print(f"Diversity sampling: keeping all {len(self.df_rebalanced)} samples.")
            return
            
        # Start with a random point
        selected = [np.random.randint(0, len(df_process))]
        remaining = set(range(len(df_process))) - set(selected)
        
        # Iteratively select the farthest point (limited iterations for performance)
        max_iterations = min(n_samples - 1, 1000)  # Limit iterations for performance
        debug_print(f"Selecting {max_iterations} additional points...")
        
        iteration_iter = range(max_iterations)
        if TQDM_AVAILABLE:
            iteration_iter = tqdm(iteration_iter, desc="Selecting diverse samples")
            
        for i in iteration_iter:
            if not remaining:
                break
                
            # Find the point that is farthest from all selected points
            max_min_dist = -1
            farthest_point = None
            
            # Check a subset of remaining points for efficiency
            candidates = list(remaining)[:min(1000, len(remaining))]
            
            candidate_iter = candidates
            if TQDM_AVAILABLE:
                candidate_iter = tqdm(candidates, desc="Evaluating candidates", leave=False)
                
            for candidate in candidate_iter:
                # Calculate minimum distance to selected points
                min_dist = float('inf')
                for sel in selected:
                    # Cosine distance = 1 - cosine similarity
                    dist = 1 - cosine_similarity(tfidf_matrix[candidate], tfidf_matrix[sel])[0][0]
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_point = candidate
            
            if farthest_point is not None:
                selected.append(farthest_point)
                remaining.remove(farthest_point)
        
        # Get the rebalanced dataframe - KEEP ALL COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = df_process.loc[df_process.index[selected]].copy()
        debug_print(f"Diversity sampling completed. Selected {len(self.df_rebalanced)} samples.")
        
    def _frequency_downscaling(self):
        """Downscale based on message frequency."""
        debug_print("Applying frequency-based downscaling...")
        
        # Calculate message frequencies
        debug_print("Calculating message frequencies...")
        message_counts = self.df_working['message'].value_counts()
        
        # Assign weights inversely proportional to frequency
        debug_print("Assigning inverse frequency weights...")
        weights = []
        
        message_iter = self.df_working['message']
        if TQDM_AVAILABLE:
            message_iter = tqdm(message_iter, desc="Calculating weights")
        else:
            debug_print("Calculating weights...")
            
        for msg in message_iter:
            count = message_counts[msg]
            # Use square root to make it less aggressive
            weight = 1.0 / np.sqrt(count) if count > 0 else 1.0
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Sample based on weights
        n_samples = int(len(self.df_working) * self.sample_ratio)
        if self.max_samples:
            n_samples = min(n_samples, self.max_samples)
            
        debug_print(f"Sampling {n_samples} rows based on inverse frequency weights...")
        selected_indices = np.random.choice(
            self.df_working.index, 
            size=min(n_samples, len(self.df_working)), 
            replace=False, 
            p=weights
        )
        
        # Get the rebalanced dataframe - KEEP ALL COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = self.df_working.loc[selected_indices].copy()
        debug_print(f"Frequency downscaling completed. Selected {len(self.df_rebalanced)} samples.")
        
    def _quality_scoring(self):
        """Score messages based on quality criteria."""
        debug_print("Applying quality scoring system...")
        
        # Calculate quality scores based on multiple criteria
        debug_print("Calculating quality scores...")
        
        # Message length score (prefer medium-length messages)
        message_lengths = self.df_working['message'].str.len()
        length_scores = []
        
        length_iter = message_lengths
        if TQDM_AVAILABLE:
            length_iter = tqdm(length_iter, desc="Calculating length scores")
        else:
            debug_print("Calculating length scores...")
            
        for length in length_iter:
            if 20 <= length <= 500:  # Good length range for commit messages
                length_scores.append(1.0)
            elif 10 <= length <= 1000:  # Acceptable length range
                length_scores.append(0.7)
            else:  # Too short or too long
                length_scores.append(0.3)
        
        # Uniqueness score (based on character diversity)
        uniqueness_scores = []
        
        message_iter = self.df_working['message']
        if TQDM_AVAILABLE:
            message_iter = tqdm(message_iter, desc="Calculating uniqueness scores")
        else:
            debug_print("Calculating uniqueness scores...")
            
        for msg in message_iter:
            if len(msg) > 0:
                # Ratio of unique characters to total characters
                unique_chars = len(set(msg.lower()))
                total_chars = len(msg)
                uniqueness_score = unique_chars / total_chars if total_chars > 0 else 0
                uniqueness_scores.append(uniqueness_score)
            else:
                uniqueness_scores.append(0)
        
        # Combine scores
        debug_print("Combining quality scores...")
        length_scores = np.array(length_scores)
        uniqueness_scores = np.array(uniqueness_scores)
        
        # Normalize uniqueness scores
        if np.max(uniqueness_scores) > 0:
            uniqueness_scores = uniqueness_scores / np.max(uniqueness_scores)
        
        # Weighted combination
        combined_scores = 0.7 * length_scores + 0.3 * uniqueness_scores
        
        # Sample based on scores
        n_samples = int(len(self.df_working) * self.sample_ratio)
        if self.max_samples:
            n_samples = min(n_samples, self.max_samples)
            
        debug_print(f"Sampling {n_samples} rows based on quality scores...")
        # Normalize scores to probabilities
        probabilities = combined_scores / np.sum(combined_scores)
        
        selected_indices = np.random.choice(
            self.df_working.index,
            size=min(n_samples, len(self.df_working)),
            replace=False,
            p=probabilities
        )
        
        # Get the rebalanced dataframe - KEEP ALL COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = self.df_working.loc[selected_indices].copy()
        debug_print(f"Quality scoring completed. Selected {len(self.df_rebalanced)} samples.")
        
    def _information_value_scoring(self):
        """Score messages based on information value."""
        debug_print("Applying information value scoring...")
        
        # For large datasets, sample for processing
        max_process_size = 50000
        if len(self.df_working) > max_process_size:
            debug_print(f"Large dataset detected, sampling {max_process_size} rows for processing...")
            df_process = self.df_working.sample(n=max_process_size, random_state=self.random_state)
        else:
            df_process = self.df_working
            
        # Estimate information value using TF-IDF scores
        debug_print("Calculating TF-IDF scores...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        if TQDM_AVAILABLE:
            tfidf_matrix = vectorizer.fit_transform(
                tqdm(df_process['message'], desc="Vectorizing messages")
            )
        else:
            debug_print("Vectorizing messages...")
            tfidf_matrix = vectorizer.fit_transform(df_process['message'])
            
        debug_print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Calculate mean TF-IDF score for each document as information measure
        debug_print("Calculating information scores...")
        information_scores = np.array(np.mean(tfidf_matrix, axis=1)).flatten()
        
        # Sample based on information scores
        n_samples = int(len(df_process) * self.sample_ratio)
        if self.max_samples:
            n_samples = min(n_samples, self.max_samples)
            
        debug_print(f"Sampling {n_samples} rows based on information scores...")
        # Normalize scores to probabilities
        probabilities = information_scores / np.sum(information_scores)
        
        selected_indices = np.random.choice(
            df_process.index,
            size=min(n_samples, len(df_process)),
            replace=False,
            p=probabilities
        )
        
        # Get the rebalanced dataframe - KEEP ALL COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = self.df_working.loc[selected_indices].copy()
        debug_print(f"Information value scoring completed. Selected {len(self.df_rebalanced)} samples.")
        
    def _topic_modeling_approach(self):
        """Apply topic modeling to balance representation across topics."""
        debug_print("Applying topic modeling approach...")
        
        # For large datasets, sample for processing
        max_process_size = 30000
        if len(self.df_working) > max_process_size:
            debug_print(f"Large dataset detected, sampling {max_process_size} rows for processing...")
            df_process = self.df_working.sample(n=max_process_size, random_state=self.random_state)
        else:
            df_process = self.df_working
        
        # Vectorize ONLY messages
        debug_print("Vectorizing messages...")
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
        
        if TQDM_AVAILABLE:
            tfidf_matrix = vectorizer.fit_transform(
                tqdm(df_process['message'], desc="Vectorizing messages")
            )
        else:
            debug_print("Vectorizing messages...")
            tfidf_matrix = vectorizer.fit_transform(df_process['message'])
            
        debug_print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Apply NMF for topic modeling
        from sklearn.decomposition import NMF
        debug_print("Applying NMF for topic modeling...")
        n_topics = min(50, max(10, int(len(df_process) * 0.001)))  # Adaptive number of topics
        nmf = NMF(n_components=n_topics, random_state=self.random_state, max_iter=200)
        topic_weights = nmf.fit_transform(tfidf_matrix)
        debug_print(f"Identified {n_topics} topics")
        
        # Assign dominant topic to each message
        df_process_topics = df_process.copy()
        df_process_topics['dominant_topic'] = np.argmax(topic_weights, axis=1)
        
        # Sample evenly from each topic
        debug_print("Sampling evenly from each topic...")
        samples_per_topic = max(1, int(len(df_process_topics) * self.sample_ratio / n_topics))
        sampled_indices = []
        
        topic_iter = range(n_topics)
        if TQDM_AVAILABLE:
            topic_iter = tqdm(topic_iter, desc="Sampling from topics")
        else:
            debug_print(f"Sampling from {n_topics} topics...")
            
        for topic_id in topic_iter:
            topic_data = df_process_topics[df_process_topics['dominant_topic'] == topic_id]
            if len(topic_data) > 0:
                n_samples = min(samples_per_topic, len(topic_data))
                sampled_indices.extend(
                    topic_data.sample(n=n_samples, random_state=self.random_state).index.tolist()
                )
        
        # Get the rebalanced dataframe - KEEP ALL COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = df_process_topics.loc[sampled_indices].copy()
        # Remove the temporary topic column - KEEP diff_text INTACT
        self.df_rebalanced = self.df_rebalanced.drop('dominant_topic', axis=1)
        
        debug_print(f"Topic modeling approach completed. Selected {len(self.df_rebalanced)} samples.")
        
    def _ngram_grouping(self):
        """Group messages by common n-grams and limit representation."""
        debug_print("Applying n-gram grouping...")
        
        # For large datasets, sample for processing
        max_process_size = 30000
        if len(self.df_working) > max_process_size:
            debug_print(f"Large dataset detected, sampling {max_process_size} rows for processing...")
            df_process = self.df_working.sample(n=max_process_size, random_state=self.random_state)
        else:
            df_process = self.df_working
            
        # Calculate trigrams for each message
        debug_print("Calculating trigrams...")
        
        trigram_iter = df_process['message']
        if TQDM_AVAILABLE:
            trigram_results = []
            for msg in tqdm(trigram_iter, desc="Calculating trigrams"):
                trigram_results.append(calculate_ngrams(msg, 3))
            df_process_trigrams = df_process.copy()
            df_process_trigrams['trigrams'] = trigram_results
        else:
            debug_print("Calculating trigrams...")
            df_process_trigrams = df_process.copy()
            df_process_trigrams['trigrams'] = df_process_trigrams['message'].apply(lambda x: calculate_ngrams(x, 3))
        
        # Create n-gram to message mapping
        debug_print("Creating n-gram to message mapping...")
        ngram_to_messages = defaultdict(list)
        
        mapping_iter = df_process_trigrams.iterrows()
        if TQDM_AVAILABLE:
            mapping_iter = tqdm(mapping_iter, total=len(df_process_trigrams), desc="Mapping n-grams")
            
        for idx, row in mapping_iter:
            for ngram in row['trigrams']:
                ngram_to_messages[ngram].append(idx)
        
        # Limit representation of each n-gram group
        debug_print("Limiting representation of n-gram groups...")
        max_per_ngram = max(1, int(self.sample_ratio * 10))  # Allow more samples per n-gram
        selected_indices = set()
        
        ngram_iter = ngram_to_messages.items()
        if TQDM_AVAILABLE:
            ngram_iter = tqdm(ngram_iter, desc="Processing n-gram groups")
            
        for ngram, message_indices in ngram_iter:
            # Limit the number of messages per n-gram
            if len(message_indices) > max_per_ngram:
                sampled = np.random.choice(message_indices, size=max_per_ngram, replace=False)
                selected_indices.update(sampled)
            else:
                selected_indices.update(message_indices)
        
        # Convert to list and limit total samples
        selected_indices = list(selected_indices)
        n_samples = int(len(df_process_trigrams) * self.sample_ratio)
        if self.max_samples:
            n_samples = min(n_samples, self.max_samples)
            
        if len(selected_indices) > n_samples:
            selected_indices = np.random.choice(selected_indices, size=min(n_samples, len(selected_indices)), replace=False)
        
        # Get the rebalanced dataframe - KEEP ALL COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = df_process_trigrams.loc[selected_indices].copy()
        # Remove the temporary trigrams column - KEEP diff_text INTACT
        self.df_rebalanced = self.df_rebalanced.drop('trigrams', axis=1)
        
        debug_print(f"N-gram grouping completed. Selected {len(self.df_rebalanced)} samples.")
        
    def _multi_criteria_balancing(self):
        """Combine multiple metrics for balancing."""
        debug_print("Applying multi-criteria balancing...")
        
        # For large datasets, sample for processing
        max_process_size = 50000
        if len(self.df_working) > max_process_size:
            debug_print(f"Large dataset detected, sampling {max_process_size} rows for processing...")
            df_process = self.df_working.sample(n=max_process_size, random_state=self.random_state)
        else:
            df_process = self.df_working
            
        # Collect multiple scores
        debug_print("Collecting multiple criteria scores...")
        
        # 1. Frequency-based score
        debug_print("Calculating frequency scores...")
        message_counts = df_process['message'].value_counts()
        freq_scores = []
        
        freq_iter = df_process['message']
        if TQDM_AVAILABLE:
            freq_iter = tqdm(freq_iter, desc="Calculating frequency scores")
        else:
            debug_print("Calculating frequency scores...")
            
        for msg in freq_iter:
            count = message_counts[msg]
            freq_score = 1.0 / np.sqrt(count) if count > 0 else 1.0  # Less aggressive than pure inverse
            freq_scores.append(freq_score)
        
        # 2. Length-based score
        debug_print("Calculating length scores...")
        message_lengths = df_process['message'].str.len()
        length_scores = []
        
        length_iter = message_lengths
        if TQDM_AVAILABLE:
            length_iter = tqdm(length_iter, desc="Calculating length scores")
        else:
            debug_print("Calculating length scores...")
            
        for length in length_iter:
            if 20 <= length <= 500:
                length_scores.append(1.0)
            elif 10 <= length <= 1000:
                length_scores.append(0.5)
            else:
                length_scores.append(0.2)
        
        # 3. Diversity score (approximated with character uniqueness)
        debug_print("Calculating diversity scores...")
        diversity_scores = []
        
        diversity_iter = df_process['message']
        if TQDM_AVAILABLE:
            diversity_iter = tqdm(diversity_iter, desc="Calculating diversity scores")
        else:
            debug_print("Calculating diversity scores...")
            
        for msg in diversity_iter:
            if len(msg) > 0:
                unique_chars = len(set(msg.lower()))
                total_chars = len(msg)
                diversity_score = unique_chars / total_chars if total_chars > 0 else 0
                diversity_scores.append(diversity_score)
            else:
                diversity_scores.append(0)
        
        # Combine scores with weights
        debug_print("Combining scores...")
        freq_scores = np.array(freq_scores)
        length_scores = np.array(length_scores)
        diversity_scores = np.array(diversity_scores)
        
        # Normalize scores
        if np.max(freq_scores) > 0:
            freq_scores = freq_scores / np.max(freq_scores)
        if np.max(diversity_scores) > 0:
            diversity_scores = diversity_scores / np.max(diversity_scores)
            
        # Weighted combination (adjust weights as needed)
        combined_scores = (
            0.5 * freq_scores + 
            0.3 * length_scores + 
            0.2 * diversity_scores
        )
        
        # Sample based on combined scores
        n_samples = int(len(df_process) * self.sample_ratio)
        if self.max_samples:
            n_samples = min(n_samples, self.max_samples)
            
        debug_print(f"Sampling {n_samples} rows based on combined scores...")
        # Normalize to probabilities
        if np.sum(combined_scores) > 0:
            probabilities = combined_scores / np.sum(combined_scores)
        else:
            probabilities = np.ones(len(combined_scores)) / len(combined_scores)
        
        selected_indices = np.random.choice(
            df_process.index,
            size=min(n_samples, len(df_process)),
            replace=False,
            p=probabilities
        )
        
        # Get the rebalanced dataframe - KEEP ALL COLUMNS INCLUDING diff_text INTACT
        self.df_rebalanced = self.df_working.loc[selected_indices].copy()
        debug_print(f"Multi-criteria balancing completed. Selected {len(self.df_rebalanced)} samples.")
        
    def visualize_after(self):
        """Create visualizations of the rebalanced dataset."""
        debug_print("Creating visualizations of rebalanced dataset...")
        
        # Message length distribution
        debug_print("Calculating message lengths...")
        orig_lengths = self.df_original['message'].str.len()
        rebalanced_lengths = self.df_rebalanced['message'].str.len()
        
        # Plot comparison
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(orig_lengths, bins=50, alpha=0.7, color='blue', label='Original')
        plt.hist(rebalanced_lengths, bins=50, alpha=0.7, color='red', label='Rebalanced')
        plt.xlabel('Message Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Message Length Distribution Comparison')
        plt.yscale('log')
        plt.legend()
        
        # Top frequent messages comparison
        debug_print("Finding top frequent messages...")
        orig_message_counts = self.df_original['message'].value_counts().head(15)
        rebalanced_message_counts = self.df_rebalanced['message'].value_counts().head(15)
        
        plt.subplot(1, 3, 2)
        orig_message_counts.plot(kind='barh', color='blue', alpha=0.7, label='Original')
        plt.xlabel('Frequency')
        plt.title('Top 15 Messages - Original')
        plt.gca().invert_yaxis()
        
        plt.subplot(1, 3, 3)
        rebalanced_message_counts.plot(kind='barh', color='red', alpha=0.7, label='Rebalanced')
        plt.xlabel('Frequency')
        plt.title('Top 15 Messages - Rebalanced')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('dataset_rebalancing_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        debug_print("Saved comparison visualization as dataset_rebalancing_comparison.png")
        
    def save_results(self):
        """Save the rebalanced dataset."""
        debug_print(f"Saving rebalanced dataset to {self.output_file}...")
        # Save the rebalanced dataset with both message and diff_text columns intact
        self.df_rebalanced.to_parquet(self.output_file, index=False)
        debug_print(f"Saved {len(self.df_rebalanced)} rows to {self.output_file}")
        
    def print_summary(self):
        """Print summary statistics."""
        debug_print("=== REBALANCING SUMMARY ===")
        debug_print(f"Strategy used: {self.strategy}")
        debug_print(f"Original dataset size: {len(self.df_original)}")
        debug_print(f"Rebalanced dataset size: {len(self.df_rebalanced)}")
        debug_print(f"Reduction ratio: {len(self.df_rebalanced)/len(self.df_original):.2%}")
        
        # Show some example messages
        debug_print("\nExample messages from original dataset:")
        for i, msg in enumerate(self.df_original['message'].head(3)):
            debug_print(f"  {i+1}. {msg[:100]}{'...' if len(msg) > 100 else ''}")
            
        debug_print("\nExample messages from rebalanced dataset:")
        for i, msg in enumerate(self.df_rebalanced['message'].head(3)):
            debug_print(f"  {i+1}. {msg[:100]}{'...' if len(msg) > 100 else ''}")

def main():
    parser = argparse.ArgumentParser(description='Rebalance dataset based on various strategies')
    parser.add_argument('input_file', help='Input parquet file path')
    parser.add_argument('output_file', help='Output parquet file path')
    parser.add_argument('--strategy', choices=[
        'remove_duplicates', 'clustering', 'near_duplicate', 'diversity', 'frequency_downscale',
        'quality_scoring', 'information_value', 'topic_modeling', 
        'ngram_grouping', 'multi_criteria'
    ], default='remove_duplicates', help='Rebalancing strategy to use')
    parser.add_argument('--sample_ratio', type=float, default=1.0, 
                        help='Ratio of samples to keep (0.0 to 1.0)')
    parser.add_argument('--max_samples', type=int, 
                        help='Maximum number of samples to keep')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='Random state for reproducibility')
    parser.add_argument('--keep_vague', action='store_true',
                        help='Keep vague messages (by default they are removed)')
    parser.add_argument('--no_multiprocessing', action='store_true',
                        help='Disable multiprocessing')
    parser.add_argument('--no_debug', action='store_true', 
                        help='Disable debug printing')
    
    args = parser.parse_args()
    
    global DEBUG
    DEBUG = not args.no_debug
    
    # Validate inputs
    if not os.path.exists(args.input_file):
        debug_print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)
        
    if args.sample_ratio <= 0 or args.sample_ratio > 1:
        debug_print("Error: sample_ratio must be between 0 and 1")
        sys.exit(1)
        
    if args.max_samples and args.max_samples <= 0:
        debug_print("Error: max_samples must be positive")
        sys.exit(1)
    
    # Create rebalancer and run
    rebalancer = DatasetRebalancer(
        input_file=args.input_file,
        output_file=args.output_file,
        strategy=args.strategy,
        sample_ratio=args.sample_ratio,
        max_samples=args.max_samples,
        random_state=args.random_state,
        remove_vague=not args.keep_vague,
        use_multiprocessing=not args.no_multiprocessing
    )
    
    try:
        rebalancer.load_data()
        rebalancer.visualize_before()
        rebalancer.rebalance()
        rebalancer.visualize_after()
        rebalancer.save_results()
        rebalancer.print_summary()
    except Exception as e:
        debug_print(f"Error during rebalancing: {str(e)}")
        raise e

if __name__ == "__main__":
    main()