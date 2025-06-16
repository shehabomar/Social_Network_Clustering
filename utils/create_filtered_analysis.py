#!/usr/bin/env python3

"""
Community Outlier Filtering and Cleaned Analysis
This script filters out outlier communities based on statistical criteria and creates cleaned analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse

def load_community_analysis(analysis_file):
    """Load community analysis results"""
    return pd.read_csv(analysis_file)

def identify_outliers(df, criteria):
    """
    Identify outlier communities based on multiple criteria
    
    Args:
        df: DataFrame with community analysis
        criteria: Dictionary with outlier criteria
    """
    outliers = pd.Series(False, index=df.index)
    outlier_reasons = pd.Series('', index=df.index)
    
    # Size-based filtering
    if 'min_size' in criteria:
        size_outliers = df['size'] < criteria['min_size']
        outliers |= size_outliers
        outlier_reasons[size_outliers] += f"size<{criteria['min_size']}; "
    
    # Conductance-based filtering
    if 'max_conductance' in criteria:
        conductance_outliers = df['conductance'] > criteria['max_conductance']
        outliers |= conductance_outliers
        outlier_reasons[conductance_outliers] += f"conductance>{criteria['max_conductance']}; "
    
    # Statistical outliers based on average degree
    if 'degree_zscore_threshold' in criteria:
        degree_zscores = np.abs((df['avg_degree'] - df['avg_degree'].mean()) / df['avg_degree'].std())
        degree_outliers = degree_zscores > criteria['degree_zscore_threshold']
        outliers |= degree_outliers
        outlier_reasons[degree_outliers] += f"degree_zscore>{criteria['degree_zscore_threshold']}; "
    
    # Density-based filtering
    if 'min_density' in criteria:
        density_outliers = df['density'] < criteria['min_density']
        outliers |= density_outliers
        outlier_reasons[density_outliers] += f"density<{criteria['min_density']}; "
    
    return outliers, outlier_reasons

def create_before_after_comparison(df_original, df_filtered, output_dir):
    """Create before/after comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Before vs After Outlier Removal', fontsize=16, fontweight='bold')
    
    # Size distribution
    ax = axes[0, 0]
    ax.hist(df_original['size'], bins=50, alpha=0.7, label='Before', color='red', density=True)
    ax.hist(df_filtered['size'], bins=50, alpha=0.7, label='After', color='green', density=True)
    ax.set_xlabel('Community Size')
    ax.set_ylabel('Density')
    ax.set_title('Size Distribution')
    ax.legend()
    ax.set_yscale('log')
    
    # Conductance distribution
    ax = axes[0, 1]
    ax.hist(df_original['conductance'], bins=50, alpha=0.7, label='Before', color='red', density=True)
    ax.hist(df_filtered['conductance'], bins=50, alpha=0.7, label='After', color='green', density=True)
    ax.set_xlabel('Conductance')
    ax.set_ylabel('Density')
    ax.set_title('Conductance Distribution')
    ax.legend()
    
    # Average degree distribution
    ax = axes[0, 2]
    ax.hist(df_original['avg_degree'], bins=50, alpha=0.7, label='Before', color='red', density=True)
    ax.hist(df_filtered['avg_degree'], bins=50, alpha=0.7, label='After', color='green', density=True)
    ax.set_xlabel('Average Degree')
    ax.set_ylabel('Density')
    ax.set_title('Average Degree Distribution')
    ax.legend()
    
    # Density distribution
    ax = axes[1, 0]
    ax.hist(df_original['density'], bins=50, alpha=0.7, label='Before', color='red', density=True)
    ax.hist(df_filtered['density'], bins=50, alpha=0.7, label='After', color='green', density=True)
    ax.set_xlabel('Community Density')
    ax.set_ylabel('Frequency Density')
    ax.set_title('Community Density Distribution')
    ax.legend()
    
    # Box plots comparison
    ax = axes[1, 1]
    data_before = [df_original['size'], df_original['conductance'], df_original['avg_degree']]
    data_after = [df_filtered['size'], df_filtered['conductance'], df_filtered['avg_degree']]
    
    positions_before = [1, 3, 5]
    positions_after = [2, 4, 6]
    
    bp1 = ax.boxplot(data_before, positions=positions_before, widths=0.6, 
                     patch_artist=True, boxprops=dict(facecolor='red', alpha=0.7))
    bp2 = ax.boxplot(data_after, positions=positions_after, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor='green', alpha=0.7))
    
    ax.set_xticklabels(['Size', 'Conductance', 'Avg Degree'])
    ax.set_xticks([1.5, 3.5, 5.5])
    ax.set_title('Metrics Comparison (Box Plots)')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Before', 'After'])
    
    # Statistics table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate statistics
    stats_data = [
        ['Metric', 'Before', 'After', 'Change'],
        ['Communities', len(df_original), len(df_filtered), f'{len(df_filtered) - len(df_original):+d}'],
        ['Avg Size', f'{df_original["size"].mean():.1f}', f'{df_filtered["size"].mean():.1f}', 
         f'{df_filtered["size"].mean() - df_original["size"].mean():+.1f}'],
        ['Avg Conductance', f'{df_original["conductance"].mean():.3f}', f'{df_filtered["conductance"].mean():.3f}',
         f'{df_filtered["conductance"].mean() - df_original["conductance"].mean():+.3f}'],
        ['Avg Degree', f'{df_original["avg_degree"].mean():.1f}', f'{df_filtered["avg_degree"].mean():.1f}',
         f'{df_filtered["avg_degree"].mean() - df_original["avg_degree"].mean():+.1f}'],
        ['Total MOFs', df_original['size'].sum(), df_filtered['size'].sum(),
         f'{df_filtered["size"].sum() - df_original["size"].sum():+d}']
    ]
    
    table = ax.table(cellText=stats_data[1:], colLabels=stats_data[0],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'before_after_comparison.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_outlier_analysis_report(df_original, df_filtered, outliers_df, criteria, output_file):
    """Create a detailed outlier analysis report"""
    
    report = f"""
# Community Outlier Analysis Report

## Filtering Criteria Applied
"""
    
    for criterion, value in criteria.items():
        report += f"- **{criterion}**: {value}\n"
    
    report += f"""

## Summary Statistics

### Before Filtering
- **Total Communities**: {len(df_original)}
- **Total MOFs**: {df_original['size'].sum():,}
- **Average Community Size**: {df_original['size'].mean():.1f}
- **Average Conductance**: {df_original['conductance'].mean():.4f}
- **Average Degree**: {df_original['avg_degree'].mean():.2f}
- **Average Density**: {df_original['density'].mean():.4f}

### After Filtering
- **Total Communities**: {len(df_filtered)}
- **Total MOFs**: {df_filtered['size'].sum():,}
- **Average Community Size**: {df_filtered['size'].mean():.1f}
- **Average Conductance**: {df_filtered['conductance'].mean():.4f}
- **Average Degree**: {df_filtered['avg_degree'].mean():.2f}
- **Average Density**: {df_filtered['density'].mean():.4f}

### Filtering Impact
- **Communities Removed**: {len(outliers_df)} ({100 * len(outliers_df) / len(df_original):.1f}%)
- **MOFs Removed**: {outliers_df['size'].sum():,} ({100 * outliers_df['size'].sum() / df_original['size'].sum():.1f}%)
- **Quality Improvement**:
  - Conductance: {df_original['conductance'].mean():.4f} → {df_filtered['conductance'].mean():.4f} ({100 * (df_original['conductance'].mean() - df_filtered['conductance'].mean()) / df_original['conductance'].mean():+.1f}%)
  - Density: {df_original['density'].mean():.4f} → {df_filtered['density'].mean():.4f} ({100 * (df_filtered['density'].mean() - df_original['density'].mean()) / df_original['density'].mean():+.1f}%)

## Outlier Communities Analysis

### Top 10 Largest Removed Communities
"""
    
    top_removed = outliers_df.nlargest(10, 'size')
    for i, (_, row) in enumerate(top_removed.iterrows()):
        report += f"{i+1:2d}. Community {row['community_id']}: {row['size']:,} MOFs, Conductance: {row['conductance']:.3f}, Reasons: {row['outlier_reasons']}\n"
    
    # Outlier reasons analysis
    reason_counts = {}
    for reasons in outliers_df['outlier_reasons']:
        for reason in reasons.split(';'):
            reason = reason.strip()
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    report += f"""

### Outlier Reasons Breakdown
"""
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        report += f"- **{reason}**: {count} communities ({100 * count / len(outliers_df):.1f}% of outliers)\n"
    
    report += f"""

## Quality Metrics Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Average Conductance | {df_original['conductance'].mean():.4f} | {df_filtered['conductance'].mean():.4f} | {100 * (df_original['conductance'].mean() - df_filtered['conductance'].mean()) / df_original['conductance'].mean():+.1f}% |
| Average Density | {df_original['density'].mean():.4f} | {df_filtered['density'].mean():.4f} | {100 * (df_filtered['density'].mean() - df_original['density'].mean()) / df_original['density'].mean():+.1f}% |
| Std Conductance | {df_original['conductance'].std():.4f} | {df_filtered['conductance'].std():.4f} | {100 * (df_original['conductance'].std() - df_filtered['conductance'].std()) / df_original['conductance'].std():+.1f}% |
| Std Density | {df_original['density'].std():.4f} | {df_filtered['density'].std():.4f} | {100 * (df_original['density'].std() - df_filtered['density'].std()) / df_original['density'].std():+.1f}% |

## Recommendations

1. **Use Filtered Dataset**: The filtered dataset shows improved quality metrics with {100 * (df_original['conductance'].mean() - df_filtered['conductance'].mean()) / df_original['conductance'].mean():.1f}% better conductance
2. **Focus on Large Communities**: {len(df_filtered[df_filtered['size'] >= 100])} communities have ≥100 MOFs and represent high-quality clusters
3. **Quality Threshold**: Communities with conductance < {df_filtered['conductance'].quantile(0.25):.3f} represent the highest quality clusters
4. **Statistical Confidence**: The filtered dataset reduces statistical variance and improves analysis reliability

## Technical Notes

- **Size threshold**: Removes very small communities that may not be statistically significant
- **Conductance threshold**: Removes poorly separated communities (high inter-community connectivity)
- **Degree outliers**: Removes communities with unusual connectivity patterns
- **Density threshold**: Ensures communities have meaningful internal structure
"""
    
    with open(output_file, 'w') as f:
        f.write(report)

def process_threshold_algorithm_combination(results_dir, threshold, algorithm, criteria):
    """Process a single threshold-algorithm combination"""
    
    alg_dir = os.path.join(results_dir, 'threshold_analysis', f'{algorithm}_t{threshold}')
    analysis_file = os.path.join(alg_dir, 'detailed_community_analysis.csv')
    
    if not os.path.exists(analysis_file):
        return None
    
    # Load original analysis
    df_original = load_community_analysis(analysis_file)
    
    # Identify outliers
    outliers_mask, outlier_reasons = identify_outliers(df_original, criteria)
    
    # Create filtered dataset
    df_filtered = df_original[~outliers_mask].copy()
    df_outliers = df_original[outliers_mask].copy()
    df_outliers['outlier_reasons'] = outlier_reasons[outliers_mask]
    
    # Create filtered results directory
    filtered_dir = os.path.join(results_dir, 'filtered_results', f'{algorithm}_t{threshold}')
    os.makedirs(filtered_dir, exist_ok=True)
    
    # Save filtered results
    df_filtered.to_csv(os.path.join(filtered_dir, 'filtered_community_analysis.csv'), index=False)
    df_outliers.to_csv(os.path.join(filtered_dir, 'outlier_communities.csv'), index=False)
    
    # Create visualizations
    create_before_after_comparison(df_original, df_filtered, filtered_dir)
    
    # Create report
    create_outlier_analysis_report(df_original, df_filtered, df_outliers, criteria,
                                  os.path.join(filtered_dir, 'outlier_analysis_report.md'))
    
    # Return summary statistics
    return {
        'threshold': threshold,
        'algorithm': algorithm,
        'original_communities': len(df_original),
        'filtered_communities': len(df_filtered),
        'removed_communities': len(df_outliers),
        'original_mofs': df_original['size'].sum(),
        'filtered_mofs': df_filtered['size'].sum(),
        'removed_mofs': df_outliers['size'].sum(),
        'conductance_improvement': (df_original['conductance'].mean() - df_filtered['conductance'].mean()) / df_original['conductance'].mean(),
        'density_improvement': (df_filtered['density'].mean() - df_original['density'].mean()) / df_original['density'].mean()
    }

def create_comprehensive_filtering_summary(filtering_results, output_dir):
    """Create comprehensive summary of all filtering results"""
    
    if not filtering_results:
        return
    
    df_summary = pd.DataFrame(filtering_results)
    
    # Save summary table
    summary_file = os.path.join(output_dir, 'filtering_summary.csv')
    df_summary.to_csv(summary_file, index=False)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Community Filtering Results Summary', fontsize=16, fontweight='bold')
    
    # Communities removed by threshold and algorithm
    ax = axes[0, 0]
    pivot_removed = df_summary.pivot(index='algorithm', columns='threshold', values='removed_communities')
    sns.heatmap(pivot_removed, annot=True, fmt='d', cmap='Reds', ax=ax)
    ax.set_title('Communities Removed')
    
    # MOFs removed by threshold and algorithm
    ax = axes[0, 1]
    pivot_mofs = df_summary.pivot(index='algorithm', columns='threshold', values='removed_mofs')
    sns.heatmap(pivot_mofs, annot=True, fmt='d', cmap='Oranges', ax=ax)
    ax.set_title('MOFs Removed')
    
    # Conductance improvement
    ax = axes[1, 0]
    pivot_conductance = df_summary.pivot(index='algorithm', columns='threshold', values='conductance_improvement')
    sns.heatmap(pivot_conductance, annot=True, fmt='.3f', cmap='Greens', ax=ax)
    ax.set_title('Conductance Improvement')
    
    # Density improvement
    ax = axes[1, 1]
    pivot_density = df_summary.pivot(index='algorithm', columns='threshold', values='density_improvement')
    sns.heatmap(pivot_density, annot=True, fmt='.3f', cmap='Blues', ax=ax)
    ax.set_title('Density Improvement')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filtering_summary_heatmaps.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Filtering summary saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Community Outlier Filtering and Analysis')
    parser.add_argument('--results_dir', required=True,
                       help='Directory containing analysis results')
    parser.add_argument('--thresholds', nargs='+', type=float,
                       default=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                       help='Thresholds to process')
    parser.add_argument('--algorithms', nargs='+', choices=['louvain', 'girvan_newman'],
                       default=['louvain', 'girvan_newman'],
                       help='Algorithms to process')
    parser.add_argument('--min_size', type=int, default=10,
                       help='Minimum community size')
    parser.add_argument('--max_conductance', type=float, default=0.1,
                       help='Maximum conductance')
    parser.add_argument('--degree_zscore_threshold', type=float, default=2.0,
                       help='Z-score threshold for degree outliers')
    parser.add_argument('--min_density', type=float, default=0.01,
                       help='Minimum community density')
    
    args = parser.parse_args()
    
    # Define filtering criteria
    criteria = {
        'min_size': args.min_size,
        'max_conductance': args.max_conductance,
        'degree_zscore_threshold': args.degree_zscore_threshold,
        'min_density': args.min_density
    }
    
    print("="*80)
    print("COMMUNITY OUTLIER FILTERING ANALYSIS")
    print("="*80)
    print(f"Processing results from: {args.results_dir}")
    print(f"Filtering criteria: {criteria}")
    
    # Create filtered results directory
    filtered_base_dir = os.path.join(args.results_dir, 'filtered_results')
    os.makedirs(filtered_base_dir, exist_ok=True)
    
    # Process each threshold-algorithm combination
    filtering_results = []
    
    for threshold in args.thresholds:
        for algorithm in args.algorithms:
            print(f"\nProcessing {algorithm} threshold {threshold}...")
            
            result = process_threshold_algorithm_combination(
                args.results_dir, threshold, algorithm, criteria
            )
            
            if result:
                filtering_results.append(result)
                print(f"  Removed {result['removed_communities']} communities ({result['removed_mofs']} MOFs)")
                print(f"  Conductance improved by {100*result['conductance_improvement']:.1f}%")
    
    # Create comprehensive summary
    create_comprehensive_filtering_summary(filtering_results, filtered_base_dir)
    
    # Save criteria used
    with open(os.path.join(filtered_base_dir, 'filtering_criteria.json'), 'w') as f:
        json.dump(criteria, f, indent=2)
    
    print(f"\n{'='*80}")
    print("FILTERING ANALYSIS COMPLETED")
    print("="*80)
    print(f"Results saved to: {filtered_base_dir}")
    print("Generated files:")
    print("  - filtering_summary.csv: Summary of all filtering results")
    print("  - filtering_summary_heatmaps.png: Visual summary")
    print("  - [algorithm]_t[threshold]/: Individual filtered results")
    print("  - filtering_criteria.json: Criteria used for filtering")

if __name__ == "__main__":
    main() 