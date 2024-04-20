import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


def visualization(df, target, path_to_save):
    target_plot(df, target, path_to_save)
    heatmap_on_target(df, target, path_to_save)
    create_violin_plots_for_each_feature(df, target, path_to_save)
    create_histograms_and_density_plots(df, target, path_to_save)


def target_plot(data, target, path_to_save):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=target, data=data)
    plt.title('Distribution of Target Outcomes')
    plt.xlabel('Target Outcome')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    if path_to_save:
        plt.savefig(f"{path_to_save}/target_plot.png")
    else:
        plt.show()
    plt.close()


def heatmap_on_target(df, target, path_to_save):
    data = df.copy()
    label_encoder = LabelEncoder()
    data[target] = label_encoder.fit_transform(data[target])
    corr_matrix_with_target = data.corr(method='pearson')
    correlations_with_target_only = corr_matrix_with_target[target].drop(target)
    correlations_with_target_df = pd.DataFrame(correlations_with_target_only).T
    correlations_sorted_abs = correlations_with_target_df.T.abs().sort_values(by=target, ascending=False)
    plt.figure(figsize=(4, 20), dpi=300)  # Increased dpi for higher resolution
    sns.heatmap(correlations_sorted_abs, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=.5)
    plt.title('Correlation of Features with Target (Sorted by Absolute Value)')
    plt.yticks(rotation=0)
    if path_to_save:
        plt.savefig(f"{path_to_save}/heatmap_target.png", bbox_inches='tight', pad_inches=0.1,
                    dpi=300)  # Save with high dpi
    else:
        plt.show()
    plt.close()
    print("Created heatmap on target")


def create_violin_plots_for_each_feature(data, target, path_to_save):
    for column in data.select_dtypes(include=[np.number]).columns:
        # We skip the target itself if it's encoded as a number
        if column == target:
            continue

        plt.figure(figsize=(10, 6))
        # Set 'x' to target, 'y' to the current column, and use 'hue' for the same coloring effect
        sns.violinplot(x=target, y=column, data=data, hue=target, inner='quart', palette='pastel', legend=False)
        plt.title(f'Violin Plot of {column} by {target}')
        # Since we're using 'hue', there will be a legend. We remove it with the following line.
        plt.legend([], [], frameon=False)
        plt.savefig(f"{path_to_save}/{column}_violin_plot.png")
        plt.close()
        print(f"Created violin plot for {column} by {target}")


def create_histograms_and_density_plots(data, target, path_to_save):
    for column in data.select_dtypes(include=[np.number]).columns:
        # Skip the target if it's numeric
        if column == target:
            continue
        plt.figure(figsize=(8, 6))
        # Plot the density for each class of the target variable, using `fill` instead of `shade`
        for class_label in data[target].unique():
            subset = data[data[target] == class_label]
            sns.kdeplot(subset[column], fill=True, label=str(class_label))  # Updated to use `fill`

        plt.title(f'Density Plot of {column} by {target}')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend(title=target)

        # Save the plot to the specified path
        plt.savefig(os.path.join(path_to_save, f"{column}_density_by_{target}.png"))
        plt.close()  # Close the figure to conserve memory
        print(f"Created density plot for {column} by {target}")


def create_importance_of_feature(top_features, path_to_save):
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, y='Feature', x='Importance', orient='h', palette='coolwarm')
    plt.title('Top 20 Most Important Features in Predicting Match Outcomes')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    if path_to_save:
        plt.savefig(f"{path_to_save}/importance_of_feature.png", bbox_inches='tight', pad_inches=0.1,
                    dpi=300)  # Save with high dpi
    else:
        plt.show()