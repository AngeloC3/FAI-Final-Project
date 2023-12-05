import matplotlib.pyplot as plt
import numpy as np

# Provided data
configurations = [
    "Base all-mpnet-base-v2", "Max Sum MegaSmall-Candidates all-mpnet-base-v2", "Base all-MiniLM-L6-v2",
    "Max Sum MegaSmall-Candidates all-MiniLM-L6-v2", "MMR Low Diversity all-mpnet-base-v2",
    "MMR Low Diversity all-MiniLM-L6-v2", "Base Universal-Sentence-Encoder", "Max Sum MegaSmall-Candidates Universal-Sentence-Encoder",
    "MMR Mid Diversity all-mpnet-base-v2", "Max Sum Low-Candidates all-mpnet-base-v2", "Max Sum Low-Candidates all-MiniLM-L6-v2",
    "MMR Mid Diversity all-MiniLM-L6-v2", "Max Sum Mid-Candidates all-mpnet-base-v2", "MMR Low Diversity Universal-Sentence-Encoder",
    "Max Sum Mid-Candidates all-MiniLM-L6-v2", "Max Sum Low-Candidates Universal-Sentence-Encoder",
    "Max Sum High-Candidates all-mpnet-base-v2", "Max Sum High-Candidates all-MiniLM-L6-v2",
    "Max Sum Mid-Candidates Universal-Sentence-Encoder", "Max Sum High-Candidates Universal-Sentence-Encoder",
    "MMR High Diversity all-MiniLM-L6-v2", "MMR High Diversity all-mpnet-base-v2", "MMR Mid Diversity Universal-Sentence-Encoder",
    "Max Sum Mid-Candidates Fasttext-wn-sw-300", "Base Fasttext-wn-sw-300", "Max Sum MegaSmall-Candidates Fasttext-wn-sw-300",
    "Max Sum Low-Candidates Fasttext-wn-sw-300", "MMR Low Diversity Fasttext-wn-sw-300",
    "MMR High Diversity Universal-Sentence-Encoder", "Max Sum High-Candidates Fasttext-wn-sw-300",
    "MMR Mid Diversity Fasttext-wn-sw-300", "MMR High Diversity Fasttext-wn-sw-300"
]

scores = [
    71.13, 71.13, 70.63, 70.63, 69.98, 68.99, 66.68, 66.68, 66.55, 66.20,
    65.38, 64.67, 63.77, 63.22, 63.16, 61.50, 61.46, 60.28, 59.04, 56.90,
    56.76, 55.60, 55.57, 50.94, 50.83, 50.83, 50.63, 50.55, 50.17, 49.94,
    48.15, 39.76
]

# Grouping configurations based on prefix
suffixes = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "Universal-Sentence-Encoder", "Fasttext-wn-sw-300"]


# Grouping scores

def get_model_avg(prefix, suffix, configurations=configurations):
    result = [score for config, score in zip(configurations, scores) if
              config.startswith(prefix) and config.endswith(suffix)]
    try:
        return sum(result) / len(result)
    except ZeroDivisionError:
        print(prefix, suffix)


base_avgs = [get_model_avg("Base", suffix) for suffix in suffixes]
mmr_avgs = [get_model_avg("MMR", suffix) for suffix in suffixes]
max_avgs = [get_model_avg("Max", suffix) for suffix in suffixes]

bar_width = 0.25
bar_positions = np.arange(len(suffixes))

# Create bar plots
plt.bar(bar_positions - bar_width, base_avgs, width=bar_width, label='Base', color='grey')
plt.bar(bar_positions, mmr_avgs, width=bar_width, label='MMR', color='lightcoral')
plt.bar(bar_positions + bar_width, max_avgs, width=bar_width, label='Max Sum', color='deepskyblue')

# Customize the plot
plt.xlabel('Models')
plt.ylabel('Average Scores')
plt.title('Average Scores for Different Models')

# Rotate x-labels
plt.xticks(bar_positions, suffixes, rotation=10, ha="right")
plt.tight_layout()

plt.legend()
plt.savefig("KeyBert/images/model_scores.png")

# Show the plot
plt.savefig("KeyBert/images/model_scores.png")

setups = ["Base",
          "Max Sum Low-Candidates", "Max Sum Mid-Candidates", "Max Sum High-Candidates",
          "MMR Low Diversity", "MMR Mid Diversity", "MMR High Diversity"]


def get_setup_avg(setup, configurations=configurations):
    result = [score for config, score in zip(configurations, scores) if setup in config]
    return sum(result) / len(result)


setup_scores = [get_setup_avg(setup) for setup in setups]

# Sort setups and scores in descending order
sorted_indices = np.argsort(setup_scores)[::-1]
setups = [setups[i] for i in sorted_indices]
setup_scores = [setup_scores[i] for i in sorted_indices]

plt.figure(figsize=(12, 6))

# Bar positions
bar_positions = np.arange(len(setups))

# Create bar plot
bar_colors = ['lightcoral' if 'MMR' in setup else 'deepskyblue' if 'Max Sum' in setup else 'grey' for setup in setups]
plt.bar(bar_positions, setup_scores, color=bar_colors)

# Customize the plot
plt.xlabel('Setups')
plt.ylabel('Average Scores')
plt.title('Average Scores for Different Setups')
plt.xticks(bar_positions, setups, rotation=10, ha='right')

# Show the plot
plt.tight_layout()
plt.savefig("KeyBert/images/setup_scores.png")

gensim_scores = [score for config, score in zip(configurations, scores)
                 if "Fasttext-wn-sw-300" in config]
tensor_scores = [score for config, score in zip(configurations, scores) if config == config == "MMR High Diversity Universal-Sentence-Encoder"]
lda_scores = [54.08073022312373, 53.172008113590266]

bar_positions = np.arange(len(gensim_scores) + len(tensor_scores) + len(lda_scores))
colors = {'Fasttext-wn-sw-300': 'deepskyblue', 'LDA': 'lightcoral', "Universal Sentence Encoder": "grey"}
bar_colors = [colors['Fasttext-wn-sw-300']] * len(gensim_scores) + [colors["Universal Sentence Encoder"]]  * len(tensor_scores) + [colors['LDA']] * len(lda_scores)

plt.figure(figsize=(8, 6))
bars = plt.bar(bar_positions, gensim_scores + tensor_scores + lda_scores, color=bar_colors)

# Customize the plot
plt.ylabel('Scores')
plt.title('Fasttext-wn-sw-300, Universal Sentence Encoder, and LDA Scores')
plt.xticks([])
plt.xlabel("Different Configurations of Fasttext-wn-sw-300, USE, and LDA")

labels = list(colors.keys())
handles = [plt.Rectangle((0,0), 1, 1, color=colors[label]) for label in labels]
plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

# Show the plot
plt.savefig("KeyBert/images/lda_vs_gensim.png")
