import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
latent_neurons = ["Normal", 2048, 1024, 512, 128]

# Data for Normal
normal_hr = [0.0463, 0.0463, 0.0463, 0.0463, 0.0463]  # HR metrics for Normal
normal_ndcg = [0.0304, 0.0304, 0.0304, 0.0304, 0.0304]  # NDCG metrics for Normal

# Data for k=32
k_32_hr = [
    [0.0463, 0.0441, 0.0457, 0.0460, 0.0460],  # HR@5
    [0.0712, 0.0670, 0.0689, 0.0693, 0.0700]   # HR@10
]
k_32_ndcg = [
    [0.0304, 0.0290, 0.0298, 0.0294, 0.0297],  # NDCG@5
    [0.0385, 0.0363, 0.0373, 0.0369, 0.0374]   # NDCG@10
]

# Data for k=16
k_16_hr = [
    [0.0463, 0.0454, 0.0452, 0.0447, 0.0440],  # HR@5
    [0.0712, 0.0690, 0.0670, 0.0672, 0.0654]   # HR@10
]
k_16_ndcg = [
    [0.0304, 0.0294, 0.0288, 0.0285, 0.0283],  # NDCG@5
    [0.0385, 0.0370, 0.0359, 0.0358, 0.0351]   # NDCG@10
]

# Plot settings
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
bar_width = 0.15
x = np.arange(len(latent_neurons))

# Add horizontal lines for Normal case
metrics_normal = {
    "HR@5": normal_hr[0],
    "HR@10": 0.0712,  # HR@10 normal value
    "NDCG@5": normal_ndcg[0],
    "NDCG@10": 0.0385  # NDCG@10 normal value
}

colors = {'HR@5': 'gray', 'HR@10': 'lightblue', 'NDCG@5': 'lightgreen', 'NDCG@10': 'pink'}
for ax in axes:
    for metric, val in metrics_normal.items():
        ax.axhline(y=val, label=f"{metric} (Normal)", linestyle="--", linewidth=1, alpha=0.7, color=colors[metric])

# Row 1: k=32
ax = axes[0]
ax.bar(x - 0.225, k_32_hr[0], bar_width, label="HR@5", color='blue')
ax.bar(x - 0.075, k_32_hr[1], bar_width, label="HR@10", color='orange')
ax.bar(x + 0.075, k_32_ndcg[0], bar_width, label="NDCG@5", color='green')
ax.bar(x + 0.225, k_32_ndcg[1], bar_width, label="NDCG@10", color='red')
ax.set_title("Performance Metrics for k=32")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(latent_neurons)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.grid(True, linestyle='--', linewidth=0.5)

# Row 2: k=16
ax = axes[1]
ax.bar(x - 0.225, k_16_hr[0], bar_width, label="HR@5", color='blue')
ax.bar(x - 0.075, k_16_hr[1], bar_width, label="HR@10", color='orange')
ax.bar(x + 0.075, k_16_ndcg[0], bar_width, label="NDCG@5", color='green')
ax.bar(x + 0.225, k_16_ndcg[1], bar_width, label="NDCG@10", color='red')
ax.set_title("Performance Metrics for k=16")
ax.set_xlabel("Latent Neurons")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(latent_neurons)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
