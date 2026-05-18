import os

import matplotlib.pyplot as plt

import _modules.config as cfg

def plot_training_curves(loss_history, time_history, val_metrics=None, metric_n=4):
    epochs = list(range(1, len(loss_history) + 1))
    fig, ax1 = plt.subplots()
    
    ax1.bar(
        epochs,
        time_history,
        alpha=0.6,
        label="Time per Epoch (s)",
        color = "#62a5ac"
    )
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Time (seconds)")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(
        epochs,
        loss_history,
        marker="o",
        linestyle="-",
        color="#ba6d64",
        label="Loss",
        markersize=6,
        linewidth=2
    )
    ax2.set_ylabel("Loss")
    ax2.tick_params(axis='y')

    plt.title("Training Loss + Time per Epoch")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, "training_loss.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # If validation metrics were provided, plot them too.
    if val_metrics:
        val_bleu4, val_cider4 = val_metrics
        fig2, ax = plt.subplots()
        ax.plot(epochs, val_bleu4, marker='o', linestyle='-', color='#2ca02c', label=f'Val BLEU-{metric_n}')
        ax.plot(epochs, val_cider4, marker='o', linestyle='-', color='#1f77b4', label=f'Val CIDEr-{metric_n}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')
        plt.title(f'Validation BLEU-{metric_n} / CIDEr-{metric_n} per Epoch')
        fig2.savefig(os.path.join(cfg.OUTPUT_DIR, 'training_metrics.png'), dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig2)