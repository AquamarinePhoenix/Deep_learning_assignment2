import matplotlib.pyplot as plt

def plot_loss_history(train_loss, val_loss=None, epoch_times=None, save_path=None):
    epochs = list(range(1, len(train_loss) + 1))

    if epoch_times is None:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, marker="o", label="Train Loss")

        if val_loss is not None:
            plt.plot(epochs, val_loss, marker="s", label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=100)
        plt.show()
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss", linewidth=2)
    if val_loss is not None:
        axes[0].plot(epochs, val_loss, marker="s", label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(epochs, epoch_times, color="steelblue", alpha=0.8, edgecolor="black")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Epoch Time")
    axes[1].grid(True, axis="y", alpha=0.3)

    for i, value in enumerate(epoch_times):
        axes[1].text(epochs[i], value, f"{value:.2f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.show()