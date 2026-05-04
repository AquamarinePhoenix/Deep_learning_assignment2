import matplotlib.pyplot as plt

def plot_training_curves(loss_history, time_history):
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
    plt.show()