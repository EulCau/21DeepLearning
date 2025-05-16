import os
import matplotlib.pyplot as plt
from collections import defaultdict

# 标签映射
augment_label_map = {
    1: 'basic',
    2: 'color',
    3: 'gray',
    4: 'strong'
}

head_label_map = {
    5: 'mlp_bn',
    6: 'mlp_no_bn',
    7: 'linear',
    8: 'none'
}

def load_metrics(file_path):
    metrics = defaultdict(lambda: defaultdict(list))  # tag -> phase -> [values]
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            tag, phase, epoch, value = int(parts[0]), parts[1], int(parts[2]), float(parts[3])
            metrics[tag][phase].append(value)
    return metrics

def plot_group(metrics, tags, label_map, phase, y_label, title, filename):
    plt.figure(figsize=(10, 6))
    for tag in tags:
        if tag in metrics and phase in metrics[tag]:
            values = metrics[tag][phase]
            plt.plot(range(1, len(values)+1), values, label=label_map[tag], linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    file_path = "../result/output.txt"
    output_dir = "../result"
    os.makedirs(output_dir, exist_ok=True)

    metrics = load_metrics(file_path)

    # 绘图：增强策略
    plot_group(metrics, tags=[1, 2, 3, 4],
               label_map=augment_label_map,
               phase="pretrain",
               y_label="Loss",
               title="Pretraining Loss (Augmentation)",
               filename=os.path.join(output_dir, "pretrain_loss_augment.png"))

    plot_group(metrics, tags=[1, 2, 3, 4],
               label_map=augment_label_map,
               phase="eval",
               y_label="Accuracy (%)",
               title="Evaluation Accuracy (Augmentation)",
               filename=os.path.join(output_dir, "eval_acc_augment.png"))

    # 绘图：projection head
    plot_group(metrics, tags=[5, 6, 7, 8],
               label_map=head_label_map,
               phase="pretrain",
               y_label="Loss",
               title="Pretraining Loss (Projection Head)",
               filename=os.path.join(output_dir, "pretrain_loss_head.png"))

    plot_group(metrics, tags=[5, 6, 7, 8],
               label_map=head_label_map,
               phase="eval",
               y_label="Accuracy (%)",
               title="Evaluation Accuracy (Projection Head)",
               filename=os.path.join(output_dir, "eval_acc_head.png"))

if __name__ == "__main__":
    main()
