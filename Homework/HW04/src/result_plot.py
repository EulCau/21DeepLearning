import os
import matplotlib.pyplot as plt
from collections import defaultdict


def load_metrics(file_path):
    """
    加载实验数据文件，按 tag 和 phase 分类
    返回结构: metrics[tag][phase] = [epoch1, epoch2, ...]
    """
    metrics = defaultdict(lambda: defaultdict(list))
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            tag, phase, epoch, value = int(parts[0]), parts[1], int(parts[2]), float(parts[3])
            metrics[tag][phase].append(value)
    return metrics


def plot_metric(metrics, phase, y_label, title, filename, label_map):
    """
    根据 phase 画图, 如 pretrain 或 eval
    """
    plt.figure(figsize=(10, 6))
    for tag, phases in metrics.items():
        if phase in phases:
            values = phases[phase]
            plt.plot(range(1, len(values) + 1), values, label=label_map.get(tag, f"Tag {tag}"), linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 路径配置
    metric_file = "../result/output.txt"
    output_dir = "../result"
    os.makedirs(output_dir, exist_ok=True)

    # tag -> name 映射
    label_map = {
        1: 'basic',
        2: 'color',
        3: 'gray',
        4: 'strong'
    }

    # 加载并绘图
    metrics = load_metrics(metric_file)

    plot_metric(metrics, phase="pretrain", y_label="Loss",
				title="SimCLR Pretraining Loss",
				filename=os.path.join(output_dir, "pretrain_loss_all.png"),
				label_map=label_map)

    plot_metric(metrics, phase="eval", y_label="Accuracy (%)",
				title="Linear Evaluation Accuracy",
				filename=os.path.join(output_dir, "eval_accuracy_all.png"),
				label_map=label_map)


if __name__ == "__main__":
    main()
