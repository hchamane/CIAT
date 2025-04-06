import matplotlib.pyplot as plt
import numpy as np
import base64
import io

def render_bar_plot(x_labels, values, title: str, xlabel: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(x_labels))
    bars = ax.bar(x_pos, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(x_labels))))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def render_horizontal_bar_plot(y_labels, values, title: str, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(y_labels))
    bars = ax.barh(y_pos, values, color=plt.cm.plasma(np.linspace(0.2, 0.8, len(y_labels))))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.1f}%', xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points", ha='left', va='center')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')
