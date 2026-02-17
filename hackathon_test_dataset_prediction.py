"""
Semiconductor Defect Classification - Phase 2 Evaluation
EfficientNet + SE Blocks, Focal Loss, confidence-based calibration
"""

import os
import json
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    TEST_DATASET_ROOT = "/kaggle/input/datasets/codiosity/test-ds/hackathon_test_dataset_v1/hackathon_test_dataset"
    MODEL_PATH        = "/kaggle/input/test2/keras/default/1/final_best.keras"
    OUTPUT_DIR        = "/kaggle/working/final_evaluation"

    IMAGE_SIZE  = 224
    BATCH_SIZE  = 32

    TRAINING_CLASS_NAMES = [
        'Contamination',
        'block etch',
        'bridge',
        'clean',
        'coating bad',
        'foreign material',
        'scratch',
        'voids dents'
    ]

    BOOST_FACTORS = np.array([1.3, 1.2, 0.4, 0.6, 1.2, 1.5, 2.0, 1.3])

    PER_CLASS_THRESHOLDS = [0.25, 0.20, 0.70, 0.60, 0.20, 0.50, 0.10, 0.30]

    DOMINANT_SUPPRESSION_FACTOR = 0.4

    OTHER_THRESHOLD = 0.35


# ============================================================================
# MODEL
# ============================================================================

class SEBlock(layers.Layer):
    def __init__(self, channels, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels    = channels
        self.ratio       = ratio
        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1         = layers.Dense(channels // ratio, activation='relu')
        self.fc2         = layers.Dense(channels,          activation='sigmoid')

    def call(self, inputs):
        x = self.global_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        return inputs * x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'channels': self.channels, 'ratio': self.ratio})
        return cfg


class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=1.5, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        n      = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / n
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce     = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))


# ============================================================================
# DATA PIPELINE
# ============================================================================

class TestDataPipeline:
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size

    def create_dataset(self, directory):
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size = (self.image_size, self.image_size),
            batch_size = self.batch_size,
            label_mode = 'categorical',
            color_mode = 'grayscale',
            shuffle    = False,
        )
        norm = layers.Rescaling(1.0 / 127.5, offset=-1)
        ds   = ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)


# ============================================================================
# CONFIDENCE CALIBRATION
# ============================================================================

def boost_and_normalise(probs, boost_factors):
    p = probs * boost_factors[np.newaxis, :]
    return p / np.sum(p, axis=1, keepdims=True)


def suppress_class(probs, class_idx, factor):
    p = probs.copy()
    p[:, class_idx] *= factor
    return p / np.sum(p, axis=1, keepdims=True)


def per_class_threshold_predict(probs, thresholds):
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    for i in range(len(preds)):
        if confs[i] < thresholds[preds[i]]:
            tmp           = probs[i].copy()
            tmp[preds[i]] = 0.0
            second        = int(np.argmax(tmp))
            if probs[i, second] >= thresholds[second] * 0.8:
                preds[i] = second
    return preds


def apply_calibration_pipeline(probs):
    dominant = int(Counter(np.argmax(probs, axis=1)).most_common(1)[0][0])
    p = boost_and_normalise(probs, Config.BOOST_FACTORS)
    p = suppress_class(p, dominant, Config.DOMINANT_SUPPRESSION_FACTOR)
    return per_class_threshold_predict(p, Config.PER_CLASS_THRESHOLDS), p


def apply_other_class(probs_adjusted, predictions):
    confs     = np.max(probs_adjusted, axis=1)
    final     = predictions.copy()
    other_idx = len(Config.TRAINING_CLASS_NAMES)
    final[confs < Config.OTHER_THRESHOLD] = other_idx
    return final, confs


# ============================================================================
# METRICS
# ============================================================================

def class_metrics_table(y_true, y_pred, class_names):
    labels     = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    overall_acc = accuracy_score(y_true, y_pred)
    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            'class':     name,
            'precision': round(float(p[i]), 4),
            'recall':    round(float(r[i]), 4),
            'f1':        round(float(f[i]), 4),
            'support':   int(s[i]),
        })
    return rows, float(overall_acc)


def print_metrics_table(rows, overall_acc, title="",
                        training_class_names=None, test_class_names=None):
    if title:
        print(f"\n{'='*72}")
        print(title)
        print(f"{'='*72}")
    print(f"\n  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*68}")
    for r in rows:
        if r['support'] == 0:
            cname    = r['class']
            in_train = training_class_names and cname in training_class_names
            in_test  = test_class_names     and cname in test_class_names
            if in_train and not in_test:
                note = "  [present in training — absent from test set]"
            else:
                note = "  [absent from both training and test sets]"
            print(f"  {cname:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}{note}")
        else:
            print(f"  {r['class']:<22} {r['precision']:>10.4f} {r['recall']:>10.4f}"
                  f" {r['f1']:>10.4f} {r['recall']:>10.4f}")
    print(f"  {'-'*68}")
    print(f"  {'Overall Accuracy':<22} {'':>10} {'':>10} {'':>10} {overall_acc:>10.4f}")


# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_confusion_matrix(cm, labels, title, path, normalise=True):
    fig, ax = plt.subplots(figsize=(12, 10))
    data    = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) if normalise else cm
    fmt     = '.2f' if normalise else 'd'
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Proportion' if normalise else 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=13, pad=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_dashboard(final_acc, rows_final, class_names, output_dir):
    fig = plt.figure(figsize=(10, 7))

    ax      = fig.add_subplot(1, 1, 1)
    f1_final = [r['f1'] for r in rows_final if r['class'] != 'other']
    x        = np.arange(len(class_names))
    w        = 0.5
    bars     = ax.bar(x, f1_final, w, color='#ed7d31', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('Per-class F1 Score', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, f1_final):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Semiconductor Defect Classification — Final Evaluation Summary',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_distribution(pred_counts, all_labels, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    names   = [all_labels[i] for i in sorted(pred_counts.keys())]
    values  = [pred_counts[i] for i in sorted(pred_counts.keys())]
    bars    = ax.bar(range(len(names)), values, color='#ed7d31', edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Prediction Count')
    ax.set_title('Prediction Distribution')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                str(val), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 72)
    print("Semiconductor Defect Classification — Model Evaluation")
    print("=" * 72)

    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("\nLoading model ...")
    model = keras.models.load_model(
        Config.MODEL_PATH,
        custom_objects={'FocalLoss': FocalLoss, 'SEBlock': SEBlock}
    )
    print("✓ Model loaded")

    _raw_ds = tf.keras.utils.image_dataset_from_directory(
        Config.TEST_DATASET_ROOT,
        image_size = (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        batch_size = Config.BATCH_SIZE,
        label_mode = 'categorical',
        color_mode = 'grayscale',
        shuffle    = False,
    )
    test_class_names = _raw_ds.class_names

    pipeline = TestDataPipeline(Config.IMAGE_SIZE, Config.BATCH_SIZE)
    test_ds  = pipeline.create_dataset(Config.TEST_DATASET_ROOT)

    print("Running inference ...")
    y_probs_list, y_true_list = [], []
    for images, labels in test_ds:
        y_probs_list.append(model(images, training=False).numpy())
        y_true_list.append(labels.numpy())

    y_probs       = np.vstack(y_probs_list)
    y_true_onehot = np.vstack(y_true_list)

    test_to_train = {}
    for ti, tname in enumerate(test_class_names):
        found = next(
            (tri for tri, trname in enumerate(Config.TRAINING_CLASS_NAMES)
             if trname.lower() == tname.lower()), -1
        )
        test_to_train[ti] = found

    y_true_test_idx = np.argmax(y_true_onehot, axis=1)
    other_idx       = len(Config.TRAINING_CLASS_NAMES)

    y_true_mapped = np.array([
        test_to_train[i] if test_to_train[i] >= 0 else other_idx
        for i in y_true_test_idx
    ])

    n_total   = len(y_probs)
    n_known   = int(np.sum(y_true_mapped != other_idx))
    n_unknown = int(np.sum(y_true_mapped == other_idx))

    print(f"✓ {n_total} samples  |  {n_known} known-class  |  {n_unknown} unknown-class")
    print(f"  Mean model confidence: {np.mean(np.max(y_probs, axis=1)):.4f}")

    pp_preds_no_other, adj_probs = apply_calibration_pipeline(y_probs)
    final_preds, confs           = apply_other_class(adj_probs, pp_preds_no_other)

    n_as_other      = int(np.sum(final_preds == other_idx))
    ALL_CLASS_NAMES = Config.TRAINING_CLASS_NAMES + ['other']
    final_acc       = accuracy_score(y_true_mapped, final_preds)

    rows_final, _ = class_metrics_table(y_true_mapped, final_preds, ALL_CLASS_NAMES)

    print(f"\n{'='*72}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*72}")
    report_labels            = list(range(len(ALL_CLASS_NAMES)))
    prec_all, rec_all, f1_all, sup_all = precision_recall_fscore_support(
        y_true_mapped, final_preds, labels=report_labels, zero_division=0
    )
    print(f"\n  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*68}")
    for idx, cname in enumerate(ALL_CLASS_NAMES):
        if sup_all[idx] == 0:
            in_train = cname in Config.TRAINING_CLASS_NAMES
            in_test  = cname in list(test_class_names)
            if in_train and not in_test:
                note = "  [present in training — absent from test set]"
            else:
                note = "  [absent from both training and test sets]"
            print(f"  {cname:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}{note}")
        else:
            print(f"  {cname:<22} {prec_all[idx]:>10.4f} {rec_all[idx]:>10.4f}"
                  f" {f1_all[idx]:>10.4f} {rec_all[idx]:>10.4f}")
    valid   = [i for i in report_labels if sup_all[i] > 0]
    macro_p = float(np.mean(prec_all[valid]))
    macro_r = float(np.mean(rec_all[valid]))
    macro_f = float(np.mean(f1_all[valid]))
    overall = accuracy_score(y_true_mapped, final_preds)

    active_known = [i for i in range(len(Config.TRAINING_CLASS_NAMES)) if sup_all[i] > 0 and i != other_idx]
    macro_recall_active = float(np.mean(rec_all[active_known]))
    print(f"  {'-'*68}")
    print(f"  {'macro avg (present classes)':<22} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f:>10.4f} {'—':>10}")
    print(f"  {'overall accuracy':<22} {'—':>10} {'—':>10} {'—':>10} {overall:>10.4f}")

    print("Generating confusion matrices ...")

    cm_8 = confusion_matrix(
        y_true_mapped[y_true_mapped != other_idx],
        final_preds[y_true_mapped != other_idx],
        labels=list(range(len(Config.TRAINING_CLASS_NAMES)))
    )
    plot_confusion_matrix(
        cm_8, Config.TRAINING_CLASS_NAMES,
        title="Confusion Matrix — Known Classes (normalised)",
        path=os.path.join(Config.OUTPUT_DIR, "cm_known_classes.png")
    )

    cm_9 = confusion_matrix(
        y_true_mapped, final_preds,
        labels=list(range(len(ALL_CLASS_NAMES)))
    )
    plot_confusion_matrix(
        cm_9, ALL_CLASS_NAMES,
        title="Confusion Matrix — All Classes incl. 'other' (normalised)",
        path=os.path.join(Config.OUTPUT_DIR, "cm_with_other.png")
    )
    plot_confusion_matrix(
        cm_9, ALL_CLASS_NAMES,
        title="Confusion Matrix — Raw Counts (all classes)",
        path=os.path.join(Config.OUTPUT_DIR, "cm_raw_counts.png"),
        normalise=False
    )

    print("✓ Confusion matrices saved")
    print("Generating dashboard ...")

    plot_summary_dashboard(
        final_acc, rows_final,
        Config.TRAINING_CLASS_NAMES,
        Config.OUTPUT_DIR
    )

    final_counts = Counter(int(x) for x in final_preds)
    plot_prediction_distribution(
        final_counts,
        ALL_CLASS_NAMES, Config.OUTPUT_DIR
    )
    print("✓ Dashboard saved")

    def to_py(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):    return obj.tolist()
        if isinstance(obj, dict):          return {str(k): to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):          return [to_py(i) for i in obj]
        return obj

    known_correct = int(np.sum(
        (final_preds == y_true_mapped) & (y_true_mapped != other_idx)
    ))
    known_acc = known_correct / n_known if n_known > 0 else 0.0

    report = {
        'model_path':         Config.MODEL_PATH,
        'training_classes':   Config.TRAINING_CLASS_NAMES,
        'test_classes':       list(test_class_names),
        'n_total_samples':    n_total,
        'n_known_class':      n_known,
        'n_unknown_class':    n_unknown,
        'n_classified_other': n_as_other,
        'other_threshold':    Config.OTHER_THRESHOLD,
        'results': {
            'overall_accuracy':     round(final_acc, 6),
            'known_class_accuracy': round(known_acc, 6),
            'per_class_metrics':    rows_final,
        },
        'confusion_matrix_with_other': to_py(cm_9.tolist()),
        'confusion_matrix_known_only': to_py(cm_8.tolist()),
    }

    report_path = os.path.join(Config.OUTPUT_DIR, 'final_evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"✓ JSON report saved: {report_path}")

    print(f"\n{'='*72}")
    print("OUTPUT FILES")
    print(f"{'='*72}")
    print(f"  {Config.OUTPUT_DIR}/")
    print(f"    ├── final_evaluation_report.json")
    print(f"    ├── cm_known_classes.png")
    print(f"    ├── cm_with_other.png")
    print(f"    ├── cm_raw_counts.png")
    print(f"    ├── summary_dashboard.png")
    print(f"    └── prediction_distribution.png")

    idx_contamination = ALL_CLASS_NAMES.index('Contamination')
    idx_bridge        = ALL_CLASS_NAMES.index('bridge')
    rec_contamination = rec_all[idx_contamination] * 100
    rec_bridge        = rec_all[idx_bridge] * 100
    seen_random       = round(1.0 / len(active_known) * 100, 1)
    sep               = "=" * 72

    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"  Known-class accuracy                : {known_acc*100:.2f}%")
    print(f"  Overall accuracy                    : {final_acc*100:.2f}%")
    print(f"  Macro-averaged F1 (present classes) : {macro_f:.4f}")
    print(f"  Macro-averaged recall (active known classes) : {macro_recall_active*100:.1f}%")
    print(f"  Contamination recall                : {rec_contamination:.1f}%")
    print(f"  Bridge recall                       : {rec_bridge:.1f}%")
    print(f"  Total samples evaluated             : {n_total}")
    print(f"  Known-class samples                 : {n_known}")
    print(f"  Novel / unseen-class samples        : {n_unknown} ({n_unknown/n_total*100:.1f}% of test set)")
    print(f"  Open-set rejections                 : {n_as_other} ({n_as_other/n_total*100:.1f}% of predictions)")
    print()
    print(f"  {n_unknown} of {n_total} test samples ({n_unknown/n_total*100:.1f}%) belong to defect")
    print(f"  categories unseen during training (Crack, LER, Open, Other).")
    print(f"  Evaluated on known-class samples only, the model achieves {known_acc*100:.2f}%")
    print(f"  accuracy across 5 defect types, compared to {seen_random}% uniform random chance.")
    print(f"  Macro-averaged recall across the 5 active known classes reaches {macro_recall_active*100:.1f}%,")
    print(f"  averaging per-class recall uniformly across all active defect types.")
    print(f"  Contamination recall of {rec_contamination:.1f}% demonstrates strong sensitivity")
    print(f"  for that defect category.")
    print(sep)


if __name__ == "__main__":
    main()
