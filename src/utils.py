import json
import numpy as np
import pandas as pd
from typing import List, Optional

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, \
    accuracy_score


def write_classification_report(all_labels, all_predictions, classification_report_path):
    cr = classification_report(all_labels, all_predictions, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_excel(classification_report_path)
    # cr_df.to_csv(classification_report_path)


def write_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        file_path: str,
        labels: Optional[List] = None,
        hide_zeroes: bool = False,
        hide_diagonal: bool = False,
        hide_threshold: Optional[float] = None,
):
    """Print a nicely formatted confusion matrix with labelled rows and columns.

    Predicted labels are in the top horizontal header, true labels on the vertical header.

    Args:
        y_true (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted labels
        labels (Optional[List], optional): list of all labels. If None, then all labels present in the data are
            displayed. Defaults to None.
        hide_zeroes (bool, optional): replace zero-values with an empty cell. Defaults to False.
        hide_diagonal (bool, optional): replace true positives (diagonal) with empty cells. Defaults to False.
        hide_threshold (Optional[float], optional): replace values below this threshold with empty cells. Set to None
            to display all values. Defaults to None.
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # find which fixed column width will be used for the matrix
    columnwidth = max(
        [len(str(x)) for x in labels] + [5]
    )  # 5 is the minimum column width, otherwise the longest class name
    empty_cell = ' ' * columnwidth

    # top-left cell of the table that indicates that top headers are predicted classes, left headers are true classes
    padding_fst_cell = (columnwidth - 3) // 2  # double-slash is int division
    fst_empty_cell = padding_fst_cell * ' ' + 't/p' + ' ' * (columnwidth - padding_fst_cell - 3)

    with open(file_path, "w", encoding="utf-8") as fw:
        # Print header
        fw.write('    ' + fst_empty_cell + ' ')
        for label in labels:
            fw.write(f'{label:{columnwidth}} ')
        fw.write("\n")

        # Print rows
        for i, label in enumerate(labels):
            fw.write(f'    {label:{columnwidth}} ')
            for j in range(len(labels)):
                # cell value padded to columnwidth with spaces and displayed with 1 decimal
                cell = f'{cm[i, j]:{columnwidth}.1f}'
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                fw.write(cell + " ")
            fw.write("\n")

    print(f"Confusion matrix saved at: {file_path}")


def save_dict_as_csv_pandas(data_dict, target_file_path, transpose=False, col_names=None):
    # d = {"key1": [1, 2, 3], "key2": [5], "key3": [6, 9]}
    """
        Will print the dict keys line by line. Ex:
            key1    1   2   3
            key2    5
            key3    6   9
            ...
    """
    if transpose:
        orient = "columns"
    else:
        orient = "index"
    df = pd.DataFrame.from_dict(data_dict, orient=orient)
    if transpose:
        df = df.transpose()

    if col_names:
        df.columns = col_names

    df.to_excel(target_file_path, index=True)
    print(f"results have been saved successfully at: {target_file_path}")


def save_any_dict_into_json(dict_to_be_saved, target_json_file_path, do_indent=True):
    with open(target_json_file_path, "w", encoding="utf-8") as file_obj:
        if do_indent:
            json.dump(dict_to_be_saved, file_obj, indent=4, ensure_ascii=False)
        else:
            json.dump(dict_to_be_saved, file_obj, ensure_ascii=False)

    print(f"file has been successfully saved at: {target_json_file_path}")


def calculate_metrics(labels, predictions):
    # metrics:
    sk_acc = accuracy_score(labels, predictions)

    micro_precision = precision_score(labels, predictions, average="micro")
    macro_precision = precision_score(labels, predictions, average="macro")
    weighted_precision = precision_score(labels, predictions, average="weighted")

    micro_recall = recall_score(labels, predictions, average="micro")
    macro_recall = recall_score(labels, predictions, average="macro")
    weighted_recall = recall_score(labels, predictions, average="weighted")

    micro_f1 = f1_score(labels, predictions, average="micro")
    macro_f1 = f1_score(labels, predictions, average="macro")
    weighted_f1 = f1_score(labels, predictions, average="weighted")

    metrics_dict = {
        "accuracy": round(sk_acc, 4) * 100,
        "micro_precision": round(micro_precision, 4) * 100,
        "micro_recall": round(micro_recall, 4) * 100,
        "macro_recall": round(macro_recall, 4) * 100,
        "weighted_recall": round(weighted_recall, 4) * 100,
        "micro_f1": round(micro_f1, 4) * 100,
        "macro_precision": round(macro_precision, 4) * 100,
        "weighted_precision": round(weighted_precision, 4) * 100,
        "macro_f1": round(macro_f1, 4) * 100,
        "weighted_f1": round(weighted_f1, 4) * 100,
    }

    return metrics_dict


def print_metrics(metrics_dict):
    padding = 25
    print()

    print("scikit-acc:".ljust(padding) + str(metrics_dict["accuracy"]))
    print("Precision-micro:".ljust(padding) + str(metrics_dict["micro_precision"]))
    print("Recall-micro:".ljust(padding) + str(metrics_dict["micro_recall"]))
    print("Recall-macro:".ljust(padding) + str(metrics_dict["macro_recall"]))
    print("Recall-weighted:".ljust(padding) + str(metrics_dict["weighted_recall"]))
    print("F1-micro:".ljust(padding) + str(metrics_dict["micro_f1"]))
    print()

    print("Precision-macro:".ljust(padding) + str(metrics_dict["macro_precision"]))
    print("Precision-weighted:".ljust(padding) + str(metrics_dict["weighted_precision"]))
    print()

    print("F1-macro:".ljust(padding) + str(metrics_dict["macro_f1"]))
    print("F1-weighted:".ljust(padding) + str(metrics_dict["weighted_f1"]))

    print()
