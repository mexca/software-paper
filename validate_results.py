"""Validate face and speaker identification as well as transcription accuracy.

Specifically:
- Calculate optimal mappings between speaker ids, face ids and annotation labels
- Calculate micro F1 score for face classification
- Calculate diarization error rate for speaker diarization
- Calculate word error rate for transcription

"""

import ast
import json
import logging
import os
from itertools import product
from typing import Dict, List, Tuple, Union

import jiwer
import jiwer.transforms as tr
import numpy as np
import pandas as pd
from constants import *
from helper_functions import calc_face_height, sub_labels
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy.optimize import linear_sum_assignment

logging.basicConfig(
    filename=os.path.join(PAPER_DIR, "validate_results.log"), level=logging.INFO
)


def prep_face_feat_ref_df(
    feat_df: pd.DataFrame, ref_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares feature and reference data frames."""
    # Get unique frame indices
    frame = np.unique(feat_df.frame)
    # Get frame timestamps
    time = frame * FRAME_RATE
    # Get detected faces for each timestamp
    feat_faces = [
        feat_df[feat_df.time == t].face_label.dropna().tolist() for t in time
    ]
    # Get detected face heights for each time stamp
    feat_face_heights = [
        feat_df[feat_df.time == t].face_box.dropna()
        # Face box must be converted from string to list
        .apply(lambda x: calc_face_height(ast.literal_eval(x))).tolist()
        for t in time
    ]
    # Get reference faces for each timestamp
    ref_faces = [
        ref_df[
            ref_df.start.le(t) & ref_df.end.ge(t + FRAME_RATE)
        ].face_name.tolist()
        for t in time
    ]
    # Create prepared data frames
    feat_df = pd.DataFrame(
        {
            "frame": frame,
            "time": time,
            "faces": feat_faces,
            "face_heights": feat_face_heights,
        }
    )
    ref_df = pd.DataFrame({"frame": frame, "time": time, "faces": ref_faces})
    return feat_df, ref_df


def get_optimal_face_mapping(
    x_list: Union[List, np.ndarray],
    y_list: Union[List, np.ndarray],
    x_labels: Union[List, np.ndarray],
    y_labels: Union[List, np.ndarray],
    x_cond: Union[List, np.ndarray],
) -> Dict:
    """Calculate the optimal mapping between detected and reference face labels."""
    # Init cost matrix
    cost_mat = np.zeros((len(x_labels), len(y_labels)))

    for x, y, c in zip(x_list, y_list, x_cond):
        # Get unique detected faces (some faces are duplicates for different speakers) larger than minimum height
        x = np.unique(np.array(x)[np.array(c) >= MIN_FACE_HEIGHT])

        # Create nested loop pairs
        matches = product(x, y)

        # Loop through pairs
        for match in matches:
            # If pair elements match increase cost matrix cell
            cost_mat[
                np.where(x_labels == match[0]), np.where(y_labels == match[1])
            ] += 1

    # Get mapping from cost matrix
    rows, cols = linear_sum_assignment(-cost_mat, maximize=False)

    mapping = {}

    # Assign labels to mapping
    for r, c in zip(rows, cols):
        mapping[str(int(r))] = y_labels[c]

    return mapping


def match_faces_to_ref(
    feat_df: pd.DataFrame, ref_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Match and substitute detected faces and reference faces in data frames and return mapping."""
    # Prepare data frames
    feat_df, ref_df = prep_face_feat_ref_df(feat_df, ref_df)
    # Calc optimal mapping
    mapping = get_optimal_face_mapping(
        feat_df.faces,
        ref_df.faces,
        np.arange(len(CANDIDATES)),
        np.array(CANDIDATES),
        feat_df.face_heights,
    )
    # Substitute face labels
    feat_df["faces"] = feat_df["faces"].apply(
        lambda x: [sub_labels(y, mapping) for y in x]
    )
    return feat_df, ref_df, mapping


def calc_face_metrics(
    feat_df: pd.DataFrame, ref_df: pd.DataFrame
) -> Tuple[int, int, int, int, int]:
    """Caculate validation metrics for detected and reference faces."""
    # Init metric counters
    h = 0
    m = 0
    c = 0
    f = 0
    r = 0

    for feat_row, ref_row in zip(feat_df.itertuples(), ref_df.itertuples()):
        # Sort and threshold detected faces
        feat_faces = list(
            set(
                sorted(
                    [
                        face
                        for i, face in enumerate(feat_row.faces.copy())
                        if feat_row.face_heights[i] > MIN_FACE_HEIGHT
                    ]
                )
            )
        )
        # Sort reference faces
        ref_faces = sorted(face.lower() for face in ref_row.faces)

        # Correct rejection
        if len(feat_faces) == 0 and len(ref_faces) == 0:
            r += 1
        else:
            for face in feat_faces[:]:
                # Hit
                if face in ref_faces:
                    h += 1
                    # Remove faces when hit
                    feat_faces.remove(face)
                    ref_faces.remove(face)
            # Update difference in detected and reference faces length
            len_diff = len(ref_faces) - len(feat_faces)

            # Miss if more reference than detected
            if len_diff > 0:
                m += len_diff
            # Confusion if equal reference to detected but not hit
            elif len_diff == 0:
                c += len(ref_faces)
            # False alarm if more detected than reference
            else:
                f += abs(len_diff)

    return h, m, c, f, r


def calc_face_metrics_batch(feat_dfs: List[pd.DataFrame]) -> Tuple[Dict, Dict]:
    """Calculate face detection validation metrics for all segments."""
    ref_filenames = sorted(
        [
            filename
            for filename in os.listdir(REF_DIR)
            if filename.endswith("video_annotation.csv")
        ]
    )

    ref_dfs = pd.Series(
        [
            pd.read_csv(os.path.join(REF_DIR, filename), index_col=0)
            for filename in ref_filenames
        ]
    )

    mappings = {}

    hit = []
    miss = []
    confusion = []
    false_alarm = []
    corr_reject = []

    for i, (feat_df, ref_df) in enumerate(zip(feat_dfs, ref_dfs)):
        feat_df, ref_df, mapping = match_faces_to_ref(feat_df, ref_df)
        mappings[i] = mapping

        metrics = calc_face_metrics(feat_df, ref_df)
        hit.append(metrics[0])
        miss.append(metrics[1])
        confusion.append(metrics[2])
        false_alarm.append(metrics[3])
        corr_reject.append(metrics[4])

    total = (
        np.array(hit)
        + np.array(corr_reject)
        + np.array(miss)
        + np.array(confusion)
    )

    metrics_batch = {
        "hit": (np.array(hit) / total).tolist(),
        "miss": (np.array(miss) / total).tolist(),
        "confusion": (np.array(confusion) / total).tolist(),
        "false_alarm": (np.array(false_alarm) / total).tolist(),
        "correct_reject": (np.array(corr_reject) / total).tolist(),
        "total": total.tolist(),
        "micro_precision": (
            np.array(hit) / (np.array(hit) + np.array(confusion))
        ).tolist(),
        "micro_recall": (
            np.array(hit) / (np.array(hit) + np.array(miss))
        ).tolist(),
    }
    metrics_batch["micro_f1"] = (
        2
        * (
            np.array(metrics_batch["micro_precision"])
            * np.array(metrics_batch["micro_recall"])
        )
        / (
            np.array(metrics_batch["micro_precision"])
            + np.array(metrics_batch["micro_recall"])
        )
    ).tolist()
    metrics_batch["micro_f1_avg"] = np.mean(metrics_batch["micro_f1"]).tolist()

    return metrics_batch, mappings


def rttm_to_annotation(filename: str) -> Annotation:
    """Convert a speech segment annotation in RTTM format into an pyannote.core.Annotation object."""
    annotation = Annotation()
    offset_start, _ = filename.split("_")[2:4]
    with open(filename, "r", encoding="utf-8") as file:
        for line in file.readlines():
            fields = line.split(" ")
            new_start = float(fields[3]) - float(offset_start)
            new_end = new_start + float(fields[4])
            new_seg = Segment(new_start, new_end)
            annotation[new_seg] = fields[7]

    return annotation


def feat_df_to_annotation(feat_df: pd.DataFrame) -> Annotation:
    """Convert a feature pandas.DataFrame into a pyannote.core.Annotation object."""
    start, end, spk = (0, 0, "")

    annotation = Annotation()

    for row in feat_df.itertuples():
        if (
            row.segment_start != start
            and row.segment_end != end
            and row.segment_speaker_label != spk
        ):
            if np.isnan(row.segment_speaker_label):
                spk = np.nan
            else:
                spk = str(int(row.segment_speaker_label))

            annotation[Segment(row.segment_start, row.segment_end)] = spk

        start, end, spk = (
            row.segment_start,
            row.segment_end,
            row.segment_speaker_label,
        )

    return annotation


def calc_speaker_metrics_batch(
    feat_dfs: List[pd.DataFrame],
) -> Tuple[Dict, Dict]:
    """Calculate validation metrics for speaker diarization."""
    ref_filenames = sorted(
        [
            filename
            for filename in os.listdir(REF_DIR)
            if filename.endswith("speaker_annotation.rttm")
        ]
    )
    ref_annotations = [
        rttm_to_annotation(os.path.join(REF_DIR, filename))
        for filename in ref_filenames
    ]

    feat_annotations = feat_dfs.apply(feat_df_to_annotation)

    der = DiarizationErrorRate()

    metrics_batch = {}
    mappings = {}

    for i, (ref, feat) in enumerate(zip(ref_annotations, feat_annotations)):
        metrics = der(ref, feat, detailed=True)
        metrics["detection error rate"] = (
            metrics["false alarm"] + metrics["missed detection"]
        ) / metrics["total"]
        mappings[i] = der.optimal_mapping(ref, feat)
        for key, val in metrics.items():
            if key not in metrics_batch:
                metrics_batch[key] = [val]
            else:
                metrics_batch[key].append(val)

        print(metrics)

    metrics_batch["der_avg"] = np.mean(
        metrics_batch["diarization error rate"]
    ).tolist()

    return metrics_batch, mappings


def txt_to_reference(filename: str) -> List[Tuple[float, float, str, str]]:
    """Convert a transcription text file into a text reference list."""
    ref = []
    with open(filename, "r") as file:
        for line in file.readlines():
            l_split = line.split("\t")
            if len(l_split) == 5:
                ref.append(
                    (
                        float(l_split[1]),
                        float(l_split[2]),
                        l_split[3],
                        l_split[4].strip(),
                    )
                )

    return ref


def txt_to_reference_batch(filenames: str) -> List[str]:
    """Convert all transcription text files into reference lists."""
    refs_conc = []

    for filename in filenames:
        refs_conc.append("\n".join([s[3] for s in txt_to_reference(filename)]))

    return refs_conc


def calc_text_metrics_batch(feat_dfs: List[pd.DataFrame]) -> List[float]:
    """Calculate transcription validation metrics for all segments."""
    ref_filenames = sorted(
        [
            os.path.join(REF_DIR, filename)
            for filename in os.listdir(REF_DIR)
            if filename.endswith("text_annotation.txt")
        ]
    )
    refs_conc = txt_to_reference_batch(ref_filenames)
    transcripts_conc = [
        "\n".join(
            [s.strip() for s in feat_df.span_text.unique() if not pd.isna(s)]
        )
        for feat_df in feat_dfs
    ]

    standardize = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.ExpandCommonEnglishContractions(),
            jiwer.RemovePunctuation(),
            tr.RemoveKaldiNonWords(),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToListOfListOfWords(),
        ]
    )

    processed_wer = []

    for ref, tra in zip(refs_conc, transcripts_conc):
        processed_wer.append(
            jiwer.process_words(
                ref,
                tra,
                reference_transform=standardize,
                hypothesis_transform=standardize,
            )
        )

    wer = [p.wer for p in processed_wer]

    processed_cer = []

    for ref, tra in zip(refs_conc, transcripts_conc):
        processed_cer.append(
            jiwer.process_characters(
                ref,
                tra,
                reference_transform=standardize,
                hypothesis_transform=standardize,
            )
        )

    cer = [p.cer for p in processed_cer]

    return {
        "wer": wer,
        "wer_avg": np.mean(wer),
        "cer": cer,
        "cer_avg": np.mean(cer),
    }


def main():
    feat_filenames = sorted(
        [
            filename
            for filename in os.listdir(RESULTS_DIR)
            if filename.endswith("features.csv")
        ]
    )

    feat_dfs = pd.Series(
        [
            pd.read_csv(os.path.join(RESULTS_DIR, filename), index_col=0)
            for filename in feat_filenames
        ]
    )

    face_metrics, face_mappings = calc_face_metrics_batch(feat_dfs)

    with open(
        os.path.join(RESULTS_DIR, "face_identification_results.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(face_metrics, file)

    with open(
        os.path.join(RESULTS_DIR, "face_identification_mappings.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(face_mappings, file)

    speaker_metrics, speaker_mappings = calc_speaker_metrics_batch(feat_dfs)

    with open(
        os.path.join(RESULTS_DIR, "speaker_identification_results.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(speaker_metrics, file)

    with open(
        os.path.join(RESULTS_DIR, "speaker_identification_mappings.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(speaker_mappings, file)

    text_metrics = calc_text_metrics_batch(feat_dfs)

    text_metrics["wer_adjusted"] = [
        w - d
        for w, d in zip(
            text_metrics["wer"], speaker_metrics["detection error rate"]
        )
    ]
    text_metrics["wer_adjusted_avg"] = np.mean(
        text_metrics["wer_adjusted"]
    ).tolist()

    with open(
        os.path.join(RESULTS_DIR, "audio_transcription_results.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(text_metrics, file)

    logging.info(
        r"""
\begin{table}[]
\begin{tabular}{lllllll}
                               \hline
                               & \multicolumn{5}{c}{Segment} &      \\
                               \cline{2-6}
                               & 1   & 2   & 3   & 4   & 5   & Avg. \\ \hline
Face identification (micro F1) & %s     & %s     \\
Speaker diarization (DER)      & %s     & %s     \\
Transription (WER)             & %s     & %s     \\
Transription (WER adjusted)    & %s     & %s     \\ \hline
\end{tabular}
\end{table}
""",
        " & ".join(str(round(s, 2)) for s in face_metrics["micro_f1"]),
        round(face_metrics["micro_f1_avg"], 2),
        " & ".join(
            str(round(s, 2)) for s in speaker_metrics["diarization error rate"]
        ),
        round(speaker_metrics["der_avg"], 2),
        " & ".join([str(round(s, 2)) for s in text_metrics["wer"]]),
        round(text_metrics["wer_avg"], 2),
        " & ".join([str(round(s, 2)) for s in text_metrics["wer_adjusted"]]),
        round(text_metrics["wer_adjusted_avg"], 2),
    )


if __name__ == "__main__":
    main()
