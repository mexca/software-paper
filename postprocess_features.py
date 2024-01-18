"""Post-process emotion expression features for further analysis.

Specifically:
- Unpack list columns
- Replace face and speaker ids by mapped labels
- Filter faces that are too small or duplicate

"""

import ast
import json
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
from constants import *
from helper_functions import calc_face_height, sub_labels

logging.basicConfig(
    filename=os.path.join(PAPER_DIR, "postprocess.log"), level=logging.INFO
)


AU_REF = [
    1,
    2,
    4,
    5,
    6,
    7,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    32,
    38,
    39,
    "L1",
    "R1",
    "L2",
    "R2",
    "L4",
    "R4",
    "L6",
    "R6",
    "L10",
    "R10",
    "L12",
    "R12",
    "L14",
    "R14",
]


def convert_strings_to_list(df: pd.DataFrame) -> pd.DataFrame:
    """Converts columns that have strings containing a list to a normal list.

    Necessary because pandas encodes lists as strings.

    """
    df.loc[df.face_box.notna(), "face_box"] = df.face_box.dropna().apply(
        ast.literal_eval
    )
    df.loc[
        df.face_landmarks.notna(), "face_landmarks"
    ] = df.face_landmarks.dropna().apply(ast.literal_eval)
    df.loc[df.face_aus.notna(), "face_aus"] = df.face_aus.dropna().apply(
        ast.literal_eval
    )

    return df


def split_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Split a list column into columns for each list element."""
    df[
        ["face_box_x1", "face_box_y1", "face_box_x2", "face_box_y2"]
    ] = df.face_box.apply(
        lambda x: x if isinstance(x, list) else [pd.NA, pd.NA, pd.NA, pd.NA]
    ).tolist()
    df[[f"face_au_{au}" for au in AU_REF]] = df.face_aus.apply(
        lambda x: x if isinstance(x, list) else [pd.NA for au in AU_REF]
    ).tolist()
    df[[f"face_landmarks_x{i}" for i in range(1, 6)]] = df.face_landmarks.apply(
        lambda x: [e[0] for e in x]
        if isinstance(x, list)
        else [pd.NA for i in range(5)]
    ).tolist()
    df[[f"face_landmarks_y{i}" for i in range(1, 6)]] = df.face_landmarks.apply(
        lambda x: [e[1] for e in x]
        if isinstance(x, list)
        else [pd.NA for i in range(5)]
    ).tolist()

    return df.drop(columns=["face_box", "face_aus", "face_landmarks"])


def sub_face_labels(df: pd.DataFrame, mapping: Dict) -> pd.DataFrame:
    """Replace face labels by mapped labels."""
    df["face_label"] = df["face_label"].apply(lambda x: sub_labels(x, mapping))
    return df


def sub_speaker_labels(df: pd.DataFrame, mapping: Dict) -> pd.DataFrame:
    """Replace speaker labels by mapped speaker labels."""
    df["segment_speaker_label"] = df["segment_speaker_label"].apply(
        lambda x: sub_labels(x, mapping)
    )
    return df


def filter_faces(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude faces below a minimum height and drop duplicate face entries."""
    face_height = df.face_box.apply(calc_face_height)
    is_valid = face_height >= MIN_FACE_HEIGHT

    logging.info(
        "Excluded %s out of %s faces (%s) because their height was below %s",
        sum(~is_valid),
        len(is_valid),
        sum(~is_valid) / len(is_valid),
        MIN_FACE_HEIGHT,
    )

    face_col_names = [
        "face_box",
        "face_aus",
        "face_prob",
        "face_landmarks",
        "face_confidence",
        "face_label",
    ]

    df.loc[~is_valid, face_col_names] = np.nan

    return df.drop_duplicates(
        subset=[col for col in df.columns if col not in face_col_names]
    )


def set_empty_strings_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["span_text"] == "", "span_text"] = np.nan
    return df


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

    with open(
        os.path.join(RESULTS_DIR, "face_identification_mappings.json"),
        "r",
        encoding="utf-8",
    ) as file:
        face_mappings = json.load(file)

    with open(
        os.path.join(RESULTS_DIR, "speaker_identification_mappings.json"),
        "r",
        encoding="utf-8",
    ) as file:
        speaker_mappings = json.load(file)

    feat_dfs = pd.Series(
        [
            sub_face_labels(df, mapping)
            for df, mapping in zip(feat_dfs, face_mappings.values())
        ]
    )
    feat_dfs = pd.Series(
        [
            sub_speaker_labels(df, mapping)
            for df, mapping in zip(feat_dfs, speaker_mappings.values())
        ]
    )
    feat_dfs = (
        feat_dfs.apply(convert_strings_to_list)
        .apply(filter_faces)
        .apply(split_list_columns)
        .apply(set_empty_strings_to_nan)
    )

    for i, filename in enumerate(feat_filenames):
        feat_dfs[i].to_csv(
            os.path.join(RESULTS_DIR, filename.split(".")[0] + "_post.csv")
        )


if __name__ == "__main__":
    main()
