"""Annotate a debate segment with output features.

Specifically:
- Annotates face bounding boxes and labels
- Annotates current speaker labels and speech transcriptions
- Adds three plots with emotion expression features over time:
    - AU12 activation
    - Voice pitch
    - Positive speech sentiment

"""

import os

import cv2
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from constants import *

# Set plot theme
sns.set_theme()
sns.set_style("white")
sns.set_style("ticks")

# Map candidate labels to colors
candidate_colors = {
    "marijnissen": "tomato",
    "kaag": "darkred",
    "rutte": "lightblue",
    "mod_m": "lightgreen",
    "klaver": "gold",
    "hoekstra": "olive",
    "wilders": "darkblue",
}

# Transform colors from RGB to BGR (for openCV)
candidate_colors_bgr = {
    key: [c * 255 for c in reversed(colors.to_rgb(candidate_colors[key]))]
    for key in candidate_colors
}


def draw_face_boxes(frame, features):
    """Draw face boxes and labels on a video frame."""
    for j, row in enumerate(features.iter_rows(named=True)):
        if not np.isnan(row["face_prob"]):
            x1 = int(row["face_box_x1"])
            x2 = int(row["face_box_x2"])
            y1 = int(row["face_box_y1"])
            y2 = int(row["face_box_y2"])

            if y2 - y1 > 70:
                lbl = str(row["face_label"])
                # Draw face box rectangle
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), candidate_colors_bgr[lbl], 2
                )

                x3 = int(x1 + (x2 - x1) / 2)
                y3 = y1 - 5

                face_label = lbl.capitalize()

                # Add face labels
                face_label_size = cv2.getTextSize(
                    face_label, font, font_scale, 1
                )[0]
                cv2.putText(
                    frame,
                    face_label,
                    (x3 - int(face_label_size[0] / 2), y3),
                    font,
                    font_scale,
                    candidate_colors_bgr[lbl],
                    1,
                    lineType=font_line_type,
                )


def draw_speaker_labels(frame, features):
    """Draw speaker labels on a video frame."""
    span_texts = features.select(pl.col("span_text")).to_series()

    txt_pos = (50, 50)

    for j, spk in enumerate(
        features.select(pl.col("segment_speaker_label")).to_series()
    ):
        if spk:
            speaker_label = str(spk).capitalize()
            speaker_label_size = cv2.getTextSize(
                speaker_label, font, font_scale, 1
            )[0]
            speaker_label_pos = (
                50,
                txt_pos[1] + j * speaker_label_size[1] + speaker_label_size[1],
            )
            cv2.putText(
                frame,
                speaker_label,
                speaker_label_pos,
                font,
                font_scale,
                candidate_colors_bgr[spk],
                1,
                lineType=font_line_type,
            )

            span_text = str(span_texts[j])
            span_text_split = span_text.split()
            txts = []
            while len(span_text_split) > 0:
                line = ""
                while len(line) < 30 and len(span_text_split) > 0:
                    line += span_text_split.pop(0) + " "
                txts.append(line[:-1])

            for k, txt in enumerate(txts):
                if txt:
                    txt_size = cv2.getTextSize(txt, font, font_scale, 1)[0]
                    txt_pos = (50, speaker_label_pos[1] + (k + 1) * txt_size[1])
                    cv2.putText(
                        frame,
                        txt,
                        txt_pos,
                        font,
                        font_scale,
                        candidate_colors_bgr[spk],
                        1,
                        lineType=font_line_type,
                    )


def create_au_plot(features, width, height):
    """Create plot for facial action unit activation over time."""
    time = features.select(pl.col("time")).to_series()
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(
        1, 1, figsize=(width * px, height * px), constrained_layout=True
    )

    for lbl in features.select(pl.col("face_label").unique()).to_series():
        au_act = features.select(pl.col("face_au_12")).to_series()
        au_act[
            features.select(pl.col("face_label")).to_series() != lbl
        ] = np.nan

        ax.plot(time, au_act, color=candidate_colors[lbl])

    lower_lim = time.min() if len(time) > 0 else 0
    upper_lim = time.max() if len(time) > 0 else 0
    ax.set_xlim(lower_lim, upper_lim)

    ax.set_yticks(np.arange(0, 1.2, step=0.2))
    ax.set_title("Lip corner puller (AU12) activation", loc="left")
    sns.despine(fig=fig, ax=ax, offset=10)
    fig.tight_layout()
    plt.close()
    return fig, ax


def create_pitch_f0_plot(features, width, height):
    """Create plot for voice pitch over time."""
    time = features.select(pl.col("time")).to_series()
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(
        1, 1, figsize=(width * px, height * px), constrained_layout=True
    )
    for lbl in features.select(
        pl.col("segment_speaker_label").unique()
    ).to_series():
        if lbl:
            pitch_f0 = features.select(pl.col("pitch_f0_hz")).to_series()
            pitch_f0[
                features.select(pl.col("segment_speaker_label")).to_series()
                != lbl
            ] = np.nan
            ax.plot(time, pitch_f0, color=candidate_colors[lbl])

    lower_lim = time.min() if len(time) > 0 else 0
    upper_lim = time.max() if len(time) > 0 else 0
    ax.set_xlim(lower_lim, upper_lim)

    ax.set_ylim((0, 400))
    ax.set_title("Voice pitch (F0 in Hz)", loc="left")
    sns.despine(fig=fig, ax=ax, offset=10)
    fig.tight_layout()
    plt.close()
    return fig, ax


def create_sent_plot(features, width, height):
    """Create plot for positive sentiment over time."""
    time = features.select(pl.col("time")).to_series()
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(
        1, 1, figsize=(width * px, height * px), constrained_layout=True
    )
    for lbl in features.select(
        pl.col("segment_speaker_label").unique()
    ).to_series():
        if lbl:
            span_sent_pos = features.select(pl.col("span_sent_pos")).to_series()
            span_sent_pos[
                features.select(pl.col("segment_speaker_label")).to_series()
                != lbl
            ] = np.nan
            ax.plot(time, span_sent_pos, color=candidate_colors[lbl])

    lower_lim = time.min() if len(time) > 0 else 0
    upper_lim = time.max() if len(time) > 0 else 0
    ax.set_xlim(lower_lim, upper_lim)

    ax.set_yticks(np.arange(0, 1.2, step=0.2))
    ax.set_title("Positive speech sentiment", loc="left")
    sns.despine(fig=fig, ax=ax, offset=10)
    fig.tight_layout()
    plt.close()
    return fig, ax


def fig_to_img(fig):
    """Convert a maplotlib figure to a image array (for openCV)."""
    # fig.tight_layout()
    fig.canvas.draw()
    fig_array = fig.canvas.buffer_rgba()
    return np.asarray(fig_array)[:, :, (2, 1, 0)]


# Get postprocessed feature df from video file
feature_df = pl.scan_csv(
    os.path.join(RESULTS_DIR, "mexca_segment_1_1864_2580_features_post.csv")
).collect()

# Define start and end of annotated video
t_start = 14
t_end = 24

# Open video stream
cap = cv2.VideoCapture(os.path.join(REF_DIR, "segment_1_1864_2580.mp4"))

# Get size for saving
width = int(cap.get(3))
height = int(cap.get(4))

# Adapt wiwdth to make space for feature plots
size = (int(width * (4 / 3)), height)

# Define font properties
font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 0.65
font_line_type = cv2.LINE_AA


# Open video writer
annotated = cv2.VideoWriter(
    os.path.join(PAPER_DIR, "video", "segment_1_1864_2580_annotated.mp4"),
    cv2.VideoWriter_fourcc(*"MJPG"),
    25,
    size,
)

# Number of frames to show at same time in feature plots
num_frames = 100

# Init couners
i = 0
t = 0.0

# Helper variable
valid_row = None

while cap.isOpened():
    # Read frame
    ret, frame = cap.read()

    # Stop if no input received
    if not ret:
        break

    if t > t_start and t < t_end:
        # Get matching outut from pipeline
        frame_row = feature_df.filter(pl.col("frame") == i)

        # Only update if there is a matching frame in the feauture df,
        # otherwise keep old features
        if frame_row.shape[0] > 0 or valid_row is None:
            # Update old valid features
            valid_row = frame_row

            # Start frame to include in feature plots
            start_frame = max((0, i - num_frames))

            # Subset of features to plot
            feature_to_frame_df = feature_df.filter(
                pl.col("frame").is_between(start_frame, i)
                & pl.col("time").ge(t_start)
                # Only select large faces (others might not be reliable)
                & (pl.col("face_box_y2") - pl.col("face_box_y1")).ge(70.0)
            )

            # Create feature plots
            au_plot, _ = create_au_plot(
                feature_to_frame_df, width=width / 3, height=height / 3
            )

            au_img = fig_to_img(au_plot)

            pitch_plot, _ = create_pitch_f0_plot(
                feature_to_frame_df, width=width / 3, height=height / 3
            )

            pitch_img = fig_to_img(pitch_plot)

            sent_plot, _ = create_sent_plot(
                feature_to_frame_df, width=width / 3, height=height / 3
            )

            sent_img = fig_to_img(sent_plot)

        draw_face_boxes(frame, valid_row)

        draw_speaker_labels(frame, valid_row)

        # Concatenate annotated video frame with feature plots
        frame_concat = cv2.hconcat(
            [
                frame,
                cv2.resize(
                    cv2.vconcat([au_img, pitch_img, sent_img]),
                    (int(width / 3), int(height)),
                    interpolation=cv2.INTER_AREA,
                ),
            ]
        )

        annotated.write(frame_concat)

    if t > t_end:
        break

    i += 1
    t += 0.04
    print(i, t)

cap.release()
annotated.release()
cv2.destroyAllWindows()
