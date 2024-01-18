# Example Analysis RTL Verkiezingsdebat

This folder contains code and result files to conduct an example analysis of the *RTL Verkiezingsdebat* (aired February 28 2021). Please understand that we currently cannot share the raw video files due to Copyright restrictions. We hope to resolve this in the future. Please contact m.luken@esciencecenter.nl for questions.

## Content

- `data/`: Folder for the (currently unavailable) raw data and annotation files.
- `figures/`: Folder for the figures resulting from the analysis.
- `resuts/`: Output from the MEXCA pipeline applied to the debate videos and the analysis. There are output files for each segment:
    - `*_features_post.csv`: Tabular merged feature data frame after postprocessing.
    - `*_features.csv`: Tabular merged feature data frame.
    - `*_sentiment.json`: Sentiment scores for each transcribed sentence.
    - `*_speaker_annotation.json`: Detected speech segments.
    - `*_transcription.srt`: Transcribed speech.
    - `*_video_annotation.json`: Features for detected faces.
    - `*_voice_features.json`: Voice features.
- `video/`: Folder for (currently unavailable) annotated video files.
    - `concat_large.sh`: Script for concatenating the screencast with the annotated video (high resolution).
    - `concat_small.sh`: Script for concatenating the screencast with the annotated video (low resolution).
    - `mexca_screencast.mp4`: Screencast of MEXCA pipeline example.
- `constants.py`: Hard coded constants for the analysis.
- `example_analysis.R`: Script for feature analysis (PCA).
- `helper_functions.py`: Helper functions.
- `postprocess_features.py`: Script for postprocessing merged feature data frames.
- `validate_results.py`: Script for calculating validation metrics (F1, DER, WER).
- `visualize_resutls.py`: Script for annotating a video with MEXCA output.

To reproduce the analysis install mexca and the additional requirements from `requirements.txt`. Then, run the Python scripts `postprocess_features.py` and `validate_results.py` in this order.
Finally, run the R script `example_analysis.R` to produce the PCA results. To create a video with annotated output from MEXCA, run `visualize_results.py`.
