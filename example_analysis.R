# Analysis script for the RTL Verkiezingsdebat 2021

library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(corrr)
library(purrr)
library(readr)
library(viridis)

# Get filenames of postprocessed feature CSV files
filenames = list.files('results', pattern = '*features_post.csv', full.names = TRUE)

# Load feature files as dfs for each segment
dfs = lapply(filenames, read_csv, col_select = -1, lazy = TRUE)

# Concatentate segment dfs
df_full = dfs %>%
	imap(function(df, i) {return(df %>% mutate(segment = i))}) %>%
	reduce(rbind.data.frame) %>%
	rowwise() %>%
	# Substitute empty strings with NA
	mutate(
		span_text = ifelse(span_text == '', NA, span_text)
	)


# Descriptives ------------------------------------------------------------

# Face displays
face_disp_df = df_full %>%
	filter(!face_label %in% c('mod_m', NA)) %>%
	group_by(face_label, segment) %>%
	summarize(frames_disp = n(),
			  time_disp = frames_disp * 0.04)

ggplot(face_disp_df, aes(x = face_label, y = time_disp)) +
	facet_wrap(vars(segment)) +
	geom_col()


# Speaker time
speaker_time_df = df_full %>%
	filter(!segment_speaker_label %in% c('mod_m', NA)) %>%
	group_by(segment_speaker_label, segment) %>%
	summarise(frames_spk = n(),
			  time_spk = frames_spk * 0.04)

ggplot(speaker_time_df, aes(x = segment_speaker_label, y = time_spk, fill = segment_speaker_label)) +
	facet_wrap(vars(segment)) +
	geom_col()


# Sentence length
sent_len_df = df_full %>%
	filter(!segment_speaker_label %in% c('mod_m', NA)) %>%
	group_by(span_start, span_end, segment_speaker_label, segment) %>%
	mutate(sent_dur = span_end - span_start,
		   sent_val = qnorm(span_sent_pos) - qnorm(span_sent_neg))

ggplot(sent_len_df, aes(x = segment_speaker_label, y = sent_dur, fill = segment_speaker_label)) +
	facet_wrap(vars(segment)) +
	geom_violin() +
	geom_point()

# Sentence sentiment
ggplot(sent_len_df, aes(x = sent_dur, y = qnorm(span_sent_pos), color = segment_speaker_label)) +
	facet_wrap(vars(segment_speaker_label)) +
	geom_smooth(formula = y ~ x) +
	geom_point()

ggplot(sent_len_df, aes(x = sent_dur, y = qnorm(span_sent_neg), color = segment_speaker_label)) +
	facet_wrap(vars(segment_speaker_label)) +
	geom_smooth(formula = y ~ x) +
	geom_point()

# Analysis ----------------------------------------------------------------

au_refs = c(4, 6, 7, 10, 12, 14, 17, 25)

# Frames speaking vs not speaking
same_id_df = df_full %>%
	filter(!is.na(face_label), !is.na(segment_speaker_label), face_label == segment_speaker_label)

diff_id_df = df_full %>%
	filter(!is.na(face_label), !is.na(segment_speaker_label), face_label != segment_speaker_label)

nrow(same_id_df)/nrow(df_full)
nrow(diff_id_df)/nrow(df_full)

# Plot AU activations for speaking vs not speaking
ggplot(df_full %>%
	   	filter(!is.na(face_label), !is.na(segment_speaker_label), face_label != "mod_m", segment_speaker_label != "mod_m") %>%
	   	group_by(face_label, segment_speaker_label) %>%
	   	select(starts_with("face_au") & !contains(c("L", "R"))) %>%
	   	mutate(face_is_speaker = face_label == segment_speaker_label) %>%
	   	group_by(face_label, face_is_speaker) %>%
	   	summarize(across(face_au_4:face_au_39, ~ mean(.x, na.rm = TRUE))) %>%
	   	pivot_longer(cols = face_au_4:face_au_39, names_to = "feature")
	   , aes(x = value, y = feature, color = face_is_speaker)) +
	facet_grid(cols = vars(face_label)) +
	geom_line(aes(group = feature), color = "grey") +
	geom_point() +
	scale_color_viridis_d()


# Create data frame with selected features
df_sub = df_full %>%
	filter(
		!is.na(face_label),
		!is.na(segment_speaker_label),
		# Exclude moderator
		face_label != "mod_m",
		segment_speaker_label != "mod_m",
		# Exclude faces that are too small
		face_box_y2 - face_box_y1 > 45.0 & face_box_x2 - face_box_x1 > 45.0
	) %>%
	group_by(face_label, segment_speaker_label) %>%
	# Only select reliable AUs
	select(
		starts_with("face_au") & contains(as.character(au_refs)) & !contains(c("L", "R")),
		-c(face_au_26, face_au_27, face_au_16, face_au_24),
		pitch_f0_hz:rms_db, starts_with("span")
	) %>%
	mutate(
		# Transform AU activations with probit
		across(starts_with(c("face_au")), qnorm),
		span_length = span_end - span_start,
		# Multiply sentiment by sentence length
		across(starts_with("span_sent"), ~ .x * span_length)
	) %>%
	# Remove frames with missing values (i.e., no speech or faces)
	select(!where(~ all(is.na(.x)))) %>%
	ungroup()

# PCA for face AUs
pca_est_face = prcomp(
	df_sub %>%
		ungroup() %>%
		select(where(is.numeric) & starts_with("face_au")) %>%
		# Remove duplicates
		distinct(),
	center = TRUE, scale. = TRUE, rank. = NULL
)

summary(pca_est_face)

# Kaiser's rule: Eigenvalues greater than 1
n_pc_face = sum(pca_est_face$sdev^2 > 1)

# PCA for voice features
pca_est_voice = prcomp(
	df_sub %>%
		ungroup() %>%
		select(segment_speaker_label, pitch_f0_hz:rms_db) %>%
		distinct() %>%
		select(pitch_f0_hz:rms_db) %>%
		drop_na(),
	center = TRUE, scale. = TRUE, rank. = NULL
)

summary(pca_est_voice)

# Set to 5 because PCs 6 and 7 are hard to interpret and not relevant for example
n_pc_voice = 5 # sum(pca_est_voice$sdev^2 > 1)

# Plot face AU PC scores
ggplot(
	# Compute PC scores
	as_tibble(
		pca_est_face$rotation[,1:n_pc_face] %*% diag(pca_est_face$sdev, n_pc_face, n_pc_face)
	) %>%
		mutate(name = row.names(pca_est_face$rotation[,1:n_pc_face])) %>%
		# Make long format for plotting
		pivot_longer(cols = where(is.numeric), names_to = "pc_id"),
	aes(x = pc_id, y = name, fill = value)
) +
	geom_tile(colour = "black") +
	geom_text(aes(label = round(value, digits = 2))) +
	# Add explained variance to break labels
	scale_x_discrete(labels = paste0(
		"PC", 1:n_pc_face, " (", round(100*pca_est_face$sdev[1:n_pc_face]^2/sum(pca_est_face$sdev^2), digits = 1), "%)"
	)) +
	scale_fill_viridis_c(limits = c(-1, 1)) +
	labs(x = "Principal component", y = "Feature", fill = "Loading") +
	theme(text = element_text(size = 12),
		  panel.background = element_blank())

ggsave("figures/pca_face.png", width = 5, height = 3.5)

# Plot voice PC scores
ggplot(
	as_tibble(
		pca_est_voice$rotation[,1:n_pc_voice] %*% diag(pca_est_voice$sdev, n_pc_voice, n_pc_voice)
	) %>%
		mutate(name = row.names(pca_est_voice$rotation[,1:n_pc_voice])) %>%
		pivot_longer(cols = where(is.numeric), names_to = "pc_id"),
	aes(x = pc_id, y = name, fill = value)) +
	geom_tile(colour = "black") +
	geom_text(aes(label = round(value, digits = 2)), size = 3) +
	scale_fill_viridis_c(limits = c(-1, 1)) +
	scale_x_discrete(labels = paste0(
		"PC", 1:n_pc_voice, "\n(", round(100*pca_est_voice$sdev[1:n_pc_voice]^2/sum(pca_est_voice$sdev^2), digits = 1), "%)")
	) +
	labs(x = "Principal component", y = "Feature", fill = "Loading") +
	theme(text = element_text(size = 8),
		  panel.background = element_blank())

ggsave("figures/pca_voice.png", width = 5, height = 3.5)


# Mean face PC scores for each face label
df_inter_face = df_sub %>%
	ungroup() %>% select(face_label, where(is.numeric) & starts_with("face_au")) %>% distinct() %>%
	select(face_label) %>%
	cbind(as.data.frame(pca_est_face[["x"]][,1:n_pc_face]) %>% rename_with(~ paste0("face_", .x))) %>%
	rename(label = face_label) %>%
	group_by(label) %>%
	summarise(across(where(is.numeric), list(
		mean = ~ mean(.x, na.rm = TRUE),
		ste = ~ sd(.x, na.rm = TRUE)/sqrt(length(.x))
	)))

# Mean voice PC scores for each speaker label
df_inter_voice = df_sub %>%
	select(segment_speaker_label, pitch_f0_hz:rms_db) %>%
	distinct() %>%
	drop_na() %>%
	select(segment_speaker_label) %>%
	cbind(as.data.frame(pca_est_voice[["x"]][,1:n_pc_voice]) %>% rename_with(~ paste0("voice_", .x))) %>%
	rename(label = segment_speaker_label) %>%
	group_by(label) %>%
	summarise(across(where(is.numeric), list(
		mean = ~ mean(.x, na.rm = TRUE),
		ste = ~ sd(.x, na.rm = TRUE)/sqrt(length(.x))
	)))

# Mean sentiment scores for each speaker label
df_inter_text = df_sub %>%
	select(segment_speaker_label, starts_with("span_sent")) %>%
	distinct() %>%
	mutate(across(starts_with("span_sent"), ~ scale(.x, scale = TRUE))) %>%
	rename(label = segment_speaker_label) %>%
	group_by(label) %>%
	summarise(across(where(is.numeric), list(
		mean = ~ mean(.x, na.rm = TRUE),
		ste = ~ sd(.x, na.rm = TRUE)/sqrt(length(.x))
	)))

# Plot mean scores for all PC and sentiment scores aggregated over labels
ggplot(
	# Join mean dfs
	left_join(df_inter_face, left_join(df_inter_voice, df_inter_text, by = "label"), by = "label") %>%
		# To long format
		pivot_longer(cols = where(is.numeric), names_to = c("feature", "type"), names_pattern = "(.*)_(.*)") %>%
		pivot_wider(id_cols = c(label, feature), names_from = "type", values_from = "value") %>%
		# Create group var for color
		mutate(group = if_else(grepl("face", feature), "face",
							   if_else(grepl("voice", feature), "voice", "text"))),
	aes(x = mean, y = feature, group = group, fill = group)
) +
	facet_grid(cols = vars(label), labeller = as_labeller(str_to_title)) +
	geom_col(aes(x = mean, y = feature)) +
	geom_errorbarh(aes(xmin = mean - 1.96*ste, xmax = mean + 1.96*ste, y = feature),
				   height = 0.4, linewidth = 0.2, color = "black") +
	scale_x_continuous(limits = c(-1.5, 1.5), breaks = seq(-1.5, 1.5, 0.5), labels = ~ ifelse(.x %% 1 == 0, .x, "")) +
	scale_fill_viridis_d(begin = 0.3, labels = str_to_title) +
	geom_hline(yintercept = c(n_pc_face + 0.5, n_pc_face + 3.5), color = "grey", alpha = 0.3) +
	geom_vline(xintercept = 0, color = "grey", linewidth = 0.3) +
	labs(x = "Mean (95% CI)", y = "Feature", fill = "Modality") +
	theme(text = element_text(size = 7),
		  axis.line.x = element_line(),
		  panel.background = element_blank())

ggsave("figures/inter_speaker_features.png", width = 5, height = 3.5)
