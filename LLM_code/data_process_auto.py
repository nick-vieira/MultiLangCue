import argparse
import pickle
import json
import os 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.utils import resample

sys.path.insert(0, '../feature_utils')

from postprocess_audio_feature_iemocap import (
    extract_thresholds_and_stats as extract_iemocap_thresholds,
    standardize_and_process_df      as standardize_iemocap_df,
    generate_concise_description    as gen_iemocap_desc,
    generate_impression             as gen_iemocap_imp,
)

from postprocess_audio_feature_meld import (
    extract_thresholds_and_stats as extract_meld_thresholds,
    standardize_and_process_csv as standardize_meld_df,
    generate_concise_description as gen_meld_desc,
    generate_impression          as gen_meld_imp,
    preprocess_speaker_df        as prep_meld_speakers,
)

scalers = {}

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in data.iterrows():
            f.write(json.dumps({
                'path': row['audio_path'],
                'input': row['text'],
                'target': row['emotion']
            }, ensure_ascii=False) + '\n')

def save_json_meld(df, filename, emotional_dict, num_classes):
    # Convert one-hot to str
    id_to_label = {v: k for k, v in emotional_dict.items()}

    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            raw_label = row['emotion']
            # Find ground truth position in one-hot vector
            if isinstance(raw_label, list):
                if len(raw_label) != num_classes:
                    raise ValueError(f"One-hotlength {len(raw_label)} != expected {num_classes}")
                try:
                    idx = raw_label.index(1)
                except ValueError:
                    raise ValueError(f"Invalid one-hot vector: {raw_label!r}")
                emotion_label = id_to_label.get(idx)
                if emotion_label is None:
                    raise KeyError(f"No label for index {idx}")
            elif isinstance(raw_label, str):
                emotion_label = raw_label
            else:
                raise ValueError(f"Unexpected emotion value: {raw_label!r}")

            out = {
                'path':  row['audio_path'],
                'input': row['text'],
                'target': emotion_label
            }
            f.write(json.dumps(out, ensure_ascii=False) + '\n')

def scale_group(group):

    cols = ['avg_pitch', 'pitch_std', 'avg_intensity', 'mean_hnr', 'intensity_variation', 'pitch_range']

    if group[cols].isnull().any().any():
        return group  # Skip if any NaNs

    if group[cols].shape[0] < 2:
        return group  # Not enough samples to compute std

    if not all(col in group for col in cols):
        return group

    scaler = StandardScaler()
    try:
        scaled = scaler.fit_transform(group[cols])
        if scaled.shape[1] != len(cols):
            return group  # Skip if scaling result is unexpected
        group[cols] = scaled
        scalers[group.name] = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
    except Exception as e:
        print(f"Skipping group {group.name} due to error: {e}")
        return group

    return group

def split_esd_mandarin(df: pd.DataFrame):

    sort_cols = ["speaker_id", "emotion"]
    if "file_id" in df.columns:
        sort_cols.append("file_id")            # keep numeric order if present
    elif "file_path" in df.columns:
        sort_cols.append("file_path")          # fall back to path name

    ordered = df.sort_values(sort_cols, ascending=True).reset_index(drop=True)

    valid_parts, test_parts, train_parts = [], [], []

    for _, group in ordered.groupby(["speaker_id", "emotion"], sort=False):
        # ensure deterministic row order within the group
        group = group.reset_index(drop=True)

        valid_parts.append(group.iloc[:20])
        test_parts.append(group.iloc[20:50])
        train_parts.append(group.iloc[50:350])

    valid_df = pd.concat(valid_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)
    train_df = pd.concat(train_parts, ignore_index=True)

    return train_df, test_df, valid_df

def generate_prose_prompt(pitch, pstd, ivar, hnr, intensity, emotion_set):
    FEW_SHOT_EXAMPLES = (
        # ENGLISH DATASETS
        "Example 1:\n"
        "avg_pitch_standardized=-0.41, pitch_std_standardized=-0.40, intensity_variation_standardized=0.03, mean_hnr_standardized=0.07, avg_intensity_standardized=1.04\n"
        "Prediction: This utterance is tense, sharp, and forceful, which tends to represent ANGER.\n\n"
        "Example 2:\n"
        "avg_pitch_standardized=-1.27, pitch_std_standardized=0.05, intensity_variation_standardized=-0.87, mean_hnr_standardized=-0.68, avg_intensity_standardized=-0.50\n"
        "Prediction: This utterance is flat, subdued, and calm, which is typical of SADNESS.\n\n"
        "Example 3:\n"
        "avg_pitch_standardized=-0.31, pitch_std_standardized=-1.19, intensity_variation_standardized=0.06, mean_hnr_standardized=-1.85, avg_intensity_standardized=0.67\n"
        "Prediction: This utterance is bright, lively, and resonant, which may be indicative of HAPPINESS.\n"

        # EMODB
        # "Example 1:\n"
        # "pitch=190.0, pitch_std=65.0, intensity_variation=13.0, mean_hnr=14.5, avg_intensity=72.0\n"
        # "Prediction: This utterance is bright, smooth, and expressive, which may be indicative of HAPPINESS.\n\n"
        # "Example 2:\n"
        # "pitch=110.0, pitch_std=35.0, intensity_variation=9.0, mean_hnr=17.2, avg_intensity=60.0\n"
        # "Prediction: This utterance is flat and low-pitched, which is typical of SADNESS.\n\n"
        # "Example 3:\n"
        # "pitch=165.0, pitch_std=50.0, intensity_variation=14.2, mean_hnr=11.0, avg_intensity=75.0\n"
        # "Prediction: This utterance is harsh and intense, which tend to represent ANGER.\n\n"

        # ESD
        # "Example 1:\n"
        # "pitch=260.0, pitch_std=60.0, intensity_variation=21.0, mean_hnr=10.5, avg_intensity=69.0\n"
        # "Prediction: This utterance is sharp, volatile, and forceful, which tend to represent ANGER..\n\n"
        # "Example 2:\n"
        # "pitch=270.0, pitch_std=65.0, intensity_variation=21.5, mean_hnr=13.2, avg_intensity=68.0\n"
        # "Prediction: This utterance is bright, lively, and resonant, which may be indicative of HAPPY.\n\n"
        # "Example 3:\n"
        # "pitch=175.0, pitch_std=38.0, intensity_variation=15.0, mean_hnr=14.1, avg_intensity=58.0\n"
        # "Prediction: This utterance is deep, flat, and soft, which is typical of SAD.\n\n"
    )
    
    prose_prompt = (
        "You are an expert in vocal emotion analysis. "
        "Below are examples describing typical emotional expressions based on acoustic features.\n\n"
        f"{FEW_SHOT_EXAMPLES}"
        "Now analyze this:\n"
        f"pitch={pitch:.2f}, pitch_std={pstd:.2f}, intensity_variation={ivar:.2f}, mean_hnr={hnr:.2f}, avg_intensity={intensity:.2f}\n"
        "Please respond in this format:\n"
        "Prediction: This utterance is ... — typical of <EMOTION>.\n"
    )

    return prose_prompt

import pandas as pd

def get_previous_turns_text(
    df: pd.DataFrame,
    video_id: str,
    current_utt: int,
    window: int = 3
) -> str:
    
    # Filter by dialogue ID
    convo = (
        df[df['video_id'] == video_id]
        .sort_values('segment_id')
        [['segment_id', 'text']]
    )

    # Keep only turns before the current one
    prior = convo[convo['segment_id'] < current_utt]
    last_k = prior.tail(window)
    
    # Formatted as sequential dialogue lines
    lines = [
        f"Turn {row.segment_id}: {row.text}"
        for row in last_k.itertuples(index=False)
    ]
    return "\n".join(lines)


def process_dataset(dataset, window=110, audio_description='True', audio_impression='False', audio_only='False', audio_context='False',experiments_setting='lora'):
    '''
    dataset: parameter that define the evaluated dataset.
    window:       parameter that control the historical context window
    speaker_mode: parameter that control the speaker identification task whether to be added in the main task

    data_path:    parameter that record the processed dataset's filepath
    '''
    label_set = {
        'iemocap': ['hap', 'sad', 'neu', 'ang', 'exc', 'fru', 'xxx', 'sur', 'dis', 'fea'],
        'meld':   ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
        'emodb': ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'sadness', 'neutral'],
        'esd': ['anger', 'happy', 'sad', 'neutral', 'surprise'],
    }
    label_text_set = {
        'iemocap':  'hap, sad, neu, ang, exc, fru, xxx, sur, dis, fea',
        'meld': 'neutral, surprise, fear, sadness, joy, disgust, anger',
        'emodb': 'anger, boredom, disgust, fear, happiness, sadness, neutral',
        'esd': 'anger, surprise, happy, sad, neutral',
    }

    ## load audio feature files
    if dataset == 'meld':
        audio_feature_train = pd.read_csv('../speech_features/meld_train_audio_features.csv')
        audio_feature_test = pd.read_csv('../speech_features/meld_test_audio_features.csv')
        audio_feature_dev = pd.read_csv('../speech_features/meld_dev_audio_features.csv')
    elif dataset == 'iemocap':
        audio_feature = pd.read_csv('../speech_features/iemocap_audio_features.csv')
        iemocap_file = pd.read_csv('../data/IEMOCAP_full_release/iemocap_full_dataset.csv')
    elif dataset == 'emodb':
        audio_feature = pd.read_csv('../speech_features/processed_emodb_audio_features.csv')
        # audio_feature = audio_feature[audio_feature['emotion'].isin(['anger', 'neutral'])]
    elif dataset == 'esd':
        audio_feature = pd.read_csv('../speech_features/processed_esd_mandarin_features.csv')
    ###

    emotional_dict = {text_label:num_label for num_label, text_label in enumerate(label_set[dataset])}
    # emotional_dict = {'happy': 0, 'neutral': 1}  # <- REDUCED LABEL SET
    content_target_dict = {}
    content_task_dict = {}
    speaker_label_dict = {}
    sentence_dict = {}
    audio_path_dict = {}
    length = []

        
    if dataset == 'iemocap':
        num_classes = 5
        train_df = audio_feature[audio_feature['mode'] == 'train']
        thresholds, stats = extract_iemocap_thresholds(train_df, num_classes)
        processed_df = standardize_iemocap_df(audio_feature, thresholds, stats, num_classes)
        # num_to_str = {
        #     0: 'very low',
        #     1: 'low',
        #     2: 'medium',
        #     3: 'high',
        #     4: 'very high',
        # }

        # for feat in ('avg_pitch', 'pitch_std', 'intensity_variation', 'mean_hnr', 'avg_intensity'):
        #     cat_col = feat + '_category'
        #     # print("avg_pitch_category values:", audio_feature['avg_pitch_category'].unique())
        #     if cat_col in processed_df:
        #         processed_df[cat_col] = processed_df[cat_col].map(num_to_str).fillna(processed_df[cat_col])
        processed_df['description'] = processed_df.apply(lambda r: gen_iemocap_desc(r, num_classes), axis=1)
        processed_df['impression'] = processed_df.apply(lambda r: gen_iemocap_imp(r, num_classes), axis=1)

        original_features = ["avg_intensity", "intensity_variation", "avg_pitch", 
                         "pitch_std", "pitch_range", "articulation_rate", "mean_hnr"]
    
        columns_to_keep = [col for col in processed_df.columns if col not in original_features]
        
        # Keep only needed audio features
        output_df = processed_df[columns_to_keep]
        output_df.to_csv('../speech_features/processed_iemocap_with_desc_imp.csv', index=False)
        audio_feature = pd.read_csv('../speech_features/processed_iemocap_with_desc_imp.csv')
        audio_feature = audio_feature.merge(
            iemocap_file[['video_id', 'text', 'path']],
            on=['video_id', 'text'],
            how='left'
        )

        audio_feature.dropna(subset=[
            'avg_pitch_standardized', 'pitch_std_standardized', 'avg_intensity_standardized', 
            'mean_hnr_standardized', 'intensity_variation_standardized', 'pitch_range_standardized'],
            inplace=True)
        
        # audio_feature = audio_feature.groupby('speaker_id', group_keys=False).apply(scale_group)
        audio_feature = audio_feature[audio_feature['emotion'].str.lower() != 'xxx']
        content_target_dict = {}
        content_task_dict = {}
        audio_path_dict = {}
        
        # Process dataset
        data_list = []
        prose_count = 0
        
        for idx, row in audio_feature.iterrows():
            all_conv_id = row['video_id'].split('.')[0]
            emotion_label = row['emotion'].lower()

            prose_rule = ""
            # speaker_id = row['speaker_id']
            speaker_id = row['segment_id'] # For RunPod

            # scale = scalers[speaker_id]['scale']
            # mean = scalers[speaker_id]['mean']

            hnr = row['mean_hnr_standardized']
            ivar = row['intensity_variation_standardized']
            pitch = row['avg_pitch_standardized']
            pstd = row['pitch_std_standardized']
            intensity = row['avg_intensity_standardized']

            prose_prompt = generate_prose_prompt(
                pitch, pstd, ivar, hnr, intensity, label_text_set[dataset]
            )
            history_text = get_previous_turns_text(
                audio_feature, 
                video_id=row['video_id'], 
                current_utt=row['segment_id'], 
                window=window
            )

            text = (
                (history_text + "\n\n") if history_text else ""
            ) + prose_prompt \
            + f"Description: {row.get('description','unknown')}\n" \
            + f"Impression: {row.get('impression','unknown')}\n" \
            + f"Please classify the emotional label of this utterance from <{label_text_set[dataset]}>.\n"
        
            data_list.append({
                'conv_id': row['video_id'].split('.')[0],
                'text': text,
                'emotion': emotion_label,
                'audio_path': row['path']  # Still used for audio, but not in model input
            })

        data = pickle.load(open(f'../original_data/{dataset}/{dataset}.pkl','rb'))
        all_conv_id = data[3] + data[4] + data[5]
        sentence_dict = data[2]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                if speaker_label == 'M':
                    temp_speaker_list.append(0)
                else:
                    temp_speaker_list.append(1)
            speaker_label_dict[conv_id] = temp_speaker_list

        data_df = pd.DataFrame(data_list)
        
        train_data, test_valid = train_test_split(data_df, test_size=0.2, stratify=data_df['emotion'], random_state=42)
        counts = test_valid['emotion'].value_counts()
        rare = counts[counts < 2].index.tolist()
        if not rare:
            test_data, valid_data = train_test_split(test_valid, test_size=0.5, stratify=test_valid['emotion'], random_state=42)
        else:
            test_data, valid_data = train_test_split(test_valid, test_size=0.5, random_state=42)

        data_path = f'../PROCESSED_DATASET/{dataset}/window/{audio_description}_{audio_impression}'
        os.makedirs(data_path, exist_ok=True)

        save_json(train_data, os.path.join(data_path, 'train.json'))
        save_json(test_data, os.path.join(data_path, 'test.json'))
        save_json(valid_data, os.path.join(data_path, 'valid.json'))

        return data_path
    
    elif dataset == 'meld':
        audio_feature_train_csv = '../original_data/meld/train/train_sent_emo.csv'
        num_classes = 7
        feature_csv_path = '../speech_features/meld_train_audio_features.csv'
        speaker_csv_path = audio_feature_train_csv
        thresholds, stats = extract_meld_thresholds(feature_csv_path, speaker_csv_path, num_classes)

        for mode in ['train', 'dev', 'test']:
            csv_file = f'../speech_features/meld_{mode}_audio_features.csv'
            speaker_csv = f'../original_data/meld/{mode}/{mode}_sent_emo.csv'
            processed_df = standardize_meld_df(csv_file, thresholds, stats, speaker_csv, num_classes)
            processed_df['description'] = processed_df.apply(lambda r: gen_meld_desc(r, num_classes), axis=1)
            processed_df['impression'] = processed_df.apply(lambda r: gen_meld_desc(r, num_classes), axis=1)

            processed_df['dialog_utt'] = (
                processed_df['Dialogue_ID'].astype(str).str.zfill(4)
                + '_'
                + processed_df['Utterance_ID'].astype(str).str.zfill(2)
            )
            
            speaker_meta = pd.read_csv(speaker_csv)
            speaker_meta['dialog_utt'] = (
                speaker_meta['Dialogue_ID'].astype(str).str.zfill(4)
                + '_'
                + speaker_meta['Utterance_ID'].astype(str).str.zfill(2)
            )

            processed_df = processed_df.merge(
                speaker_meta[['dialog_utt','Emotion']],
                on='dialog_utt',
                how='left'
            )
            processed_df.rename(columns={'Emotion':'emotion'}, inplace=True)
            processed_df.dropna(subset=['emotion'], inplace=True)
            # processed_df.drop(columns=['filename_no_ext'], inplace=True)
            original_features = ["avg_intensity", "intensity_variation", "avg_pitch", 
                            "pitch_std", "pitch_range", "articulation_rate", "mean_hnr"]
        
            columns_to_keep = [col for col in processed_df.columns if col not in original_features]
            
            # Keep only set audio features
            output_df = processed_df[columns_to_keep]
            output_file =f'../speech_features/processed_with_desc_and_imp_{os.path.basename(csv_file)}'
            output_df.to_csv(output_file, index=False)

        # right after you read the three processed CSVs
        df_train = pd.read_csv('../speech_features/processed_with_desc_and_imp_meld_train_audio_features.csv')
        df_dev = pd.read_csv('../speech_features/processed_with_desc_and_imp_meld_dev_audio_features.csv')
        df_test = pd.read_csv('../speech_features/processed_with_desc_and_imp_meld_test_audio_features.csv')


        audio_feature = pd.concat([df_train, df_dev, df_test], ignore_index=True)
        # processed_df.dropna(subset=['Emotion'], inplace=True)
        # print("FULL SET:", audio_feature.shape, audio_feature.columns.tolist())
        data_list = []

        for idx, row in audio_feature.iterrows():
            all_conv_id = row['filename'].split('.')[0]
            emotion_label = row['emotion'].lower()

            prose_rule = ""
            # speaker_id = row['speaker_id']

            hnr = row['mean_hnr_standardized']
            ivar = row['intensity_variation_standardized']
            pitch = row['avg_pitch_standardized']
            pstd = row['pitch_std_standardized']
            intensity = row['avg_intensity_standardized']

            prose_prompt = generate_prose_prompt(
                pitch, pstd, ivar, hnr, intensity, label_text_set[dataset]
            )

            text = prose_prompt + "\n" + (
                # f"Pitch: {row.get('avg_pitch', 'unknown')}, Variation: {row.get('pitch_std', 'unknown')}, "
                # f"Intensity: {row.get('avg_intensity', 'unknown')}\n"
                f"Please classify the emotional label of this utterance from <{label_text_set[dataset]}>.\n"
            )
        
            data_list.append({
                # 'conv_id': row['filename'].split('.')[0],
                'text': text,
                'emotion': emotion_label,
                'audio_path': row['filename']  # Still used for audio, but not in model input
            })

        data_df = pd.DataFrame(data_list)
        
        train_data, test_valid = train_test_split(data_df, test_size=0.2, stratify=data_df['emotion'], random_state=42)
        test_data, valid_data = train_test_split(test_valid, test_size=0.5, stratify=test_valid['emotion'], random_state=42)
        data_path = f'../PROCESSED_DATASET/{dataset}/window/{audio_description}_{audio_impression}'
        os.makedirs(data_path, exist_ok=True)

        dfs = []
        max_n = train_data[train_data['emotion']=='neutral'].shape[0]
        for emo, g in data_df.groupby('emotion'):
            if emo != 'neutral':
                dfs.append(resample(g, replace=True, n_samples=max_n, random_state=42))
            else:
                dfs.append(g)
        train_balanced = pd.concat(dfs).sample(frac=1, random_state=42)

        save_json(train_balanced, os.path.join(data_path, 'train.json'))
        save_json(test_data,  os.path.join(data_path, 'test.json'))
        save_json(valid_data, os.path.join(data_path, 'valid.json'))

        # Plot histogram
        plt.hist(data_df['text'].apply(len), bins=20)
        plt.xlabel('Length of the input text')
        plt.ylabel('Frequency')
        plt.title('Histogram of the length of the input text')
        plt.savefig(os.path.join(data_path, 'histogram.png'))

        return data_path

    elif dataset == 'emodb':
        audio_feature.dropna(subset=['avg_pitch', 'pitch_std', 'avg_intensity', 'mean_hnr', 'intensity_variation', 'pitch_range'], inplace=True)
        scaler = StandardScaler()
        audio_feature = audio_feature.groupby('speaker_id', group_keys=False).apply(scale_group)

        # Single-class label text set reductions
        # label_text_set['emodb'] = 'anger, neutral'
        # label_text_set['emodb'] = 'sadness, neutral'
        # label_text_set['emodb'] = 'happiness, neutral'
        content_target_dict = {}
        content_task_dict = {}
        audio_path_dict = {}
        
        # Process dataset
        data_list = []
        prose_count = 0
        
        for idx, row in audio_feature.iterrows():
            all_conv_id = row['filename'].split('.')[0]
            emotion_label = row['emotion'].lower()

            prose_rule = ""
            speaker_id = row['speaker_id']

            scale = scalers[speaker_id]['scale']
            mean = scalers[speaker_id]['mean']

            hnr = row['mean_hnr'] * scale[3] + mean[3]
            ivar = row['intensity_variation'] * scale[4] + mean[4]
            pitch = row['avg_pitch'] * scale[0] + mean[0]
            pstd = row['pitch_std'] * scale[1] + mean[1]
            intensity = row['avg_intensity'] * scale[2] + mean[2]

            prose_prompt = generate_prose_prompt(
                pitch, pstd, ivar, hnr, intensity, label_text_set[dataset]
            )

            text = prose_prompt + "\n" + (
                # f"Pitch: {row.get('avg_pitch', 'unknown')}, Variation: {row.get('pitch_std', 'unknown')}, "
                # f"Intensity: {row.get('avg_intensity', 'unknown')}\n"
                f"Please classify the emotional label of this utterance from <{label_text_set[dataset]}>.\n"
            )
        
            data_list.append({
                'conv_id': row['filename'].split('.')[0],
                'text': text,
                'emotion': emotion_label,
                'audio_path': row['filename']  # Still used for audio, but not in model input
            })

        data_df = pd.DataFrame(data_list)

        train_data, test_valid = train_test_split(data_df, test_size=0.2, stratify=data_df['emotion'], random_state=42)
        test_data, valid_data = train_test_split(test_valid, test_size=0.5, stratify=test_valid['emotion'], random_state=42)

        # Prose rule evaluation
        data_path = f'../PROCESSED_DATASET/{dataset}/cleaned/no_leakage'

        # No prose rule evaluation 
        # data_path = f'../PROCESSED_DATASET/{dataset}/cleaned/no_leakage_no_prose'
        
        os.makedirs(data_path, exist_ok=True)

        save_json(train_data, os.path.join(data_path, 'train.json'))
        save_json(test_data, os.path.join(data_path, 'test.json'))
        save_json(valid_data, os.path.join(data_path, 'valid.json'))

        # Plot histogram
        plt.hist(data_df['text'].apply(len), bins=20)
        plt.xlabel('Length of the input text')
        plt.ylabel('Frequency')
        plt.title('Histogram of the length of the input text')
        plt.savefig(os.path.join(data_path, 'histogram.png'))

        return data_path

    elif dataset == 'esd':
        audio_feature.dropna(subset=['avg_pitch', 'pitch_std', 'avg_intensity', 'mean_hnr', 'intensity_variation', 'pitch_range'], inplace=True)
        scaler = StandardScaler()
        audio_feature = audio_feature.groupby('speaker_id', group_keys=False).apply(scale_group)

        # Single-class label text set reductions
        # label_text_set['esd'] = 'anger, neutral'
        # label_text_set['esd'] = 'sad, neutral'
        # label_text_set['esd'] = 'happy, neutral'
    
        content_target_dict = {}
        content_task_dict = {}
        audio_path_dict = {}
        
        # Process dataset
        data_list = []
        prose_count = 0
        
        for idx, row in audio_feature.iterrows():
            all_conv_id = row['filename'].split('.')[0]
            emotion_label = row['emotion'].lower()

            prose_rule = ""
            speaker_id = row['speaker_id']

            scale = scalers[speaker_id]['scale']
            mean = scalers[speaker_id]['mean']

            hnr = row['mean_hnr'] * scale[3] + mean[3]
            ivar = row['intensity_variation'] * scale[4] + mean[4]
            pitch = row['avg_pitch'] * scale[0] + mean[0]
            pstd = row['pitch_std'] * scale[1] + mean[1]
            intensity = row['avg_intensity'] * scale[2] + mean[2]

            prose_prompt = generate_prose_prompt(
                pitch, pstd, ivar, hnr, intensity, label_text_set[dataset]
            )

            text = prose_prompt + "\n" + (
                # f"Pitch: {row.get('avg_pitch', 'unknown')}, Variation: {row.get('pitch_std', 'unknown')}, "
                # f"Intensity: {row.get('avg_intensity', 'unknown')}\n"
                f"Please classify the emotional label of this utterance from <{label_text_set[dataset]}>.\n"
            )

            # text = (
            #     f"Pitch: {row.get('avg_pitch', 'unknown')}, Variation: {row.get('pitch_std', 'unknown')}, "
            #     f"Intensity: {row.get('avg_intensity', 'unknown')}\n"
            #     f"Please classify the emotional label of this utterance from <{label_text_set[dataset]}>.\n"
            # )
        
            data_list.append({
                'conv_id': row['filename'].split('.')[0],
                'text': text,
                'emotion': emotion_label,
                'audio_path': row['filename'],
                'speaker_id': speaker_id
            })

        data_df = pd.DataFrame(data_list)

        train_data, test_data, valid_data = split_esd_mandarin(data_df)

        # Prose rule evaluation
        # data_path = f'../PROCESSED_DATASET/{dataset}/cleaned/no_leakage'

        # No prose rule evaluation 
        data_path = f'../PROCESSED_DATASET/{dataset}/cleaned/no_leakage_no_prose'
        
        os.makedirs(data_path, exist_ok=True)

        save_json(train_data, os.path.join(data_path, 'train.json'))
        save_json(test_data, os.path.join(data_path, 'test.json'))
        save_json(valid_data, os.path.join(data_path, 'valid.json'))

        # Plot histogram
        plt.hist(data_df['text'].apply(len), bins=20)
        plt.xlabel('Length of the input text')
        plt.ylabel('Frequency')
        plt.title('Histogram of the length of the input text')
        plt.savefig(os.path.join(data_path, 'histogram.png'))

        return data_path

    if dataset == 'iemocap':
        train_ids, test_ids, valid_ids = data[3], data[4], data[5]
    elif dataset == 'meld':
        train_ids, test_ids, valid_ids = data[4], data[5], data[6]

    new_train_id, new_test_id, new_valid_id = [], [], []
    # new_train_target, new_test_target, new_valid_target = [], [], []
    for train_id in train_ids:
        for conv_turn in range(len(sentence_dict[train_id])):
            new_train_id.append(f'{train_id}_{conv_turn}')
            
    for test_id in test_ids:
        for conv_turn in range(len(sentence_dict[test_id])):
            new_test_id.append(f'{test_id}_{conv_turn}')
    
    for valid_id in valid_ids:
        for conv_turn in range(len(sentence_dict[valid_id])):
            new_valid_id.append(f'{valid_id}_{conv_turn}')

    # train_data, test_valid_data = train_test_split(data_df, test_size=0.2, stratify=data_df['emotion'], random_state=42)
    # test_data, valid_data = train_test_split(test_valid_data, test_size=0.5, stratify=test_valid_data['emotion'], random_state=42)

    # dataset_list = ['train', 'test', 'valid']

parser = argparse.ArgumentParser(description='Data processing script')
parser.add_argument('--dataset', type=str, default='meld', help='Dataset name or path')
parser.add_argument('--historical_window', type=int, default=12, help='Historical window size')
parser.add_argument('--audio_description', type=str, default='False', help='Audio description task type')
parser.add_argument('--audio_impression', type=str, default='False', help='Audio impression task type')
parser.add_argument('--audio_only', type=str, default='False', help='Audio only task type')
parser.add_argument('--audio_context', type=str, default='False', help='Audio context task type')
parser.add_argument('--experiments_setting', type=str, default='lora', help='Experiments setting type')
args = parser.parse_args()


# Process data
processed_data_path = process_dataset(dataset=args.dataset, window=args.historical_window
        , audio_description=args.audio_description, audio_impression=args.audio_impression, 
        audio_only=args.audio_only, audio_context=args.audio_context, experiments_setting=args.experiments_setting)

print(processed_data_path)