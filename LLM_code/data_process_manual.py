import argparse
import pickle
import json
import os 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scalers = {}

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in data.iterrows():
            f.write(json.dumps({
                'path': row['audio_path'],
                'input': row['text'],
                'target': row['emotion']
            }, ensure_ascii=False) + '\n')

def scale_group(group):
    cols = ['avg_pitch', 'pitch_std', 'avg_intensity', 'mean_hnr', 'intensity_variation', 'pitch_range']

    # Ensure all three columns exist and have valid numeric values
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
    """
    Inputs:
      • evaluation/valid  = first 20 rows
      • test              = next 30 rows
      • train             = next 300 rows

    Returns
    -------
    train_df, test_df, valid_df : pd.DataFrame
    """

    sort_cols = ["speaker_id", "emotion"]
    if "file_id" in df.columns:
        sort_cols.append("file_id")
    elif "file_path" in df.columns:
        sort_cols.append("file_path")

    ordered = df.sort_values(sort_cols, ascending=True).reset_index(drop=True)

    valid_parts, test_parts, train_parts = [], [], []

    for _, group in ordered.groupby(["speaker_id", "emotion"], sort=False):
        group = group.reset_index(drop=True)

        valid_parts.append(group.iloc[:20])
        test_parts.append(group.iloc[20:50])
        train_parts.append(group.iloc[50:350])

    # Concatenate into DataFrames
    valid_df = pd.concat(valid_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)
    train_df = pd.concat(train_parts, ignore_index=True)

    return train_df, test_df, valid_df

def generate_prose_prompt(pitch, pstd, ivar, hnr, intensity, emotion_set):
    FEW_SHOT_EXAMPLES = (
        # EMODB
        # "Example 1:\n"
        # "pitch=190.0, pitch_std=65.0, intensity_variation=13.0, mean_hnr=14.5, avg_intensity=72.0\n"
        # "Prediction: This utterance is bright, smooth, and expressive — typical of HAPPINESS.\n\n"
        # "Example 2:\n"
        # "pitch=110.0, pitch_std=35.0, intensity_variation=9.0, mean_hnr=17.2, avg_intensity=60.0\n"
        # "Prediction: This utterance is flat and low-pitched — typical of SADNESS.\n\n"
        # "Example 3:\n"
        # "pitch=165.0, pitch_std=50.0, intensity_variation=14.2, mean_hnr=11.0, avg_intensity=75.0\n"
        # "Prediction: This utterance is harsh and intense — typical of ANGER.\n\n"

        # ESD
        "Example 1:\n"
        "pitch=260.0, pitch_std=60.0, intensity_variation=21.0, mean_hnr=10.5, avg_intensity=69.0\n"
        "Prediction: This utterance is sharp, volatile, and forceful — typical of ANGER..\n\n"
        "Example 2:\n"
        "pitch=270.0, pitch_std=65.0, intensity_variation=21.5, mean_hnr=13.2, avg_intensity=68.0\n"
        "Prediction: This utterance is bright, lively, and resonant — typical of HAPPY.\n\n"
        "Example 3:\n"
        "pitch=175.0, pitch_std=38.0, intensity_variation=15.0, mean_hnr=14.1, avg_intensity=58.0\n"
        "Prediction: This utterance is deep, flat, and soft — typical of SAD.\n\n"
    )
    
    prose_prompt = (
        "You are an expert in vocal emotion analysis. "
        "Below are examples describing typical emotional expressions based on acoustic features.\n\n"
        f"{FEW_SHOT_EXAMPLES}"
        "Now analyze this:\n"
        f"pitch={pitch:.2f}, pitch_std={pstd:.2f}, intensity_variation={ivar:.2f}, mean_hnr={hnr:.2f}, avg_intensity={intensity:.2f}\n"
        "Please respond in this format:\n"
        "Prediction: This utterance is ... — typical of <EMOTION>.\n"
        "Prediction: This utterance is "
    )

    return prose_prompt

def process_dataset(dataset, window=110, audio_description='True', audio_impression='False', audio_only='False', audio_context='False',experiments_setting='lora'):
    '''
    dataset: parameter that define the evaluated dataset.
    window:       parameter that control the historical context window
    speaker_mode: parameter that control the speaker identification task whether to be added in the main task

    data_path:    parameter that record the processed dataset's filepath
    '''
    label_set = {
        'iemocap':['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
        'meld':   ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
        'emodb': ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'sadness', 'neutral'],
        'esd': ['anger', 'happy', 'sad', 'neutral', 'surprise'],
    }
    label_text_set = {
        'iemocap':'happy, sad, neutral, angry, excited, frustrated',
        'meld'   :'neutral, surprise, fear, sad, joyful, disgust, angry',
        'emodb': 'anger, boredom, disgust, fear, happiness, sadness, neutral',
        'esd': 'anger, surprise, happy, sad, neutral',
    }

    ## load audio feature files
    if dataset == 'meld':
        audio_feature_train = pd.read_csv('../speech_features/meld_processed_5class_train_audio_features.csv')
        audio_feature_test = pd.read_csv('../speech_features/meld_processed_5class_test_audio_features.csv')
        audio_feature_dev = pd.read_csv('../speech_features/meld_processed_5class_dev_audio_features.csv')
    elif dataset == 'iemocap':
        audio_feature = pd.read_csv('../speech_features/processed_iemocap_audio_features_5.csv')
        iemocap_file = pd.read_csv('../data/IEMOCAP_full_release/iemocap_full_dataset.csv')
    elif dataset == 'emodb':
        audio_feature = pd.read_csv('../speech_features/processed_emodb_audio_features.csv')
        # audio_feature = audio_feature[audio_feature['emotion'].isin(['anger', 'neutral'])]
    elif dataset == 'esd':
        audio_feature = pd.read_csv('../speech_features/processed_esd_mandarin_features.csv')
    ###

    # emotional_dict = {text_label:num_label for num_label, text_label in enumerate(label_set[dataset])}
    emotional_dict = {'happy': 0, 'neutral': 1}  # <- REDUCED LABEL SET
    content_target_dict = {}
    content_task_dict = {}
    speaker_label_dict = {}
    sentence_dict = {}
    audio_path_dict = {}
    length = []

    if dataset == 'iemocap':
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

    elif dataset == 'meld':
        data = pickle.load(open(f'../original_data/{dataset}/{dataset}.pkl','rb'))
        all_conv_id = data[4] + data[5] + data[6]
        sentence_dict = data[3]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                temp_speaker_list.append(speaker_label.index(1))
            speaker_label_dict[conv_id] = temp_speaker_list

    elif dataset == 'emodb':
        audio_feature.dropna(subset=['avg_pitch', 'pitch_std', 'avg_intensity', 'mean_hnr', 'intensity_variation', 'pitch_range'], inplace=True)
        scaler = StandardScaler()
        # Sample fix to update normalization to speaker-specific
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

            speaker_id = row['speaker_id']
            prose_rule = ""

            scale = scalers[speaker_id]['scale']
            mean = scalers[speaker_id]['mean']

            # --- IEMOCAP + EMO-DB thresholds ---
            # anger Q3
            anger_hnr_q3 = 11.16
            anger_ivar_q3 = 12.97
            anger_pitch_q3 = 284.29
            anger_pstd_q3  = 81.84
            anger_intensity_q3 = 73.35

            # sadness Q1
            sadness_hnr_q1 = 5.54
            sadness_ivar_q1 = 5.74
            sadness_pitch_q1 = 124.23
            sadness_pstd_q1 = 25.80
            sadness_intensity_q1 = 46.15

            # happiness Q3
            happy_hnr_q3 = 11.33
            happy_ivar_q3 = 10.29
            happy_pitch_q3 = 243.28
            happy_pstd_q3 = 85.14
            happy_intensity_q3 = 65.91

            # compute scaled features
            hnr = row['mean_hnr'] * scale[3] + mean[3]
            ivar = row['intensity_variation'] * scale[4] + mean[4]
            pitch = row['avg_pitch'] * scale[0] + mean[0]
            pstd = row['pitch_std'] * scale[1] + mean[1]
            intensity = row['avg_intensity'] * scale[2] + mean[2]

            # classify
            if (hnr >= anger_hnr_q3
                and ivar >= anger_ivar_q3
                and pitch >= anger_pitch_q3
                and pstd >= anger_pstd_q3
                and intensity >= anger_intensity_q3):
                # emotion = 'anger'
                prose_rule = (
                    f"This sample is likely ANGER because mean HNR is high (≥{anger_hnr_q3:.2f} dB), "
                    f"intensity variation is high (≥{anger_ivar_q3:.2f} dB), avg pitch is high (≥{anger_pitch_q3:.2f} Hz), "
                    f"pitch std is high (≥{anger_pstd_q3:.2f} Hz), and avg intensity is high (≥{anger_intensity_q3:.2f} dB)."
                )
            elif (hnr <= sadness_hnr_q1
                and ivar <= sadness_ivar_q1
                and pitch <= sadness_pitch_q1
                and pstd <= sadness_pstd_q1
                and intensity <= sadness_intensity_q1):
                # emotion = 'sadness'
                prose_rule = (
                    f"This sample is possibly SADNESS because mean HNR is low (≤{sadness_hnr_q1:.2f} dB), "
                    f"intensity variation is low (≤{sadness_ivar_q1:.2f} dB), avg pitch is low (≤{sadness_pitch_q1:.2f} Hz), "
                    f"pitch std is low (≤{sadness_pstd_q1:.2f} Hz), and avg intensity is low (≤{sadness_intensity_q1:.2f} dB)."
                )
            elif (hnr >= happy_hnr_q3
                and ivar >= happy_ivar_q3
                and pitch >= happy_pitch_q3
                and pstd >= happy_pstd_q3
                and intensity >= happy_intensity_q3):
                # emotion = 'happiness'
                prose_rule = (
                    f"This sample is indicative of HAPPINESS because mean HNR is high (≥{happy_hnr_q3:.2f} dB), "
                    f"intensity variation is medium–high (≥{happy_ivar_q3:.2f} dB), avg pitch is high (≥{happy_pitch_q3:.2f} Hz), "
                    f"pitch std is high (≥{happy_pstd_q3:.2f} Hz), and avg intensity is high (≥{happy_intensity_q3:.2f} dB)."
                )
            else:
                # emotion = 'neutral/mixed'
                prose_rule = "No strong match to anger, sadness, or happiness thresholds."


            # text = (
            #     "Now you are an expert in sentiment and emotional analysis using only audio features.\n"
            #     f"Audio file: {row['filename']}\n"
            #     f"Pitch: {row.get('avg_pitch', 'unknown')}, Variation: {row.get('pitch_std', 'unknown')}, "
            #     f"Intensity: {row.get('avg_intensity', 'unknown')}\n"
            #     f"Please classify the emotional label of this utterance from <{label_text_set[dataset]}>.\n"
            # )

            text = (
                f"Pitch: {row.get('avg_pitch', 'unknown')}, Variation: {row.get('pitch_std', 'unknown')}, "
                f"Intensity: {row.get('avg_intensity', 'unknown')}\n"
                f"{prose_rule}"
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
        label_text_set['esd'] = 'happy, neutral'
    
        content_target_dict = {}
        content_task_dict = {}
        audio_path_dict = {}
        
        # Process dataset
        data_list = []
        prose_count = 0
        
        for idx, row in audio_feature.iterrows():
            all_conv_id = row['filename'].split('.')[0]
            emotion_label = row['emotion'].lower()

            speaker_id = row['speaker_id']
            prose_rule = ""

            scale = scalers[speaker_id]['scale']
            mean = scalers[speaker_id]['mean']

            # --- ESD Mandarin thresholds ---
            # anger Q3
            anger_hnr_q3_esd = 12.70
            anger_ivar_q3_esd = 21.86
            anger_pitch_q3_esd = 274.15
            anger_pstd_q3_esd  = 69.09
            anger_intensity_q3_esd = 70.75

            # sadness Q1
            sad_hnr_q1_esd = 10.19
            sad_ivar_q1_esd = 14.53
            sad_pitch_q1_esd = 138.41
            sad_pstd_q1_esd = 24.85
            sad_intensity_q1_esd = 55.87

            # happiness Q3
            happy_hnr_q3_esd = 14.35
            happy_ivar_q3_esd = 21.76
            happy_pitch_q3_esd = 298.87
            happy_pstd_q3_esd = 72.76
            happy_intensity_q3_esd = 70.83

            # compute scaled features
            hnr = row['mean_hnr'] * scale[3] + mean[3]
            ivar = row['intensity_variation'] * scale[4] + mean[4]
            pitch = row['avg_pitch'] * scale[0] + mean[0]
            pstd = row['pitch_std'] * scale[1] + mean[1]
            intensity = row['avg_intensity'] * scale[2] + mean[2]

            # classify
            if (hnr >= anger_hnr_q3_esd
                and ivar >= anger_ivar_q3_esd
                and pitch >= anger_pitch_q3_esd
                and pstd >= anger_pstd_q3_esd
                and intensity >= anger_intensity_q3_esd):
                # emotion = 'anger'
                prose_rule = (
                    f"This utterance is typical of ANGER because mean HNR ≥{anger_hnr_q3_esd:.2f} dB, "
                    f"intensity var ≥{anger_ivar_q3_esd:.2f} dB, pitch ≥{anger_pitch_q3_esd:.2f} Hz, "
                    f"pstd ≥{anger_pstd_q3_esd:.2f} Hz, intensity ≥{anger_intensity_q3_esd:.2f} dB."
                )
            elif (hnr <= sad_hnr_q1_esd
                and ivar <= sad_ivar_q1_esd
                and pitch <= sad_pitch_q1_esd
                and pstd <= sad_pstd_q1_esd
                and intensity <= sad_intensity_q1_esd):
                # emotion = 'sadness'
                prose_rule = (
                    f"This sample is likely to be SADNESS because mean HNR ≤{sad_hnr_q1_esd:.2f} dB, "
                    f"intensity var ≤{sad_ivar_q1_esd:.2f} dB, pitch ≤{sad_pitch_q1_esd:.2f} Hz, "
                    f"pstd ≤{sad_pstd_q1_esd:.2f} Hz, intensity ≤{sad_intensity_q1_esd:.2f} dB."
                )
            elif (hnr >= happy_hnr_q3_esd
                and ivar >= happy_ivar_q3_esd
                and pitch >= happy_pitch_q3_esd
                and pstd >= happy_pstd_q3_esd
                and intensity >= happy_intensity_q3_esd):
                # emotion = 'happiness'
                prose_rule = (
                    f"This utterance may be HAPPINESS because mean HNR ≥{happy_hnr_q3_esd:.2f} dB, "
                    f"intensity var ≥{happy_ivar_q3_esd:.2f} dB, pitch ≥{happy_pitch_q3_esd:.2f} Hz, "
                    f"pstd ≥{happy_pstd_q3_esd:.2f} Hz, intensity ≥{happy_intensity_q3_esd:.2f} dB."
                )
            else:
                # emotion = 'neutral/mixed'
                prose_rule = "No clear emotion match under ESD thresholds."

            # text = (
            #     "Now you are an expert in sentiment and emotional analysis using only audio features.\n"
            #     f"Audio file: {row['filename']}\n"
            #     f"Pitch: {row.get('avg_pitch', 'unknown')}, Variation: {row.get('pitch_std', 'unknown')}, "
            #     f"Intensity: {row.get('avg_intensity', 'unknown')}\n"
            #     f"Please classify the emotional label of this utterance from <{label_text_set[dataset]}>.\n"
            # )
            text = (
                "Now you are an expert in sentiment and emotional analysis using only audio features.\n"
                f"Pitch: {row.get('avg_pitch', 'unknown')}, Variation: {row.get('pitch_std', 'unknown')}, "
                f"Intensity: {row.get('avg_intensity', 'unknown')}\n"
                f"{prose_rule}"
                f"Please classify the emotional label of this utterance from <{label_text_set[dataset]}>.\n"
            )
        
            data_list.append({
                'conv_id': row['filename'].split('.')[0],
                'text': text,
                'emotion': emotion_label,
                'audio_path': row['filename'],
                'speaker_id': speaker_id
            })

        # Convert to DataFrame
        data_df = pd.DataFrame(data_list)

        # Stratified split
        train_data, test_data, valid_data = split_esd_mandarin(data_df)

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
    
    for conv_id in all_conv_id:
        if conv_id==125:
            continue
        temp_conv_turn = 0
        for conv_turn in range(len(sentence_dict[conv_id])):
            temp_content_str = 'Now you are expert of sentiment and emotional analysis.'
            temp_content_str += 'The following conversation noted between \'### ###\' involves several speakers. '

            if audio_context == 'True':
                temp_content_str += 'The last three utterances are followed by its speech features. ### '
            else:
                temp_content_str += ' ### '

            index_w = max(conv_turn-window, 0)
            for i, (speaker_label, sub_sent) in enumerate(zip(speaker_label_dict[conv_id][index_w:conv_turn+1], sentence_dict[conv_id][index_w:conv_turn+1])):
                sub_sent = sub_sent.replace('', "'")
                sub_sent = sub_sent.replace('', "")
                sub_sent = sub_sent.replace('', "")
                sub_sent = sub_sent.replace('', "")
                sub_sent = sub_sent.replace('', " ")
                sub_sent = sub_sent.replace('', " ")
                temp_content_str += (f'\t Speaker_{speaker_label}:"{sub_sent}"')
                if audio_context == 'True':
                    if index_w+i<conv_turn-3:
                        continue
                    if dataset == 'iemocap':
                        filtered_rows = audio_feature.loc[(audio_feature['video_id'] == conv_id) & (audio_feature['text'] == sub_sent)]
                        avg_intensity = filtered_rows['avg_intensity_category'].values[0]
                        intensity_variation = filtered_rows['intensity_variation_category'].values[0]
                        avg_pitch = filtered_rows['avg_pitch_category'].values[0]
                        pitch_variation = filtered_rows['pitch_std_category'].values[0]
                        description = filtered_rows['description'].values[0].split(': ')[1]
                        # temp_content_str += f' ({avg_intensity} volume with {intensity_variation} variation)'
                        # temp_content_str += f' ({description})'
                        temp_content_str += f' ({avg_pitch} pitch with {pitch_variation} variation)'
                    elif dataset == 'meld':
                        sub_sent = sub_sent.replace('   ', ' ').replace('  ', ' ').strip()
                        if conv_id >= 1153:
                            diag_id = conv_id - 1153
                            audio_feature_file = audio_feature_test
                        elif conv_id >= 1039:
                            diag_id = conv_id - 1039
                            audio_feature_file = audio_feature_dev
                        else:
                            diag_id = conv_id
                            audio_feature_file = audio_feature_train
                        audio_feature_file['Utterance'] = audio_feature_file['Utterance'].apply(lambda x: str(x).replace('   ', ' ').replace('  ', ' ').strip())
                        try:
                            filtered_rows = audio_feature_file.loc[(audio_feature_file['Utterance'] == sub_sent)]
                            avg_intensity = filtered_rows['avg_intensity_category'].values[0]
                        except:
                            break
                        avg_intensity = filtered_rows['avg_intensity_category'].values[0]
                        intensity_variation = filtered_rows['intensity_variation_category'].values[0]
                        avg_pitch = filtered_rows['avg_pitch_category'].values[0]
                        pitch_variation = filtered_rows['pitch_std_category'].values[0]
                        description = filtered_rows['description'].values[0].split(': ')[1]
                        #temp_content_str += f' ({avg_intensity} volume with {intensity_variation} variation)'
                        #temp_content_str += f' ({description})'
                        temp_content_str += f' ({avg_pitch} pitch with {pitch_variation} variation)'
            content_target_dict[f'{conv_id}_{conv_turn}'] = label_set[dataset][data[1][conv_id][conv_turn]]
            target_utterance = temp_content_str.split('\t')[-1].split(' (')[0]
            temp_content_str += ' ###'
            
            if audio_only == 'True':
                temp_content_str = 'Now you are expert of sentiment and emotional analysis using only audio features.'

            # add audio feature and impression
            # if audio_description == 'True' or audio_impression == 'True':
            if dataset == 'meld':
                mode = ''
                if conv_id >= 1153:
                    diag_id = conv_id - 1153
                    audio_feature_file = audio_feature_test
                    mode = 'test'
                elif conv_id >= 1039:
                    diag_id = conv_id - 1039
                    audio_feature_file = audio_feature_dev
                    mode = 'dev'
                else:
                    diag_id = conv_id
                    audio_feature_file = audio_feature_train
                    mode = 'train'
                
                audio_directory = f'../data/MELD/{mode}/{mode}_audio'
                
                while temp_conv_turn<100:
                    filename = f'dia{diag_id}_utt{temp_conv_turn}.wav'
                    filtered_rows = audio_feature_file.loc[audio_feature_file['filename'] == filename]
                    if not filtered_rows.empty:
                        # get audio features
                        description = filtered_rows['description'].values[0]
                        impression = filtered_rows['impression'].values[0]
                        duration = filtered_rows['duration'].values[0]
                        # find audio path
                        path = audio_directory + "/" + filename
                        audio_path_dict[f'{conv_id}_{conv_turn}'] = path
                        # if conv_id==1149:
                        #     print(temp_conv_turn,path, conv_turn, target_utterance)
                        temp_conv_turn += 1
                        break
                    else:
                        temp_conv_turn += 1
                
                if duration > 0.5:
                    if audio_description == 'True':
                        temp_content_str += f' {description}'
                    if audio_impression == 'True':
                        temp_content_str += f' {impression}'

            elif dataset == 'iemocap':
                target_sentence = target_utterance.split(':')[1][1:-1]
                filtered_rows = audio_feature.loc[(audio_feature['video_id'] == conv_id) & (audio_feature['text'] == target_sentence)]
                description = filtered_rows['description'].values[0]
                impression = filtered_rows['impression'].values[0]
                audio_path = iemocap_file.loc[(audio_feature['video_id'] == conv_id) & (audio_feature['text'] == target_sentence)]['path'].values[0]
                audio_directory = '../data/IEMOCAP_full_release'
                audio_path_dict[f'{conv_id}_{conv_turn}'] = audio_directory + '/' + audio_path
                if audio_description == 'True':
                    temp_content_str += f' {description}'
                if audio_impression == 'True':
                    temp_content_str += f' {impression}'

            ##-------------------------------------------------------
            if experiments_setting != 'zero_shot':
                if audio_only == 'True':
                    temp_content_str += f' Please select the emotional label from <{label_text_set[dataset]}> based on the audio features. Respond with just one label:'
                elif audio_description == 'True' or audio_impression == 'True':
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> from <{label_text_set[dataset]}> based on both the context and audio features. Respond with just one label:'
                else:
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> from <{label_text_set[dataset]}> based on the context. Respond with just one label:'
            else:
                if audio_only == 'True':
                    temp_content_str += f' Please select the emotional label based on the audio features. Please ONLY output only one label from <{label_text_set[dataset]}> without any other words: '
                elif audio_description == 'True' or audio_impression == 'True':
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> based on both the context and audio features. Please output ONLY ONE label from <{label_text_set[dataset]}> as the first word without any other words: '
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> based on both the context and audio features. Please output ONLY ONE label from <{label_text_set[dataset]}> as the first word without any other words: '
                else:
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> based on the context. Please output ONLY ONE label from <{label_text_set[dataset]}> as the first word without any other words: '
                    temp_content_str += f' Please select the emotional label of <{target_utterance}> based on the context. Please output ONLY ONE label from <{label_text_set[dataset]}> as the first word without any other words:'
            length.append(len(temp_content_str))
            
            content_task_dict[f'{conv_id}_{conv_turn}'] = temp_content_str


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
    if dataset == 'iemocap' or 'meld':
        data_path = f'../PROCESSED_DATASET/{dataset}/window/{audio_description}_{audio_impression}'
    os.makedirs(data_path, exist_ok=True)

    with open(f'{data_path}/train.json', 'w', encoding='utf-8') as f_train:
        for train_id in new_train_id:
            if train_id.split('_')[0]=='125':
                continue
            f_train.write(json.dumps({'path': audio_path_dict[train_id], \
            'input':f'{content_task_dict[train_id]}','target':f'{content_target_dict[train_id]}'}, ensure_ascii=False)+ '\n')

    with open(f'{data_path}/test.json', 'w', encoding='utf-8') as f_test:
        for test_id in new_test_id:
            f_test.write(json.dumps({'path': audio_path_dict[test_id],\
            'input':f'{content_task_dict[test_id]}','target':f'{content_target_dict[test_id]}'}, ensure_ascii=False)+ '\n')

    with open(f'{data_path}/valid.json', 'w', encoding='utf-8') as f_valid:
        for valid_id in new_valid_id:
            if valid_id.split('_')[0]=='1149':
                continue
            f_valid.write(json.dumps({'path': audio_path_dict[valid_id], \
            'input':f'{content_task_dict[valid_id]}','target':f'{content_target_dict[valid_id]}'}, ensure_ascii=False)+ '\n')
        
    # draw histogram of the length of the input text
    plt.hist(length, bins=20)
    plt.xlabel('Length of the input text')
    plt.ylabel('Frequency')
    plt.title('Histogram of the length of the input text')
    plt.savefig(f'{data_path}/histogram.png')

    return data_path

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

# # Adjust this path to your actual test split path
# json_path = "../PROCESSED_DATASET/emodb/cleaned/no_leakage/test.json"

# # Load the JSON lines
# df = pd.read_json(json_path, lines=True)

# # Count how many inputs include prose rules (look for "This utterance")
# prose_examples = df[df['input'].str.contains("This utterance")]
# print(f"Total inputs: {len(df)}")
# print(f"Inputs with prose rules: {len(prose_examples)}")
# print(f"Percentage with prose: {100 * len(prose_examples)/len(df):.2f}%")

# # Optionally view some examples
# print(prose_examples.sample(3)['input'].values)