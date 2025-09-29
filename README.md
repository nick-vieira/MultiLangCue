# MultiLangCue

#### Note: 
This is a fork of [the original SpeechCueLLM repo](https://github.com/zehuiwu/SpeechCueLLM) that attempts to add multilingual support as part of my hpoythesis for my master's research project. 

## Extract Speech Features

#### Use Existing Features
You can also use the existing feature files in the speech_features folder.

#### Data Directory
To run our preprocessing codes directly, please download data and put into the data drirectory.

#### Run Extraction Code
1. extract basic speech features (adjust arguments based on the dataset)
    ```
    python feature_utils/extract_audio_feature.py
    ```
2. post-process the basic features
    ```
    python feature_utils/postprocess_audio_feature_{dataset}.py
    Dataset choices include (meld, iemocap, esd, emodb)
    ```
    
3. test the extracted features (adjust variables (dataset, classes) inside the main function)
    ```
    python model_audio_features.py
    ```

## LLM Modeling
Credit: The project was built on the foundation of [InstructERC](https://github.com/LIN-SHANG/InstructERC) and [SpeechCueLLM](https://github.com/zehuiwu/SpeechCueLLM).

#### Environment setup:
1. create a new environment using python 3.8.10
2. install dependencies
    ```
    pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
    cd env && pip install -r requirements.txt
    ```

#### Training
1. download LLMs from HuggingFace and store them in the LLM_bases folder (This project used "lightblue/suzume-3-8b-instruct".
2. update the model path and adjust training parameters in ```train_and_inference.sh```
3. start training
    ```
    cd LLM_code
    bash train_and_inference.sh
    ```
    The script will first run the ```data_process.py``` script which will process the pickle files from the ```original_data``` folder, or the postprocessed audio feature data for the non-English corpora, and create a new folder called ```PROCESSED_DATASET``` to store the inputs for LLMs. It will then run the training script which will create a new folder called ```experiemnts``` to store the training results.
    
    If you want to train projection-based models, you need to download the raw audio of the dataset and store them in the ```data``` folder.
