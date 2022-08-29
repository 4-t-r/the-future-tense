# the-future-tense

Welcome to our github page!! We are four Students from the Leipzig University.

Within the scope of the seminar "Big Data and Language Technologies 2022" we decided to examine perceptions towards the future of AI. Therefore we analyzed statements about the future corresponding to several AI related topics.
To realize this we utlized the web Archive of the Webis Group (https://webis.de/), from which we extracted AI statements using the WARC-DL pipeline (https://github.com/webis-de/WARC-DL).

In the following we describe our approach and explain how our code can be executed.

## Workflow

The following chart shows the workflow of our project. First AI statements are extracted from the WARC archive. Afterwards the model pipeline is excuted. Since the Topic model contains dummy topics only, the output of the Model Pipeline is utilized for the topic selection.
After this step the selected topics are given to the topic assignment model and the Model Pipeline is ready to use. Subsequently we utilize the output of further executions for the analyzis and visualization.

![overview-chart][chart-relative]

### Stage_1 WARC-DL Extraction

### Stage_2_1 Models

All scripts at this step serve as the preparation for the model pipeline.

#### Future Model Training

1. Navigate to the directory: `the-future-tense/stage_2_1_models/future_model/dataset`

2. Extract dataset to train the future model: `./extract.py`

3. Navigate to the model directory: `the-future-tense/stage_2_1_models/future_model/training/future_model_ft`

4. Run the jupyter notebook script: `future_model_ft.ipynp`

#### Sentiment Model Evaluation

1. Navigate to the directory: `the-future-tense/stage_2_1_models/sentiment_model`

2. Run the sentiment model test: `./test_sentiment_model.py`

#### Topic Selection

1. Navigate to the following directory: `the-future-tense/stage_2_1_models/topic_model`

2. Run the following jupyter notebook : `topic_eval.ipynb`

### Stage_2_2 Model Pipeline

The Model Pipeline can now be executed in order to create the final dataset.

1. Navigate to the Model Pipeline directory: `the-future-tense/stage_2_2_model_pipeline`

2. Execute the Model Pipeline: `sbatch run_main.job`

### Stage_3 Visualization

The visualization for the analysis is generated at this stage.

1. Navigate to the visualization directory: `the-future-tense/stage_3_visualization`

2. Deposit your [Openai API-Key](https://beta.openai.com/account/api-keys) in your .env as OPENAI_API_KEY

3. Execute the jupyter notebook `visualize.ipynb`

[chart-relative]: images/project_overview.png "overview-chart"
