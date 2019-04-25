# bottlneck_cnn_text_classification
This is a deep CNN architecture with bottleneck CNN for binary text classification

## Getting Started
To get the summarizer to work one first has to install requiered libraries and run some preprocessings

### Prerequisites

Besides Python 3.x and tensorflow 1.3 following libraries are additionally requiered,
run the following commands in your shell

```
pip install numpy
pip install tqdm
```
The default for all following instructions is to have the project folder as cd.

First run the following commands
```
mkdir ./data
mkdir ./ckpt
mkdir ./log

## Preprocess Data
```
To test the architecture for hatespeech on tweets, put hate-speech-and-offensive-language/data/labeled_data.csv from 
```
https://github.com/t-davidson/hate-speech-and-offensive-language
```
into your ./data folder.

Then run the following code via

```
cd src/
python3 preprocess.py
```

## Running the classification

To run the classification you can adapt the hyperparameters in hyperparameters.py 
Then execute
```
cd src/
python3 summarize.py
```
