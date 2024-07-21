# CORA

CORA (Collaborative Organizational Responsive Assistant) is a personal assistant
project, where I am experimenting with the basic concepts of NLP, NLU, and NLG.

## Installation notes

Install the requirements.txt packages.

If you are running a newer version of python you may get an install error for
the `playsound` package. Try running the following instead:

```
pip install git+https://github.com/killjoy1221/playsound.git
```

This will install the package directly from github.

You will also need a spacy model for entity recognition. I'm using the en_core_web_sm
model by default since it offers what we need.

Run the following to dowload that model:

```
python -m spacy download en_core_web_sm
```

## Model training

You will need to train the entity and bert models. This can be done by running:

```
python main_pipeline/main_pipeline.py
```

## Hue

Cora supports light controls for Hue brand lights. You will need to set the
enviromentment variables `HUE_IP_ADDRESS` and `HUE_APPLICATION_KEY` to your Hue
IP address and application key. Check the Hue documention for how to find these values.
