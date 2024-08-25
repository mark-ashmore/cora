# CORA

CORA (Collaborative Organizational Responsive Assistant) is a personal assistant
project, where I am experimenting with the basic concepts of NLP, NLU, and NLG.

## Installation notes

Install the requirements.txt packages using:

```sh
pip install -r requirements.txt
```

### Debian Installation

If you are on a Debian/Raspbian/Ubuntu you will need to run:

```sh
sudo apt-get install libffi-dev portaudio19-dev python3-pyaudio
```

You should also run:

```sh
sudo apt install libgirepository1.0-dev libcairo2-dev
```

You should also have
[GStreamer](https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c)
installed.

### Mac Installation

Macs will need pyobject. Run the following:

```sh
pip install pyobjc
```

Linux will need to run:

```sh
pip install pygobject
```

### General Installation

If you run into an error for setuptools you will need to first run:

```sh
pip install -U setuptools
```

Then try to install from requirements again.

If you are running a newer version of python you may get an install error for
the `playsound` package. Try running the following instead:

```sh
pip install git+https://github.com/killjoy1221/playsound.git
```

This will install the package directly from github.

### Spacy Installation

You will also need a spacy model for entity recognition. I'm using the en_core_web_sm
model by default since it offers what we need.

Run the following to dowload that model:

```sh
python -m spacy download en_core_web_sm
```

## Environment Variables

You will need to set a few environment variables with the appropriate values for this
project. You will need the following:

- `GOOGLE_API_KEY`
  - This project uses `gcloud` and Gemini for the LLM responses. You will need an API
    key and to have gcloud installed.
- See the Hue section below if you plan on using this for light controls. Current HUE
  controls are supported.

## Hue

Cora supports light controls for Hue brand lights. You will need to set the
enviromentment variables `HUE_HOST_NAME` and `HUE_APPLICATION_KEY` to your Hue IP address
and application key. Check the Hue documention for how to find these values.

## Model training

You will need to train the entity and bert models. This can be done by running:

```sh
python assistant_main.py --mode=train
```

## Model Testing

You can also test what the classifier model will return by running:

```sh
python assistant_main.py --mode=predict
```

## Lauching CORA

To launch Cora as your personal assistant, run:

```sh
python assistant_main.py [--mode=on]
```
