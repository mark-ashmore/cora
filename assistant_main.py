import json
import logging
import os
import pickle
import re
import sys
from itertools import pairwise
from textwrap import TextWrapper
from typing import Any

import google.generativeai as genai
import pandas as pd
import spacy
import speech_recognition as sr
import torch
from google.api_core.exceptions import InternalServerError
from google.cloud import texttospeech
from google.generativeai.types.generation_types import StopCandidateException
from Levenshtein import distance
from playsound import playsound
from transformers import BertTokenizer, BertForSequenceClassification

from agent_actions.light_control import HUE, Lights
from utils import paths

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
f_handler = logging.FileHandler("assistant_main.log")
f_format = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s | %(funcName)s | "
    "%(lineno)s | %(message)s"
)
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)
logger.setLevel(level=logging.DEBUG)


class AgentResponse:
    """A class using google text to speech to generate natural speech."""

    def __init__(self) -> None:
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
            name="en-US-Journey-O",
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

    def say(self, response_message: str) -> None:
        """Generate speech response."""
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=response_message)
        response = client.synthesize_speech(
            input=synthesis_input, voice=self.voice, audio_config=self.audio_config
        )
        with paths.AGENT_RESPONSE_PATH.open(mode="wb") as out:
            out.write(response.audio_content)
        print(TextWrapper(width=80).fill(response_message))
        playsound(sound=paths.AGENT_RESPONSE_PATH)

    def save_message(self, response_message: str, file_name: str) -> None:
        """Generate speech response and save to audio folder."""
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=response_message)
        response = client.synthesize_speech(
            input=synthesis_input, voice=self.voice, audio_config=self.audio_config
        )
        audio_file = paths.AUDIO_PATH / f"{file_name}.mp3"
        with audio_file.open(mode="wb") as out:
            out.write(response.audio_content)
        logger.info("Message written to file.")

    def play_message(self, file_name: str) -> None:
        """Play audio file from audio folder."""
        playsound(sound=(paths.AUDIO_PATH / f"{file_name}.mp3"))


class DeepThought:
    """Deep Thought is an agent class that will listen to and speak with a user.
    Deep Thought can also understand and respond to specific command requests.
    """

    def __init__(
        self,
        confidence_threshold: float,
        device_index: int,
        energy_threshold: int,
        longer_wake_word: str,
        phrase_time_limit: float,
        prompt: str,
        short_wake_words: list[str],
        similarity_threshold: int,
        timeout: float,
        wake_word_time_limit: float,
        wake_word_timeout: float,
        default_settings: bool,
    ) -> None:
        """Initialize Deep Thought."""
        self._bert_path = paths.BERT_PATH
        self._entity_model_path = paths.ENTITY_MODEL_PATH
        self._tokenizer_path = paths.TOKENIZER_PATH
        self._labels_path = paths.LABELS_PATH
        self.active_listening = False
        self.agent_response = AgentResponse()
        self.confidence_threshold = confidence_threshold
        self.default_settings = default_settings
        self.device_index = device_index
        self.longer_wake_word = longer_wake_word
        self.phrase_time_limit = phrase_time_limit
        self.prompt = prompt
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.short_wake_words = short_wake_words
        self.sigmoid = torch.nn.Sigmoid()
        self.similarity_threshold = similarity_threshold
        self.start_up = True
        self.timeout = timeout
        self.wake_word_time_limit = wake_word_time_limit
        self.wake_word_timeout = wake_word_timeout
        self._load_chat_model()
        self._load_predictor_model()
        self._load_entity_model()
        if self.default_settings:
            self._check_for_mics()

    def _load_chat_model(self) -> None:
        """Load chat model."""
        logger.debug("Loading chat model.")
        with (paths.PROMPTS_PATH / f"{self.prompt}.md").open(
            "r", encoding="utf-8"
        ) as f:
            self._prompt_string = f.read()
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.genai_model = genai.GenerativeModel(
            model_name="gemini-1.5-pro", system_instruction=self._prompt_string
        )
        chat_history = []
        if paths.CHAT_HISTORY.exists():
            logger.debug("Loading history from file.")
            chat_history = self._read_history()
        self.chat = self.genai_model.start_chat(history=chat_history)

    def _load_predictor_model(self) -> None:
        """Load model."""
        logger.debug("Loading classifier model.")
        self.model: BertForSequenceClassification | Any = (
            BertForSequenceClassification.from_pretrained(self._bert_path)
        )
        self.tokenizer = BertTokenizer.from_pretrained(self._tokenizer_path)
        self.id2label = pd.read_csv(self._labels_path).to_dict()["label"]

    def _load_entity_model(self) -> None:
        """Load entity phrase matcher model."""
        logger.debug("Loading phrase matcher model and nlp.")
        self.nlp = spacy.load((self._entity_model_path / "nlp"))
        self.phrase_matcher = pickle.load(
            (self._entity_model_path / "matcher.pkl").open("rb")
        )

    def _check_for_mics(self) -> None:
        """Check for available microphones."""
        mics = list(enumerate(sr.Microphone.list_microphone_names()))
        if not mics:
            sys.exit("No mics detected. Please connect a microphone.")
        mics_string = ""
        for index, name in mics:
            mics_string += f"{index} - {name}\n"
        self.device_index = int(
            input(
                "Starting up.\n\nPlease let me know which microphone I can use:\n\n"
                f"{mics_string}\n"
            )
        )
        with sr.Microphone(device_index=self.device_index) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def _write_history(self) -> None:
        """Write history to file."""
        with paths.CHAT_HISTORY.open("wb+") as f:
            pickle.dump(self.chat.history, f)
        logger.info("History written to file.")

    def _read_history(self) -> list:
        """Read history from file."""
        with paths.CHAT_HISTORY.open("rb") as f:
            chat_history = pickle.load(f)
        logger.info("History read from file.")
        return chat_history

    def _write_custom_settings(self) -> None:
        """Write custom settings to file."""
        custom_settings = {
            "confidence_threshold": self.confidence_threshold,
            "device_index": self.device_index,
            "energy_threshold": self.recognizer.energy_threshold,
            "longer_wake_word": self.longer_wake_word,
            "phrase_time_limit": self.phrase_time_limit,
            "prompt": self.prompt,
            "short_wake_words": self.short_wake_words,
            "similarity_threshold": self.similarity_threshold,
            "timeout": self.timeout,
            "wake_word_timeout": self.wake_word_timeout,
            "wake_word_time_limit": self.wake_word_time_limit,
            "default_settings": False,
        }
        with paths.CUSTOM_SETTINGS_PATH.open("w+", encoding="utf-8") as f:
            json.dump(custom_settings, f, indent=4)

    def _wakeword_detect(self, words: list[str] | Any) -> bool:
        """Detect wake word."""
        if not words:
            return False
        if len(words) == 1:
            logger.debug("One word detected. %s", str(words))
            min_distance = min(
                [
                    distance(words[0].lower().replace(".", ""), short_ww.lower())
                    for short_ww in self.short_wake_words
                ]
            )
            return min_distance < self.similarity_threshold
        logger.debug("Several words detected. %s", str(words))
        min_distance = min(
            [
                distance(
                    f"{word_pair[0]} {word_pair[1]}".lower(),
                    self.longer_wake_word.lower(),
                )
                for word_pair in pairwise(words)
            ]
        )
        return min_distance < self.similarity_threshold

    def predict_intent(self, query: str) -> str:
        """Predict intent.

        If confidence is below threshold, return 'no_match'.

        Returns:
            str: intent
        """
        logger.debug("Making a prediction")
        encoding = self.tokenizer(query, return_tensors="pt")
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
        probs = self.sigmoid(outputs.logits.squeeze().cpu())
        predictions = {label: self.id2label[x] for x, label in enumerate(probs)}
        logger.debug("Predictions: %s", str(predictions))
        confidence = max(predictions.keys())
        prediction = predictions[confidence]
        logger.debug(f"Confidence: {confidence}")
        logger.debug(f"Prediction: {prediction}")
        if confidence < self.confidence_threshold:
            prediction = "no_match"
        return prediction

    def get_entity_values(self, query: str, prediction: str) -> list:
        """Get entity values for the predicted intent.

        Returns:
            list: entity values
        """
        doc = self.nlp(query)
        matches = self.phrase_matcher(doc)
        match_list = []
        for match_id, start, end in matches:
            span = doc[start:end]
            match_list.append((span.text, self.nlp.vocab.strings[match_id], start, end))
        return_matches = []
        if prediction:
            intent_path = paths.MODEL_TRAINING_PATH / f"{prediction}.json"
            with intent_path.open("r", encoding="utf-8") as f:
                intent_dict = json.load(f)
            intent_entities = intent_dict["entities"]
            if match_list:
                return_matches = [
                    match for match in match_list if match[1].lower() in intent_entities
                ]
        return return_matches

    def perform_action(self, intent: str, entities: list, user_input: str) -> None:
        """Perform user action."""
        logger.debug(f"Intent: {intent}")
        logger.debug(f"Entities: {entities}")
        logger.debug(f"User input: {user_input}")
        if intent == "turn_on_lights":
            lights = Lights(HUE)
            try:
                lights.turn_on_light("Living room 1")
                lights.turn_on_light("Living room 2")
                self.agent_response.say("Turning on the lights.")
            except ConnectionError:
                self.agent_response.play_message(file_name="hue_connection_error")
        elif intent == "turn_off_lights":
            lights = Lights(HUE)
            try:
                lights.turn_off_light("Living room 1")
                lights.turn_off_light("Living room 2")
                self.agent_response.say("Turning off the lights.")
            except ConnectionError:
                self.agent_response.play_message(file_name="hue_connection_error")
        else:
            self.agent_response.say(
                f'I heard you say "{user_input}" and '
                f"identified the intent {intent}. "
                "Unfortunately I can't help with that yet"
            )

    def chat_with_user(self, user_text: str) -> None:
        """Chat with user."""
        logger.info(f'I heard you say "{user_text}"\n')
        logger.info("\nWaiting for agent...\n")
        try:
            response = self.chat.send_message(user_text)
            response_text = re.sub(r"\*.*?\*|\(.*?\)", "", response.text)
            response_text = (
                response_text.replace("\n\n", " ")
                .replace("\n", " ")
                .replace(":", " ")
                .replace("*", " ")
                .replace("  ", " ")
            )
        except InternalServerError:
            self.agent_response.play_message(file_name="gemini_server_error")
            return
        except StopCandidateException:
            self.agent_response.play_message(file_name="gemini_stop_candidate_error")
            return
        self.agent_response.say(response_text)

    def listen_for_input(self) -> str:
        """Listen for input."""
        logger.info("Listening...")
        text = ""
        with sr.Microphone(device_index=self.device_index) as source:
            try:
                audio = self.recognizer.listen(
                    source=source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )
                text = self.recognizer.recognize_whisper(
                    audio_data=audio, model="base.en", language="english"
                )
            except sr.UnknownValueError:
                logger.debug("Audio not understood.")
                return text
            except sr.RequestError:
                logger.debug("Something went wrong with Whisper")
                return text
            except sr.WaitTimeoutError:
                logger.debug("Waited too long")
                return text
        return text.strip()

    def chat_with_agent(self, user_input: str, greeting: bool = False) -> str:
        """Greet user with message and listen to response."""
        if user_input and user_input != ".":
            logger.info('Agent heard "%s"', user_input)
            prediction = self.predict_intent(user_input)
            logger.info("Intent: %s", prediction)
            if (prediction != "no_match") and not greeting:
                logger.info("Taking an action")
                entities = self.get_entity_values(user_input, prediction)
                self.perform_action(prediction, entities, user_input)
            else:
                logger.info("Having a chat with user")
                self.chat_with_user(user_input)
            return "active_user"
        else:
            return "no_input"

    def loop_conversation(self) -> str:
        """Loop conversation."""
        user_status = "user_inactive"
        for _ in range(5):
            user_status = self.chat_with_agent(self.listen_for_input())
            if user_status == "active_user":
                break
        if user_status == "no_input":
            user_status = "user_inactive"
        return user_status

    def run_conversation(self) -> None:
        """Run conversation."""
        user_status = self.chat_with_agent("Hello", greeting=True)
        while self.active_listening:
            while user_status == "active_user":
                user_status = self.chat_with_agent(self.listen_for_input())
            user_status = self.loop_conversation()
            if user_status == "user_inactive":
                self.active_listening = False

    def run_wakeword_listen_loop(self) -> None:
        """Run wakeword listen loop."""
        if self.start_up:
            playsound(paths.WAKE_SOUND)
            self.start_up = False
        while True:
            text = ""
            with sr.Microphone(device_index=self.device_index) as source:
                try:
                    audio = self.recognizer.listen(
                        source=source,
                        timeout=self.wake_word_timeout,
                        phrase_time_limit=self.wake_word_time_limit,
                    )
                    text = self.recognizer.recognize_whisper(
                        audio_data=audio, model="base.en", language="english"
                    )
                except sr.UnknownValueError:
                    logger.debug("Audio not understood.")
                except sr.RequestError:
                    logger.debug("Something went wrong with Whisper")
                except sr.WaitTimeoutError:
                    logger.debug("Waited too long")
            if self._wakeword_detect(text.split()):
                playsound(paths.GREETING_SOUND)
                self.active_listening = True
                self.run_conversation()


if __name__ == "__main__":
    if paths.CUSTOM_SETTINGS_PATH.exists():
        with paths.CUSTOM_SETTINGS_PATH.open("r", encoding="utf-8") as f:
            agent_settings = json.load(f)
    else:
        with paths.SETTINGS_PATH.open("r", encoding="utf-8") as f:
            agent_settings = json.load(f)
    deep_thought = DeepThought(**agent_settings)
    try:
        deep_thought.run_wakeword_listen_loop()
        # deep_thought._check_for_mics()
    except KeyboardInterrupt:
        deep_thought._write_history()
        deep_thought._write_custom_settings()
        logger.info("Closing DeepThought agent.")
