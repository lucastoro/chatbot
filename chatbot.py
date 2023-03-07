import sys
import os
from configparser import ConfigParser
from abc import ABC, abstractmethod
import typing
import asyncio
import boto3
import openai
import numpy
import sounddevice
from pathlib import Path
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream


# when true stops recording from the mic.
silence = False


def from_config(filename: str, label: str, section: str|None = None):
    config_path = os.path.sep.join([
        str(Path.home()),
        filename.replace('/', os.path.sep)
    ])

    parser = ConfigParser()
    parser.read(config_path)

    if section is None:
        if len(parser.sections()) == 1:
            section = parser.sections()[0]
        else:
            section = parser.default_section

    return parser[section][label]


def get_prop(env: str, file: str, key: str):
    return os.environ.get(env) or from_config(file, key)


class SpeechSynthesizer(ABC):

    """
    Generic interface for a speech synthetizer that takes a text as input
    and returns a buffer holding the waveform of the synthetized speech.
    """

    @abstractmethod
    def synthesize(self, text: str) -> typing.Tuple[numpy.ndarray, int]:
        """Returns the buffer and the sampling rate"""
        pass

    def speak(self, text: str, blocking: bool = False):
        """Convenience utility to playback a given text."""
        wave, rate = self.synthesize(text)
        sounddevice.play(wave, samplerate=rate, blocking=blocking)


class ChatPeer(ABC):

    """
    Generic interface for a chat responder,
    will return some text in response to a message.
    """

    @abstractmethod
    def chat(self, message: str) -> str:
        pass


class Polly(SpeechSynthesizer):

    """
    AWS/Polly-based implementation of a SpeechSynthesizer
    """

    def __init__(self, voice_id: str = 'Matthew', engine: str = 'neural'):
        self.voice_id = voice_id
        self.client = boto3.client('polly')
        self.engine = engine

    def synthesize(self, text: str) -> typing.Tuple[numpy.ndarray, int]:
        """Converts the given text to a PCM waveform"""

        speech = self.client.synthesize_speech(
            Engine=self.engine,
            VoiceId=self.voice_id,
            Text=text,
            LanguageCode='en-US',
            OutputFormat='pcm',
            TextType='text'
        )

        # recover the audio stream from Polly's response
        return numpy.frombuffer(
            speech['AudioStream'].read(),
            dtype=numpy.int16
        ), 16000


class ChatGPT(ChatPeer):

    """
    OpenAI-based chat peer.
    Keeps a conversation log to send back to the bot
    as context for more meaningful conversations.
    """

    def __init__(self,
                 api_key: str|None = None,
                 max_length: int = 10):
        self.conversation = []
        self.max_length = max_length
        self.api_key = api_key
        self.conversation = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }]

    def chat(self, message: str) -> str:

        # store your message in the conversation
        self.conversation.append({
            'role': 'user',
            'content': message
        })

        # remove older messages from the conv. eventually
        if len(self.conversation) > self.max_length:
            # remove the first 2 messages, mine and his
            self.conversation = self.conversation[2:]

        response = openai.ChatCompletion.create(
            api_key=self.api_key,
            model="gpt-3.5-turbo",
            messages=self.conversation,
        )

        # decode and cleanup the bot's response
        message = response['choices'][0]['message']
        content = message['content']
        role = message['role']


        # store his message in the conversation
        self.conversation.append({
            'role': role,
            'content': content
        })

        return content


class MyEventHandler(TranscriptResultStreamHandler):

    def __init__(self,
                 stream: TranscriptResultStream,
                 peer: ChatPeer,
                 synth: SpeechSynthesizer):
        super().__init__(stream)
        self.peer = peer
        self.synth = synth

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial and len(result.alternatives) > 0:

                global silence

                transcript = result.alternatives[0].transcript
                print(f"You said: {transcript}")

                response = self.peer.chat(transcript)
                print(f"Bot said: {response}")

                # stops listening on the mic while playing the
                # reponse back to avoid the feedback loop.
                silence = True
                self.synth.speak(response, blocking=True)
                silence = False


async def mic_stream():
    # This function wraps the raw input stream from the microphone forwarding
    # the blocks to an asyncio.Queue.
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        if not silence:
            loop.call_soon_threadsafe(
                input_queue.put_nowait,
                (bytes(indata), status)
            )

    # Be sure to use the correct parameters for the audio stream that matches
    # the audio formats described for the source language you'll be using:
    # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    stream = sounddevice.InputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        dtype=numpy.int16,
    )
    # Initiate the audio stream and asynchronously yield the audio chunks
    # as they become available.
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status


async def record_audio(stream):
    # This connects the raw audio chunks generator coming from the microphone
    # and passes them along to the transcription stream.
    async for chunk, _ in mic_stream():
        await stream.send_audio_event(audio_chunk=chunk)
    await stream.end_stream()


async def basic_transcribe():

    # TranscribeStreamingClient does not infer the region automatically
    region = get_prop('AWS_DEFAULT_REGION', '.aws/config', 'region')

    if region is None:
        print('Cannot determine AWS region')
        sys.exit(1)

    # OpenAI API-key is assumed to be in ~/.openai/key
    if 'OPENAI_API_KEY' not in os.environ:
        openai.api_key_path = os.path.join(
           str(Path.home()),
            '.openai',
            'key'
        )

    # Set up our client with our chosen AWS region
    client = TranscribeStreamingClient(region=region)

    # Start transcription to generate our async stream
    transcribe_stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )

    peer = ChatGPT()
    synt = Polly(voice_id='Matthew')

    # Instantiate our handler and start processing events
    handler = MyEventHandler(transcribe_stream.output_stream, peer, synt)

    print("** speak now **")

    await asyncio.gather(
        record_audio(transcribe_stream.input_stream),
        handler.handle_events()
    )

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(basic_transcribe())
    loop.close()
