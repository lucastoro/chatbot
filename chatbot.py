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
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream


def from_config(filename: str, label: str, section: str = None):
    config_path = os.path.sep.join([
        os.environ['HOME'],
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

    def speak(self, text: str):
        """Convenience utility to playback a given text"""
        wave, rate = self.synthesize(text)
        sounddevice.play(wave, samplerate=rate)


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


class OpenAiFriend(ChatPeer):

    """
    OpenAI-based chat peer.
    Keeps a conversation log to send back to the bot
    as context for more meaningful conversations.
    """

    def __init__(self,
                 api_key: str = None,
                 max_length: int = 10,
                 engine: str = 'davinci'):
        self.conversation = []
        self.max_length = max_length
        self.engine = engine
        self.api_key = api_key

    def _flatten_conversation(self) -> str:
        chat = [
            f"You: {self.conversation[i]}" if i % 2 == 0
            else f"Friend: {self.conversation[i]}"
            for i in range(len(self.conversation))
        ]
        chat.append('Friend:')
        return '\n'.join(chat)

    def chat(self, message: str) -> str:

        # store your message in the conversation
        self.conversation.append(message)

        # remove older messages from the conv. eventually
        if len(self.conversation) > self.max_length:
            # remove the first 2 messages, mine and his
            self.conversation = self.conversation[2:]

        # converts the conversation log to a 'chat-like' text
        prompt = self._flatten_conversation()

        # submit the chat transcript to OpenAI
        response = openai.Completion.create(
            api_key=self.api_key,
            engine=self.engine,
            prompt=prompt,
            temperature=0.4,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=["You:"]
        )

        # decode and cleanup the bot's response
        responseText = response['choices'][0]['text'].strip()

        # sometimes the bot returns multiple messages back
        responseText = responseText.replace("\nFriend:", ".")

        # store his message in the conversation
        self.conversation.append(responseText)

        return responseText


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

                transcript = result.alternatives[0].transcript
                print(f"You said: {transcript}")
                response = self.peer.chat(transcript)
                print(f"Bot said: {response}")
                self.synth.speak(response)
                break


async def mic_stream():
    # This function wraps the raw input stream from the microphone forwarding
    # the blocks to an asyncio.Queue.
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
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
    async for chunk, status in mic_stream():
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
            os.environ['HOME'],
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

    peer = OpenAiFriend(engine='davinci')
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
