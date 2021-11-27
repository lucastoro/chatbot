# chatbot
A simple speech-to-speech chatbot based on cloud tech.

## How it works
- Real time speech recognition is performed through AWS/Transcribe.
- Transcriptions are transformed into a chat log that is sent to OpenAI's "Friend chat" bot.
- The bot's responses are passed to AWS/Polly for text-to-speech synthesis.
- Synthetized voice from AWS/Polly is played.
- Repeat. 