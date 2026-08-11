# How fast whisper actually is on this machine

RTX 4090 (24 GB, ~8.6 GB held by the VoxCPM2 TTS engine on :8883 throughout — every GPU
number below was measured with it loaded, because that is the real machine), 24-core CPU,
faster-whisper 1.2.1 / CTranslate2 4.6.3.

**Winner: `large-v3-turbo` on CUDA float16.** It is the fastest configuration *and* the most
accurate one on Portuguese. That is not the trade-off anyone expected, and the reason is in
the transcripts below: `large-v3` silently drops whole phrases on the 45-second clip.

Six clips, same clips through every configuration: Portuguese and English at roughly 3s, 15s
and 60s. Cells are **time-to-first-segment / time-to-full-transcript**, in seconds, after one
warm-up pass. Decoding settings are the server's own (beam 5, VAD filter), so this measures
the server rather than a tuned-up variant of it.

| configuration | pt 2.6s | pt 10.1s | pt 45.1s | en 4.2s | en 10.6s | en 50.2s | WER pt | WER en |
|---|---|---|---|---|---|---|---|---|
| `base` · CPU int8 **(old default)** | 0.43 / 0.43 | 0.60 / 0.60 | 0.95 / 1.69 | 0.42 / 0.42 | 0.50 / 0.50 | 0.79 / 1.82 | 12.4% | 0.6% |
| `large-v3` · CPU int8 | 5.21 / 5.21 | 6.50 / 6.50 | 10.89 / 18.72 | 5.12 / 5.12 | 6.19 / 6.19 | 9.38 / 18.75 | 9.4% | 0.0% |
| `base` · GPU fp16 | 0.07 / 0.07 | 0.20 / 0.20 | 0.46 / 0.64 | 0.09 / 0.09 | 0.18 / 0.18 | 0.35 / 0.56 | 12.4% | 0.6% |
| `medium` · GPU fp16 | 0.14 / 0.14 | 0.30 / 0.30 | 0.74 / 1.22 | 0.13 / 0.13 | 0.25 / 0.25 | 0.63 / 1.02 | 3.5% | 0.0% |
| `large-v3` · GPU int8_float16 | 0.19 / 0.19 | 0.40 / 0.40 | 1.04 / 1.50 | 0.18 / 0.19 | 0.35 / 0.35 | 0.84 / 1.39 | 8.2% | 0.0% |
| `large-v3` · GPU fp16 | 0.19 / 0.19 | 0.40 / 0.40 | 1.03 / 1.48 | 0.17 / 0.17 | 0.34 / 0.34 | 0.84 / 1.39 | 7.1% | 0.0% |
| `large-v3` · GPU fp16 batched | 0.17 / 0.17 | 0.36 / 0.36 | 0.95 / 0.95 | 0.15 / 0.15 | 0.29 / 0.29 | 0.81 / 0.82 | 4.7% | 0.6% |
| `large-v3-turbo` · GPU int8_float16 | 0.09 / 0.09 | 0.16 / 0.16 | 0.37 / 0.55 | 0.10 / 0.10 | 0.16 / 0.16 | 0.30 / 0.50 | 2.9% | 0.0% |
| `large-v3-turbo` · GPU fp16 batched | 0.08 / 0.08 | 0.15 / 0.15 | 0.32 / 0.32 | 0.10 / 0.10 | 0.13 / 0.13 | 0.33 / 0.33 | 3.5% | 0.6% |
| `large-v3-turbo` · GPU fp16 **(winner)** | 0.10 / 0.10 | 0.17 / 0.17 | 0.40 / 0.59 | 0.11 / 0.11 | 0.15 / 0.15 | 0.32 / 0.54 | 2.9% | 0.0% |

Word error rate is accent- and punctuation-insensitive with small digits spelled out, so
"3 segundos" and "três segundos" both count as correct (`bench/score.py`).

## The four questions

**1. Does it fit on the GPU, and how much faster is it?** Yes, with room to spare, and the
answer is roughly **30x**. Measured VRAM with the TTS engine resident (~15.9 GB free):

| model / compute type | VRAM |
|---|---|
| `large-v3` float16 | 4217 MiB |
| `large-v3` int8_float16 | 2233 MiB |
| `large-v3-turbo` float16 | 2521 MiB |
| `large-v3-turbo` int8_float16 | 1465 MiB |
| `medium` float16 | 2361 MiB |

`large-v3` goes from 18.72s to 1.48s on the 45-second Portuguese clip. **`int8_float16` buys
nothing on this card** — 1.50s vs 1.48s for `large-v3`, 0.55s vs 0.59s for turbo, both inside
the noise. It halves VRAM, which does not matter when 15 GB is free, so the default stays
float16 and keeps the quantization risk off the table.

**2. What does streaming buy?** As written, nothing at all — **the websocket was broken.**
`MediaRecorder.start(1000)` puts the container header in the first chunk only; every chunk
after it is a bare continuation fragment. The old handler wrote each chunk to its own temp
file and handed it to ffmpeg, so chunk 0 transcribed and chunks 1..N returned
`Invalid data found when processing input`. Reproduced with `bench/ws_probe.py`:

```
chunk 0 (4199B) 13.99s -> {"text":"Preciso que você...","language":"pt"}
chunk 1 (4199B)  0.00s -> {"error":"[Errno 1094995529] Invalid data found ..."}
chunk 2 (4199B)  0.00s -> {"error":"[Errno 1094995529] Invalid data found ..."}
   ... every remaining chunk identical
```

Fixed by accumulating the chunks onto the header and transcribing the whole buffer each time,
which is also what makes partial text possible. Same probe against the new server:

```
chunk 0 0.33s -> Preciso que voe.
chunk 2 0.16s -> Preciso que você transcreva o áudio em português.
chunk 5 0.21s -> ... sem inventar palavras. O servidor roda numa placa de
chunk 7 0.24s -> ... e a latência importa mais do que
chunk 9 0.22s -> ... e a latência importa mais do que qualquer outra coisa neste momento.
```

Text now grows while the sentence is still being spoken, at ~0.2s per one-second chunk against
a growing buffer. That is dictation you can watch. The socket also takes a `reset` text frame
to clear the buffer between utterances, and tags replies `"partial": true` — the client
replaces the transcript rather than appending it.

**3. Is the library old?** No. **`faster-whisper` 1.2.1 is the current release**, so the pin is
already at the top. `BatchedInferencePipeline` has been available since 1.1.0 and is measured
above: it helps only on long audio (turbo 45s clip 0.59s → 0.32s) by giving up incremental
segments, and it costs a little accuracy. For dictation-length utterances it is a wash, so the
default stays unbatched. CTranslate2 could go 4.6.3 → 4.8.1; not taken, since the GPU path
works and this card is not the place to find out what changed. `large-v3-turbo` is available
and is the winner.

**4. Is `large-v3` even the right model?** **No.** On Portuguese `large-v3` scores 7.1% WER
against turbo's 2.9%, while taking 2.5x as long. The failure is not noise — on the 45-second
clip `large-v3` reaches "Palavras difíceis." and then **skips the entire list of hard words**,
jumping to the next sentence. Turbo transcribes the list. `medium` is respectable (3.5% pt) and
also keeps the list, but turbo beats it on both axes, so there is no reason to prefer it.

## What landed

- Default model `large-v3-turbo`, CUDA float16, falling back to CPU int8 when there is no GPU.
- `cuda_libs.py` preloads libcublas/libcudnn from the `nvidia-*-cu12` wheels, because
  CTranslate2 dlopens them by soname and cannot see inside site-packages. No LD_LIBRARY_PATH
  needed at launch.
- The websocket accumulates chunks and streams partial transcripts.
- `WHISPER_MODEL` and `WHISPER_DEVICE=cpu` override the defaults.

End to end through the HTTP endpoint on the new default: 0.13s for a spoken sentence, 0.75s
for 45 seconds of Portuguese. Old default was 0.43s and 1.69s at 12.4% Portuguese WER.

## Reproducing

```bash
uv run python bench/make_clips.py     # synthesize clips from the TTS on :8883
uv run python bench/bench.py          # every configuration, one subprocess each
uv run python bench/score.py          # WER against the reference text
uv run python bench/ws_probe.py ws://127.0.0.1:8878/ws/transcribe bench/clips/pt_15.wav
```

The clips are not committed; `make_clips.py` regenerates them and carries the reference text.

## Transcripts

### pt_60

**reference (text fed to the TTS)**

> Bom dia. Hoje eu quero medir quanto tempo o modelo leva para transcrever a minha voz, porque digitar comando por comando ja esta cansativo demais. A ideia e simples: eu falo, o computador escuta, e o texto aparece antes de eu terminar a frase. Se demorar tres segundos para uma frase de tres segundos, nao serve. Entao vamos testar o modelo grande na placa de video, vamos testar a versao turbo, e vamos testar o modelo medio tambem, para ver se a diferenca de qualidade justifica a diferenca de tempo. Palavras dificeis: manutencao, orcamento, exception, deployment, roboflow, kubernetes, transcricao simultanea. Se essas palavras sairem certas em portugues, o modelo passou no teste. Se sairem em ingles ou viraram outra coisa, o modelo reprovou, por mais rapido que ele seja.

**cpu-base-int8** — WER 12.4%

> Bom dia. Hoje eu quero medir quanto tempo modelo leva para transcrever a minha voz porque digitar comando por comando já está cansativo demais. A ideia é simples. Eu falo, o computador escuta e o texto aparece antes do terminar a frase. Se demorar três segundos para uma frase de três segundos, nao serve. Então vamos testar o modelo grande na placa de vídeo. Vamos testar versar o turbo e vamos testar o modelo médio também para ver se a diferença de qualidade justifica a diferença de tempo. Palavras difíceis, manutencal, orcamento, excepção, deployment, oboflow, Kubernetes, transcricar o simultâneo. Essas palavras saírem certas em português, o modelo passou no teste. Cesar e minha inglês ouviraram outra coisa, o modelo reprovou por mais rápido que ele seja.

**cpu-lv3-int8** — WER 12.4%

> Bom dia. Hoje eu quero medir quanto tempo o modelo leva para transcrever a minha voz, porque digitar comando por comando já está cansativo demais. A ideia é simples. Eu falo, o computador escuta, e o texto aparece antes de eu terminar a frase. Se demorar 3 segundos para uma frase de 3 segundos, não serve. Então vamos testar o modelo grande na placa de vídeo. Vamos testar a versal turbo, e vamos testar o modelo médio também, para ver se a diferenca de qualidade justifica a diferenca de tempo. Essas palavras saírem certas em português, o modelo passou no teste. Se Zyrem e Inglês ouviraram outra coisa, o modelo reprovou, por mais rápido que ele seja.

**cuda-medium-fp16** — WER 4.7%

> Bom dia. Hoje eu quero medir quanto tempo o modelo leva para transcrever a minha voz porque digitar comando por comando já está cansativo demais. A ideia é simples. Eu falo, o computador escuta e o texto aparece antes de eu terminar a frase. Se demorar 3 segundos para uma frase de 3 segundos, não serve. Então vamos testar o modelo grande na placa de vídeo. Vamos testar a Versal Turbo e vamos testar o modelo médio também, para ver se a diferença de qualidade justifica a diferença de tempo. Palavras difíceis. Manutencal, orcamento, excepção, deployment, roboflow, kubernetes, transcrical simultânea. Essas palavras saírem certas em português, o modelo passou no teste. Se serem em inglês ou viraram outra coisa, o modelo reprovou, por mais rápido que ele seja.

**cuda-lv3-fp16** — WER 9.3%

> Bom dia. Hoje eu quero medir quanto tempo o modelo leva para transcrever a minha voz, porque digitar comando por comando já está cansativo demais. A ideia é simples. Eu falo, o computador escuta, e o texto aparece antes de eu terminar a frase. Se demorar 3 segundos para uma frase de 3 segundos, não serve. Então vamos testar o modelo grande na placa de vídeo. Vamos testar a versão turbo, e vamos testar o modelo médio também, para ver se a diferenca de qualidade justifica a diferenca de tempo. Palavras difíceis. Essas palavras saírem certas em português, o modelo passou no teste. Se Xyrem e Inglês o viraram outra coisa, o modelo reprovou, por mais rápido que ele seja.

**cuda-turbo-fp16** — WER 3.9%

> Bom dia. Hoje eu quero medir quanto tempo o modelo leva para transcrever a minha voz, porque digitar comando por comando já está cansativo demais. A ideia é simples. Eu falo, o computador escuta, e o texto aparece antes de eu terminar a frase. Se demorar 3 segundos para uma frase de 3 segundos, now serve. Então vamos testar o modelo grande na placa de vídeo. Vamos testar a versão turbo, e vamos testar o modelo médio também, para ver se a diferenca de qualidade justifica a diferenca de tempo. Palavras difíceis. Manutencal, Orcamento, Exception, Deployment, Roboflow, Kubernetes, Transcrical Simultânea. Essas palavras saírem certas em português, o modelo passou no teste. Se Zyrem em inglês ou viraram outra coisa, o modelo reprovou, por mais rápido que ele seja.


### pt_15

**reference (text fed to the TTS)**

> Preciso que voce transcreva o audio em portugues sem inventar palavras. O servidor roda numa placa de video e a latencia importa mais do que qualquer outra coisa neste momento.

**cpu-base-int8** — WER 13.3%

> Preciso que você transcreva o áudio em português sem inventar palavras. O servidor roda numa placa de vídeo e ela tem se importa mais o que qualquer outra coisa neste momento.

**cpu-lv3-int8** — WER 0.0%

> Preciso que você transcreva o áudio em português sem inventar palavras. O servidor roda numa placa de vídeo e a latência importa mais do que qualquer outra coisa neste momento.

**cuda-medium-fp16** — WER 0.0%

> Preciso que você transcreva o áudio em português sem inventar palavras. O servidor roda numa placa de vídeo e a latência importa mais do que qualquer outra coisa neste momento.

**cuda-lv3-fp16** — WER 0.0%

> Preciso que você transcreva o áudio em português sem inventar palavras. O servidor roda numa placa de vídeo e a latência importa mais do que qualquer outra coisa neste momento.

**cuda-turbo-fp16** — WER 0.0%

> Preciso que você transcreva o áudio em português sem inventar palavras. O servidor roda numa placa de vídeo e a latência importa mais do que qualquer outra coisa neste momento.


### pt_03

**reference (text fed to the TTS)**

> Abre a porta da garagem e acende a luz da cozinha.

**cpu-base-int8** — WER 9.1%

> Abra a porta da garagem e acende a luz da cozinha.

**cpu-lv3-int8** — WER 0.0%

> Abre a porta da garagem e acende a luz da cozinha.

**cuda-medium-fp16** — WER 0.0%

> Abre a porta da garagem e acende a luz da cozinha.

**cuda-lv3-fp16** — WER 0.0%

> Abre a porta da garagem e acende a luz da cozinha.

**cuda-turbo-fp16** — WER 0.0%

> Abre a porta da garagem e acende a luz da cozinha.


### en_60

**reference (text fed to the TTS)**

> Good morning. Today I want to measure how long the model takes to transcribe my voice, because typing every command out by hand has become tiring. The idea is simple: I speak, the computer listens, and the text shows up before I finish the sentence. If it takes three seconds to handle a three second sentence, it is useless. So we are going to test the large model on the graphics card, we are going to test the turbo variant, and we are going to test the medium model too, to see whether the quality difference justifies the time difference. Hard words: maintenance, exception handling, deployment, roboflow, kubernetes, simultaneous transcription. If those come out right, the model passed. If they come out garbled, the model failed, however fast it was.

**cpu-base-int8** — WER 0.8%

> Good morning. Today I want to measure how long the model takes to transcribe my voice, because typing every command out by hand has become tiring. The idea is simple. I speak, the computer listens, and the text shows up before I finish the sentence. If it takes three seconds to handle a three second sentence, it is useless. So we are going to test the large model on the graphics card. We are going to test the turbo variant, and we are going to test the media model, too, to see whether the quality difference justifies the time difference. Hard words, maintenance, exception handling, deployment, roboflow, Kubernetes, simultaneous transcription. If those come out right, the model passed. If they come out garbled, the model failed, however fast it was.

**cpu-lv3-int8** — WER 0.0%

> Good morning. Today, I want to measure how long the model takes to transcribe my voice because typing every command out by hand has become tiring. The idea is simple. I speak, the computer listens, and the text shows up before I finish the sentence. If it takes three seconds to handle a three-second sentence, it is useless. So we are going to test the large model on the graphics card. We are going to test the turbo variant, and we are going to test the medium model, too, to see whether the quality difference justifies the time difference. Hard words, maintenance, exception handling, deployment, RoboFlow, Kubernetes, simultaneous transcription. If those come out right, the model passed. If they come out garbled, the model failed, however fast it was.

**cuda-medium-fp16** — WER 0.0%

> Good morning. Today, I want to measure how long the model takes to transcribe my voice, because typing every command out by hand has become tiring. The idea is simple. I speak, the computer listens, and the text shows up before I finish the sentence. If it takes three seconds to handle a three-second sentence, it is useless. So we are going to test the large model on the graphics card. We are going to test the turbo variant, and we are going to test the medium model, too, to see whether the quality difference justifies the time difference. Hard words, maintenance, exception handling, deployment, Roboflow, Kubernetes, simultaneous transcription. If those come out right, the model passed. If they come out garbled, the model failed, however fast it was.

**cuda-lv3-fp16** — WER 0.0%

> Good morning. Today, I want to measure how long the model takes to transcribe my voice because typing every command out by hand has become tiring. The idea is simple. I speak, the computer listens, and the text shows up before I finish the sentence. If it takes three seconds to handle a three-second sentence, it is useless. So we are going to test the large model on the graphics card. We are going to test the turbo variant, and we are going to test the medium model, too, to see whether the quality difference justifies the time difference. Hard words, maintenance, exception handling, deployment, RoboFlow, Kubernetes, simultaneous transcription. If those come out right, the model passed. If they come out garbled, the model failed, however fast it was.

**cuda-turbo-fp16** — WER 0.0%

> Good morning. Today, I want to measure how long the model takes to transcribe my voice because typing every command out by hand has become tiring. The idea is simple. I speak, the computer listens, and the text shows up before I finish the sentence. If it takes three seconds to handle a three-second sentence, it is useless. So we are going to test the large model on the graphics card. We are going to test the turbo variant, and we are going to test the medium model too, to see whether the quality difference justifies the time difference. Hard words, maintenance, exception handling, deployment, roboflow, kubernetes, simultaneous transcription. If those come out right, the model passed. If they come out garbled, the model failed, however fast it was.

