# Seabass - Python Implementation of the 2f Model (and some of PEAQ)

![logo](docs/assets/seabass.png)

## Usage

### CLI
```shell
seabass estimate-mean-mushra-score \
    --test-path /path/to/test-file.wav \
    --ref-path /path/to/reference-file.wav \
    --output /path/to/output.csv
```

### Python

Evaluating directly from files:
```python
from seabass.core.models.mms import estimate_mean_mushra_score_file

mms = estimate_mean_mushra_score_file(test_path, ref_path)

print(f"Estimated MMS: {mms:3.2f}")
```

Evaluating from loaded signals:
```python
import soundfile as sf
from seabass.core.models.mms import estimate_mean_mushra_score

ref_signal, fs_r = sf.read(ref_path, always_2d=True)
test_signal, fs_t = sf.read(test_path, always_2d=True)

# soundfile returns (n_samples, n_channels) but seabass expectes (n_channels, n_samples)
ref_signal = ref_signal.T 
test_signal = test_signal.T

assert fs_r == fs_t == 48000
# PEAQ only works with 48 kHz signals
# if your signals are not 48 kHz, resample them

mms = estimate_mean_mushra_score(test_signal, ref_signal)

print(f"Estimated MMS: {mms:3.2f}")
```

## The Gotchas

The computation of MOVs within the PEAQ component requires that the audio signal are scaled to have amplitudes in the range `[-32768, 32767]`. By default, `seabass` handles this automatically: it assumes that the audio signals it receives is floating point in the range [-1, 1), and scales them up by `32768`, for compatibility with `soundfile`.

If your audio signals are normalized differently, scale it up manually to `[-32768, 32767]` and pass the argument/flag `signal_dtype='int16'` to the function.


## References
- PEAQ: Peter Kabal's Implementation, [PQEvalAudio](https://www-mmsp.ece.mcgill.ca/Documents/Software/index.html) 
- 2f-Model: [SEBASS-DB](https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS)
---

This project was seeded via `newt init --app-type python`, see [go/python](http://go/python) for more info.
