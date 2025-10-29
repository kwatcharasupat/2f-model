<div style="text-align:center">
  <img src="docs/docs/assets/seabass-nobg.png" alt="the Seabass logo" width="250" />
</div>

# Seabass - Python Implementation of the 2f Model

Seabass is a Python implementation of the **2f-model** developed by the Internaltional Audio Laboratories Erlangen.
It is a model for estimating subjective quality of separated audio source signals, specifically the mean MUSHRA score. 
The part of this code, particularly the computation of the Model Output Variables are based on the MATLAB implementation, 
[PQEvalAudio](http://www-mmsp.ece.mcgill.ca/Documents/Software/index.html), by Peter Kabal.
See [here](https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS) for more details about the 2f-model.

## Installation

```shell
git clone this-repository
cd 2f-model
pip install -e .
```

## Usage

> The computation of MOVs requires that the audio signal are scaled to have amplitudes in the range `[-32768, 32767]`. 
    By default, `seabass` handles this automatically when using the CLI mode or `estimate_mean_mushra_score_file`, 
    but _not always_ when using `estimate_mean_mushra_score`.
>
>   When reading from files, `soundfile` reads the header and normalize it to floating point in the range [-1, 1). 
    `seabass` then scales them up by `32768`.

### CLI
```shell
seabass --test-path /path/to/test-file.wav \
    --ref-path /path/to/reference-file.wav \
    --output-path /path/to/output.json
```

### Python

#### Evaluating directly from files
```python
from seabass.core.models.mms import estimate_mean_mushra_score_file

mms = estimate_mean_mushra_score_file(test_path, ref_path)

print(f"Estimated MMS: {mms:3.2f}")
```



#### Evaluating from loaded signals

>    When using loaded audio, make sure that your audio signals are either
>
>    - in the range `[-32768, 32767]` and pass the argument `signal_dtype='int16'` to the function, or
>    - in the range `[-1, 1)` and pass the argument `signal_dtype='double'` to the function.
>    
>    If your audio signals are in the range `[-1, 1)`, they will be multiplied by `32768` to scale them to `[-32768, 32768)`.
>    If this is not what you want, you can scale them yourself before passing them to the function with `signal_dtype='int16'`.

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
