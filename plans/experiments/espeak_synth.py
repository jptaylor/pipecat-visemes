"""Synthesize speech with espeak-ng via ctypes, capturing phoneme events with audio positions."""
import ctypes, ctypes.util, numpy as np

class EspeakEvent(ctypes.Structure):
    class _Id(ctypes.Union):
        _fields_ = [("number", ctypes.c_int), ("name", ctypes.c_char_p), ("string", ctypes.c_char * 8)]
    _fields_ = [
        ("type", ctypes.c_int), ("unique_identifier", ctypes.c_uint), ("text_position", ctypes.c_int),
        ("length", ctypes.c_int), ("audio_position", ctypes.c_int), ("sample", ctypes.c_int),
        ("user_data", ctypes.c_void_p), ("id", _Id),
    ]

EVENT_WORD, EVENT_SENTENCE, EVENT_END, EVENT_PHONEME = 1, 2, 5, 7
_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.POINTER(EspeakEvent))

_lib = None
_rate = 0
_audio = []
_events = []

def _callback(wav, numsamples, events):
    if numsamples > 0 and wav:
        _audio.append(np.ctypeslib.as_array(wav, shape=(numsamples,)).copy())
    i = 0
    while True:
        ev = events[i]
        if ev.type == 0:
            break
        if ev.type == EVENT_PHONEME:
            _events.append((ev.audio_position / 1000.0, ev.id.string.decode("utf-8", "replace"), ev.length, ev.text_position))
        elif ev.type in (EVENT_WORD, EVENT_SENTENCE, EVENT_END):
            _events.append((ev.audio_position / 1000.0, {1: "<word>", 2: "<sent>", 5: "<end>"}[ev.type], ev.length, ev.text_position))
        i += 1
    return 0

_cb_ref = _CB(_callback)

def init():
    global _lib, _rate
    if _lib is not None:
        return _rate
    path = ctypes.util.find_library("espeak-ng") or "libespeak-ng.so.1"
    _lib = ctypes.CDLL(path)
    # AUDIO_OUTPUT_SYNCHRONOUS=2; options: PHONEME_EVENTS(1) | PHONEME_IPA(2)
    _rate = _lib.espeak_Initialize(2, 0, None, 1 | 2)
    _lib.espeak_SetSynthCallback(_cb_ref)
    return _rate

def synth(text: str, voice: str = "en-us", rate: int = 160, pitch: int = 50, prange: int = 50):
    """Return (int16 audio at init() rate, phoneme events [(t_sec, ipa, len, textpos)])."""
    init()
    _audio.clear(); _events.clear()
    _lib.espeak_SetVoiceByName(voice.encode())
    _lib.espeak_SetParameter(1, rate, 0)    # espeakRATE
    _lib.espeak_SetParameter(3, pitch, 0)   # espeakPITCH
    _lib.espeak_SetParameter(4, prange, 0)  # espeakRANGE
    b = text.encode("utf-8")
    ret = _lib.espeak_Synth(b, len(b) + 1, 0, 1, 0, 0, None, None)
    _lib.espeak_Synchronize()
    audio = np.concatenate(_audio) if _audio else np.zeros(0, dtype=np.int16)
    return audio, list(_events)

if __name__ == "__main__":
    r = init(); print("rate", r)
    a, ev = synth("Mama made more mashed potatoes for my mother.")
    print(len(a) / r, "s audio;", len(ev), "events")
    for e in ev[:60]: print(e)
