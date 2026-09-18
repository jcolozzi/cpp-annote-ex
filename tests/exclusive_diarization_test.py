"""Integration checks using the shared C API and a PCM16 WAV recording.

Usage: python tests/exclusive_diarization_test.py <library> <seg.onnx> <emb.onnx> <audio.wav>
Put ONNX Runtime's shared library next to the supplied library before running.
"""
import array
import ctypes as ct
import json
import sys
import wave
from pathlib import Path


def coverage(turns):
    merged = []
    for t in sorted(turns, key=lambda t: t["start"]):
        if merged and t["start"] <= merged[-1][1] + 1e-4:
            merged[-1][1] = max(merged[-1][1], t["end"])
        else:
            merged.append([t["start"], t["end"]])
    return merged


def main():
    library, seg, emb, wav = sys.argv[1:]
    lib = ct.CDLL(str(Path(library).resolve()))
    lib.cpp_annote_init.argtypes = [ct.c_char_p, ct.c_char_p]
    lib.cpp_annote_init.restype = ct.c_void_p
    lib.cpp_annote_free.argtypes = [ct.c_void_p]
    lib.cpp_annote_free_string.argtypes = [ct.c_void_p]
    for name in ("cpp_annote_diarize", "cpp_annote_diarize_exclusive"):
        fn = getattr(lib, name)
        fn.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int,
                       ct.c_int, ct.POINTER(ct.c_void_p)]
        fn.restype = ct.c_int
    ctx = lib.cpp_annote_init(seg.encode(), emb.encode())
    assert ctx, "model initialization failed"

    def run(samples, sr, exclusive):
        buf = (ct.c_float * len(samples))(*samples)
        out = ct.c_void_p()
        fn = (lib.cpp_annote_diarize_exclusive if exclusive
              else lib.cpp_annote_diarize)
        assert fn(ctx, buf, len(buf), sr, ct.byref(out)) == 0
        try:
            return json.loads(ct.string_at(out))
        finally:
            lib.cpp_annote_free_string(out)

    try:
        with wave.open(wav, "rb") as f:
            assert f.getsampwidth() == 2, "PCM16 input required"
            sr, channels = f.getframerate(), f.getnchannels()
            pcm = array.array("h", f.readframes(f.getnframes()))
            if sys.byteorder != "little":
                pcm.byteswap()
            samples = [sum(pcm[i:i+channels]) / (32768.0 * channels)
                       for i in range(0, len(pcm), channels)]
        normal = run(samples, sr, False)
        exclusive = run(samples, sr, True)
        assert normal and exclusive, "speech fixture must produce turns"
        assert all(t["end"] > t["start"] for t in exclusive)
        assert all(a["end"] <= b["start"] + 1e-4
                   for a, b in zip(exclusive, exclusive[1:])), "overlapping exclusive turns"
        assert {t["speaker"] for t in exclusive} <= {t["speaker"] for t in normal}
        a, b = coverage(normal), coverage(exclusive)
        assert len(a) == len(b), "speech regions changed"
        assert all(abs(x-y) < 1e-4 for p, q in zip(a, b) for x, y in zip(p, q))
        for mode in (False, True):
            assert run([0.0] * 160000, 16000, mode) == [], "silence became speech"
        print(f"PASS: {len(normal)} regular turns, {len(exclusive)} exclusive turns; "
              "no exclusive overlap, matching speech coverage and speaker IDs, silence preserved")
    finally:
        lib.cpp_annote_free(ctx)


if __name__ == "__main__":
    main()
