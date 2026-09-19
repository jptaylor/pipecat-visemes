# CMU pronunciation dictionary

`cmudict.bin.gz` derives from the [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict),
copyright Carnegie Mellon University. Its redistribution terms are preserved in
`CMUDICT-LICENSE`. The pinned revision, source checksum and sizes are in `cmudict.json`.

Rebuild from `server/` with `python -m benchmarks.vendor_cmudict` (network required only
when rebuilding). The deterministic conversion keeps the first pronunciation of ASCII
English words/contractions and removes stress digits. Each record stores the word and
one byte per ARPAbet phone; sorted offsets support binary search in a single shared blob.
No dictionary or neural model is downloaded at runtime.
