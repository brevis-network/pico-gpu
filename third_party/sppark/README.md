# sppark (vendored)

Code in this directory is derived from [sppark](https://github.com/supranational/sppark) at tag
`v0.1.12`, licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). Each file's header
states whether it is an unmodified copy or a modified one and summarises the changes.

| Path | Contents |
|---|---|
| `ff/` | Montgomery-form 32-bit field arithmetic (`mont32_t`), the KoalaBear field defined on it in the style of sppark's `ff/baby_bear.hpp` (`koala_bear.hpp`, written for this project), and the KoalaBear degree-4 extension |
| `ntt/` | The generic NTT kernels and their host-side launch parameters |
| `ntt_koalabear/` | The KoalaBear instantiation: field and domain parameters, the bit-reversal kernel, and an `ntt.cuh` wrapper that loads those parameters ahead of the generic header |
