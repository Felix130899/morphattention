# 2026-10-01: margin ring test

Finished experiment record, not a pipeline step. `margin_ring_test.py` tested whether the background margin
around a fish mask leaks the genus: a frozen DINOv2 linear probe on image variants that show only certain pixels
(`outer` = background band near the fish vs. `outer_shape` = the same band painted flat). Decision rule fixed
before the run: leak = `outer` − `outer_shape`.

Result (2026-10-01): **safe in both domains**: photos −0.7 % [−3.9, +2.4], X-rays +2.4 % [−0.7, +5.5]; the
positive control (background alone) was far above chance, so the null result is meaningful. Unchanged without
suspected bleed images.

Thesis note (Neo's vault, not in this repo): experiments/2026-10-01-margin-ring-test. Tested by
`tests/test_margin_ring_test.py`.
