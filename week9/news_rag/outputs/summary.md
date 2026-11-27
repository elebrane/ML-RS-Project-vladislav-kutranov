# HW3 development results (not official benchmark scores)

## D0_june / nollm_nonews
cutoff 2021-06-01, users 100, protocol_hash `50efd4aeb415`, hw3_protocol_hash `0cf8a7448873`

| Method | nDCG@10 all | buyers | n | bought new | n |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.0286 | 0.2599 | 11 | 0.0000 | 5 |
| CBF | 0.0201 | 0.1827 | 11 | 0.0000 | 5 |
| Latest buy | 0.0277 | 0.2520 | 11 | 0.0000 | 5 |
| Latest buy or sell | 0.0271 | 0.2464 | 11 | 0.0000 | 5 |
| C0 initial order (matched control) | 0.0262 | 0.2379 | 11 | 0.0369 | 5 |
| ceiling: perfect block order | 0.0262 | 0.2379 | 11 | 0.0369 | 5 |

seconds: `{"preparation_cohort_and_blocks": 8.6, "recommendation_loop_wall": 0.0, "generation_by_arm_incl_cached": {}, "total_wall": 8.6}`

## D1_may / nollm_nonews
cutoff 2021-05-01, users 1498, protocol_hash `e8a41099ddb7`, hw3_protocol_hash `381cbd19c21f`

| Method | nDCG@10 all | buyers | n | bought new | n |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.0968 | 0.3556 | 408 | 0.2841 | 160 |
| CBF | 0.0878 | 0.3223 | 408 | 0.1933 | 160 |
| Latest buy | 0.1478 | 0.5427 | 408 | 0.3049 | 160 |
| Latest buy or sell | 0.1469 | 0.5393 | 408 | 0.2984 | 160 |
| C0 initial order (matched control) | 0.1486 | 0.5454 | 408 | 0.3084 | 160 |
| ceiling: perfect block order | 0.1545 | 0.5672 | 408 | 0.3639 | 160 |

seconds: `{"preparation_cohort_and_blocks": 14.3, "recommendation_loop_wall": 0.0, "generation_by_arm_incl_cached": {}, "total_wall": 14.4}`

## TUNE_apr / nollm_nonews
cutoff 2021-04-01, users 1491, protocol_hash `5a53a633ce77`, hw3_protocol_hash `f8265839ca5f`

| Method | nDCG@10 all | buyers | n | bought new | n |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.0878 | 0.3118 | 420 | 0.2450 | 180 |
| CBF | 0.0949 | 0.3370 | 420 | 0.1926 | 180 |
| Latest buy | 0.1605 | 0.5696 | 420 | 0.3000 | 180 |
| Latest buy or sell | 0.1562 | 0.5546 | 420 | 0.2937 | 180 |
| C0 initial order (matched control) | 0.1608 | 0.5710 | 420 | 0.3162 | 180 |
| ceiling: perfect block order | 0.1692 | 0.6006 | 420 | 0.3853 | 180 |

seconds: `{"preparation_cohort_and_blocks": 13.8, "recommendation_loop_wall": 0.0, "generation_by_arm_incl_cached": {}, "total_wall": 13.9}`

