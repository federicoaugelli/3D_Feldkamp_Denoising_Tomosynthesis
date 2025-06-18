## Results
|angles   |noise|rel err             |psnr                |ssim                 |rmse                |
|---------|-----|--------------------|--------------------|---------------------|--------------------|
|-15, 15  |0.00 | 0.8329463005065918 | 11.025908377093222 | 0.200415325317181   | 0.2898489236831665 |
|-8.5, 8.5|0.00 | 0.8593526482582092 | 10.742693468006873 | 0.19586413319069834 | 0.299014687538147  |
|-15, 15  |0.01 | 0.8331770300865173 | 11.023297479660243 | 0.19592756581773943 | 0.2899376153945923 |
|-8.5, 8.5|0.01 | 0.8595587611198425 | 10.74059823486409  | 0.1922153868176137  | 0.2990887761116028 |

## NN performance

1. Train 0: train with NN n.1, custom loss (rmse + psnr + re), 20 epochs: BAD
2. Train 1: train with NN n.1, custom loss (rmse + psnr), 20 epochs
3. Train 2: train with NN n.1, custom loss (rmse + psnr), 50 epochs, adam (lr 1e-5, wd 1e-6, betas)
4. train 3: train with NN n.1, custom loss (...)        , 20 epochs, adam (lr 1e-4, wd 1e-6)
5. Train 4: train with NN n.1, custom loss (rmse + freq), 20 epochs, adam (lr 1e-4, wd 1e-6)
6. Train 5: train with NN n.1, custom loss (rmse + freq), 50 epochs, adam (lr 1e-4, wd 1e-6)
7. train 6: train with NN n.1, custom loss (20*rmse + 60*freq + 20*edge), 20 epochs, adam ...
8. train 7: train with NN n.1, custom loss (20*freq + 70*psnr + 10*edge), 20 epochs, adam ...

| train   | psnr  | ssim   | loss   | val loss | epochs | time |
|---------|-------|--------|--------|----------|--------|------|
| train 1 | 30.20 | 0.7313 | 0.1598 | 0.2540   | 20     | 11m  |
| train 2 | 28.39 | 0.7119 | 0.1470 | 0.2767   | 50     | 36m  |
| train 3 |       |        |        |          | 20     | 11m  |
| train 4 | 25.38 | 0.5622 | 10.8865| 13.1988  | 20     | 11m  |
| train 5 | 5.13  | 0.0488 | 8.2243 | 12.5267  | 50     | 35m  |
| train 6 | 23    | 0.5    | 9.2875 | 11.5997  | 20     | 11m  |
| train 7 | BAD   | BAD    | BAD    | BAD      | 20     | 11m  |
