---
title: Lab 6 - Transformer Quantization Report

---

# Lab 6 - Transformer Quantization Report

1. **What's the difference between SmoothQuant and the method in Lab3?** 
Lab3 使用的是 per-tensor 的 Quant 方法，SmoothQuant 的是對 activation 使用 per-token，對 weight 是使用 per-channel

2. **When applying SmoothQuant, where do activation values get divided by the smooth factor?**
對 activation 做除以 smooth factor 是利用 LayerNorm 的 weight 和 bias，對他們均除以 smooth factor

3. **How is the smooth factor being calculated?**
以 per-channel 的方式找 activation 和 weight 的最大值，將兩陣列相除後開根號可得 smooth factor

4. **What's the difference between ViT-S and CNN models when doing quantization?**
CNN 會需要 fuse module，ViT-S 不需要。CNN 的 activation 不會有太大的 outliers，所以不需要前處理。

5. **What's your observation on the visualization of weight and activation values distribution?**
activation 從有一個有很明顯的 outliers 變平滑，weight 原本就很平滑，難度轉移後變得比較不平滑，但還是蠻平滑的