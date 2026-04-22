---
title: Lab 3 Report Template

---

# Lab 3 Report Template

## 1. Model Architecture (10%)

* Describe how the `forward()` method and the `fuse_model()` function were implemented. Explain the rationale behind your design choices, such as why certain layers were fused and how this contributes to efficient inference and quantization readiness.
* QuantizableResNet
  * forward( )
      * x = self.quant(x) :
    量化區域的進入點。QuantStub 會觀察傳入的 fp32 影像資料，並使用計算出的scale 和 zero-point，將其轉換為 qint8 格式。所有後續的量化層都將在這個低精度格式下進行運算。
      * x = self.dequant(x) :
    量化區域的離開點。在模型的全連接層 self.fc 之後，DeQuantStub 會將 8 位元整數的運算結果轉換回 32 位元浮點數。在 PyTorch 中，損失函數通常期望 fp32 的輸入，這也確保了模型的輸出與非量化模型具有相同的資料類型，便於後續處理。
  * fuse_model( )
  融合自己的層，以及觸發所有sub-module的融合
    * tq.fuse_modules(self, ['conv1', 'bn1', 'relu'], ...) :
  融合 ResNet 的 Conv-BN-ReLU 序列，將這三個操作融合成一個單一的    FusedConvBNReLU 運算子，能大幅減少記憶體讀寫並消除 BN 層的運算
    * for m in self.modules(): ... m.fuse_model() :
  遍歷模型中所有的子模組，當它找到一個 QuantizableBasicBlock 或     QuantizableBottleneck 時，它會呼叫該實例自己的 fuse_model()。
* QuantizableBasicblock
  * forward( )
    * 標準（非量化）作法: out = F.relu(out + identity)
    * 量化感知 (Quantization-Aware) 作法 :
  使用了 nn.quantized.FloatFunctional() 提供的 add_relu 函數。這樣做的目的是向 PyTorch 的量化框架說有一個元素相加操作，它後面跟著一個 ReLU。
  * fuse_model( )
    * tq.fuse_modules(self, ['conv1', 'bn1', 'relu'], ...) :
  融合區塊中的第一個 Conv-BN-ReLU 序列
    * tq.fuse_modules(self, ['conv2', 'bn2'], ...) :
  融合區塊中的第二個 Conv-BN 序列，這裡沒有 relu 因為 relu 是在與殘差 identity 相加之後才用，這個模式已經由 forward 方法中的 self.add_relu 處理了。
    * tq.fuse_modules(self.downsample, ["0", "1"], ...) :
  如果 downsample 層存在，它內部也會有一個 Conv2d ("0") 和一個 BatchNorm2d ("1")。這行程式碼負責將它們也融合起來
* QuantizableBottleneck 
  * forward( )
  與 BasicBlock 的原理相同，只是它有三個卷積層。
  * fuse_model( )
    * tq.fuse_modules(self, ['conv1', 'bn1', 'relu1'], ...) :
    融合第一個 1x1 卷積、BN 和 ReLU。
    * tq.fuse_modules(self, ['conv2', 'bn2', 'relu2'], ...) :
    融合第二個 3x3 卷積、BN 和 ReLU。
    * tq.fuse_modules(self, ['conv3', 'bn3'], ...) :
    融合第三個 1x1 卷積和 BN，這裡沒有 relu，因為 relu 是在最後的殘差相加時才由 self.skip_add_relu 處理的。
    * tq.fuse_modules(self.downsample, ["0", "1"], ...) :
    如果存在，同樣融合 downsample 層，downsample層通常是conv + bn。
    
* 沒有fuse BN 的話
    在量化 INT8 推論時，每一層都會 quantize → dequantize，而越多層會越多誤差
    * 如何fuse
    數學上證明，BN層的參數可以融合進Conv層
 * add_relu功用
    加完殘差後，通常會有個relu，這邊會需要fuse 殘差加法+Relu，不然也會需要多次quant→dequant，造成acc下降
    
## 2. Training and Validation Curves (10%)

* Provide plots of **training vs. validation loss** and **training vs. validation accuracy** for your best baseline model.
* QAT
 ![QAT_loss_accuracy](https://hackmd.io/_uploads/Sk1Hpn5e-g.png)
 ![QAT_loss_accuracy-1](https://hackmd.io/_uploads/H13Mm6qeZe.png)

* Baseline 
![loss_accuracy](https://hackmd.io/_uploads/S1XAqi9lZe.png)

* Discuss whether overfitting occurs, and justify your observation with evidence from the curves.
  * baseline存在輕微的過度擬合，雖然valid loss和accuracy仍在持續改善，但訓練集和驗證集之間的差距隨著訓練的進行而擴大
  * QAT我只用2個epoch，再用8個epoch微調，量化微調初期趨勢良好，但在微調階段訓練不穩定

## 3. Accuracy Tuning and Hyperparameter Selection (20%)

Explain the strategies you adopted to improve accuracy:

- **Data Preprocessing:** What augmentation or normalization techniques were applied? How did they impact model generalization?
  * 隨機裁切、水平翻轉、顏色擾動
  * 提升模型對位置偏移的魯棒性、對水平對稱的理解、對不同光線環境更穩定，減少方向上的 overfitting，防止模型記住特定色彩分佈
- **Hyperparameters:** List the chosen hyperparameters (learning rate, optimizer, scheduler, batch size, weight decay/momentum, etc.) and explain why they were selected.
  * 較大的 Batch Size 計算出的梯度相較於小 Batch Size 的梯度噪音更小、更穩定
  * 多分類問題的標準損失函數
  * label smoothing助於提高模型的泛化能力，並防止過度擬合
  * 對於圖像分類任務，optimizer 選 SGD 配合momentum能提供最佳的泛化性能
  * momentum幫助梯度更新累積歷史梯度的方向，能有效加速收斂、減少訓練震盪，並幫助模型跳出局部極小值，0.9最常見
  * Weight Decay使模型使用更小的、更平滑的權重，從而提高模型的泛化能力，大多用5e-4
  * learning rate選 0.1是較安全的
  * Scheduler選CosineAnnealingLR比簡單的階梯式衰減能更好地幫助模型穩定收斂
- **Ablation Study (Optional, +10% of this report):** Compare different hyperparameter settings systematically. Provide quantitative results showing how each parameter affects performance.
  * 使用Adam做為optimizer，batch size改成256，learning rate改為3e-4，跑90個epoch，data augmentation用一樣的，其餘也像下面一樣，我的baseline正確率有比原本高，94.84%。原本的要調很高的epoch，才有機會，但Colab吃不消。改成這個版本後不需要太多epoch就能有94%多的結果。
You may summarize your settings and results in a table for clarity:

| Hyperparameter | Loss Function | Optimizer | Scheduler | Weight Decay / Momentum | Epochs | Final Accuracy |
| -------------- | ------------- | --------- | --------- | ----------------------- | ------ | -------------- |
| Value          |     CrossEntrophy   |        SGD   |     CosineAnnealingLR      |              momentum=0.9, weight_decay=5e-4           |   100     |         93.58       |

## 4. Custom QConfig Implementation (25%)

Detail how your customized quantization configuration is designed and implemented:

1. **Scale and Zero-Point:** Explain the mathematical formulation for calculating scale and zero-point in uniform quantization.
 \begin{align}
\beta &= -\alpha \text{ (symmetric)} \\
s &= \frac{2 \max(|\alpha|, |\beta|)}{q_\max - q_\min} \\
z &= 0 \text{ (signed)} \quad \text{or} \quad 128 \text{ (unsigned)}
\end{align}
* activation
  用的是qunit8(unsigned)，範圍是[0,255]
  * scale
  qmax - qmin = 254(不會用255)
  scale = 2amax / 254 = amax / 127
  * zero_point
  範圍是[0,255]，zero point是128 (不是全對稱)
* weight
  用的是qint8(signed)，範圍是[-128,127]，但會使用[−127,127]
  * scale
  qmax - qmin = 254
  scale = 2amax / 254 = amax / 127
  * zero_point
  範圍是[-127,127]，zero point是0
  
  
2. **CustomQConfig Approximation:** Describe how the `scale_approximate()` function in `CusQuantObserver` is implemented. Why is it useful?
  * 將 MinMax Observer 計算出的浮點數 scale，近似成最接近的二的冪次（power-of-two scale）
  * 若 scale 是：0、負數、NaN
、inf，就直接回傳一個安全值 1.0。
這避免在後續使用 log2(scale) 時產生錯誤
  * 計算 log₂(scale)，取得 scale 所對應的 2 的指數，為後續「找最近的二的冪次」做準備
  * 將 log₂(scale) 四捨五入到最近的整數
  * 限制 exponent 的範圍，避免 exponent 過大或過小，導致 scale 太極端，會破壞 quantization 的分佈
  * 重建近似後的 scale，產生一個真正的 power-of-two scale
  * 確保 scale 永遠是正而且非零。
  * 利於硬體加速，不需要浮點乘法、浮點除法
3. **Overflow Considerations:** Discuss whether overflow can occur when implementing `scale_approximate()` and how to prevent or mitigate it.
  * 當做指數計算 2 ** nearest 可能 overflow
  * 如果 nearest 沒有限制，有可能 nearest = 2000、3000…，而2**2000會overflow，所有 activation / weight 都變成 0 或飽和
  * 檢查 scale 是否 > 0（避免 log2 失效，數學定義）
  * 對 log2(scale) 的結果做 clamp（避免指數爆炸）
  * 防止 approx_scale 變成 0 或 NaN
  * 設定合理的 max_shift_amount（一般 6~10）
## 5. Comparison of Quantization Schemes (25%)

Provide a structured comparison between **FP32, PTQ, and QAT**:

- **Model Size:** Compare file sizes of FP32 vs. quantized models.
- **Accuracy:** Report top-1 accuracy before and after quantization.
- **Accuracy Drop:** Quantify the difference relative to the FP32 baseline.
- **Trade-off Analysis:** Fill up the form below.

| Model   | Size (MB) | Accuracy (%) | Accuracy Drop (%) |
|---------|-----------|--------------|-------------------|
| FP32    |       94.41    |       93.58       |                   |
| PTQ     |        23.66   |        93.02      |        0.56           |
| QAT     |       23.66    |       93.36       |           0.22        |

## 6. Discussion and Conclusion (10%)

- Did QAT outperform PTQ as expected?
  * PTQ 只利用少量校正資料
  PTQ 在量化時並沒有讓模型重新學習，因此量化後的權重與 activation 可能會因 scale 設定不佳而產生較大的量化誤差，導致 accuracy 下跌
  * QAT會在訓練過程中模擬量化誤差
  QAT 在 forward 中插入 fake-quant 讓模型看到量化後的行為，使模型可以透過反向傳播主動調整權重，以補償量化引入的誤差，因此模型更能適應 INT8 表示方式。
  * QAT 優於 PTQ 的結果在預期之內，因為模型學會「預先適應量化」，因此在換成 INT8 後不會有明顯性能下降
  
- What challenges did you face in training or quantization, and how did you address them?
  * 一開始沒看清楚weight和activation是用不同的方式，所以在scale_approximate()那裡統一用一個zero_point處理，導致acc降很多，後來寫if-else去處理就好了
  * baseline acc 一直突破不了95%，後來換了好多方法，包刮data augmentation、optimizer、epochs等等，最高也就94.84%，所以放棄了
- Any feedbacks for Lab3 Quantization?
  * 不好意思我的baseline結果不小心又按了一次執行導致我的結果不見，但PTQ和QAT的結果都還在，可以看到FP32的accuracy，請助教網開一面，謝謝你 :+1: 
