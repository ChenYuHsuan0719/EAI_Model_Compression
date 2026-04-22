---
title: N26141907_Lab4_Report

---

# Lab4 - Homework Template
### 1. About Knowledge Distillation (15%)
* feature-base and response-base
* logits 是模型全連接層輸出的未經 Softmax 處理的原始分數向量，可以提供 Teacher model 額外提供的 Dark Knowledge ; Teacher 的 Logits 經過 Softmax 處理後，會產生一個平滑的概率分佈。這個分佈叫 Soft Targets ，比 Hard Label 更有學習參考價值，學生 model 的目標就是去模仿這個 soft target 分佈 ; 在算 Distillation Loss 時，logits 再經過 KL 散度來衡量stident model 的 Softmax 輸出與 teacher model 的 Softmax 輸出之間的差異
 * T 控制 Logits 轉換，T 值越高，Logits 經過 Softmax 轉換後的概率分佈越平滑，所有類別的概率差異會被縮小，分布變均勻 ; 原本 Logits 中微小的差異 (dark knowledge) 在概率分佈中的相對重要性會被放大。這使得學生模型更容易區分出非正確類別之間的細微差別
  *  features 來自架構中的卷積層，layer 1、2、3、4的輸出

### 2. Response-Based KD (30%)
  * KD 相關論文建議 $T$ 參數設定在 $1$ 到 $20$ 之間，尤其 $T=3$ 到 $T=8$ 是最常被使用的範圍，$5$ 算中偏高，足夠將 Logits 分佈進行平滑化，使那些極低的暗知識變得相對明顯，讓 student model 更能學習到 Teacher model 類別相似性結構
  * alpha 選 $0.5$ 因為我覺得學生模型應該同時重視學習正確標籤和模仿 teacher model 的知識，還能防止過度依賴 Teacher model 的輸出分佈
  * $$
  \mathcal{L}_{\text{KD}} = (1-\alpha) \mathcal{L}(\mathbf{W}_S, x) +  \alpha \cdot \tau^2 \cdot \text{KLdiv}(Q_S, Q_T)
  $$
  * loss_re() 要傳入 students_logits 、 teacher_logits 和 label 。label 用於算出 student model 的 cross_entrophy loss，其他的再用來算出 KL Divergence Loss。studnent_logit 要用 log_softmax 的原因是 nn.KLDivLoss 他的要求是 input must be log-probabilities
target must be probabilities，來自於他的公式
* 
  ```python
  def loss_re(student_logits, teacher_logits, labels):
    T = 5.0 # Set temperature parameter
    alpha = 0.5 # Set weighting parameter

    # Implement loss calculation 
    # Hard Loss 學習真實標籤
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft Loss 模仿老師的輸出分佈
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)

    loss = (1. - alpha) * hard_loss + alpha * soft_loss

    return loss
    ```

### 3. Feature-based KD (30%)
  * 在 forward function 中，當資料 x 通過每個 layer 之後，立即將當前的 x 賦值給一個 feature

  * $$\mathcal{L}_{\text{feature}} = \|\mathbf{f}_s - \mathbf{f}_t\|_2^2
  $$
* Task Loss 用 CrossEntrophy 計算，Feature Loss 用 MSE 計算，最後結合兩者
* 
  ```python
  def loss_fe(student_features, teacher_features, student_logits, labels): 
    # Feature Loss 模仿中間層特徵
    feature_loss = 0.0
    for s_feat, t_feat in zip(student_features, teacher_features):
        feature_loss += F.mse_loss(s_feat, t_feat)
    
    # Task Loss 學習正確分類
    # 訓練 Student 的最後一層 (FC Layer)
    task_loss = F.cross_entropy(student_logits, labels)
    
    # Total Loss: 結合兩者
    alpha = 0.01
    loss = task_loss + alpha * feature_loss

    return loss```
### 4. Comparison of student models w/ & w/o KD (5%)

|                            | loss     | accuracy |
| -------------------------- | -------- | -------- |
| Teacher from scratch       | 0.41     | 86.82     |
| Student from scratch       | 0.50     | 85.91     |
| Response-based student     | 0.56     | 87.89     |
| Featured-based student     | 0.47     | 87.79     |

### 5. Implementation Observations and Analysis (20%)

  * Response-based student 和 Featured-based student 都比 teacher model 還要高
  * 我去看 paper 有些會有這樣的狀況，所以應該是正常的
  * 忘記 student feature channel 數與 teacher 不同，需要調 channel 數，調完就成功了。
  