---
title: Lab 6 - Transformer Pruning Report

---

# Lab 6 - Transformer Pruning Report
  
1. **請說明 get_real_idx 實作部分是怎麼做的** 10%
```
def get_real_idx(idxs, fuse_token):
    for i in range(1, len(idxs)):
        ########################################
        #       請實作         #
        ########################################
        tmp = idxs[i - 1]
        if fuse_token:
            B = tmp.size(0)
            tmp = torch.cat([tmp, torch.zeros(B, 1, dtype=tmp.dtype, device=tmp.device)], dim=1)
        idxs[i] = torch.gather(tmp, dim=1, index=idxs[i])
        ########################################

    return idxs
```
* 利用 tmp = idxs[i - 1] 取得上一層留下來的那些 Token 在原始圖片中的真實索引
* fuse_token 是一個 boolean 值，檢查這一層是否需要保留 fuse_token，若需要，被刪除的 tokens fuse 成一個額外的 token 放在序列最後面。
* tmp 的長度只有 len(tmp)，所以要在 tmp 後面補0，避免發生錯誤 

2. **實際在哪些層做了 pruning ?** 10%
```
model = EViT(keep_rate=(1, 1, 1, 0.7) + (1, 1, 0.7) + (1, 1, 0.7) + (1, 1))
```
* keep_rate 是一個長度為 12 的 tuple，定義了每一層的 token 保留率。只有當 keep_rate < 1 時才會 Pruning。第一個數字代表第一層的 keep_rate，接下來以此類推所以可以知道是 layer 4、7、10 要做 prunning
    
3. **如果沒有 get_real_idx 可視化結果會長怎樣，為什麼 ?** 10%
如果沒有執行 get_real_idx，直接拿模型輸出的 idx 去做 mask 視覺化，結果會是錯誤且雜亂的，因為模型輸出的 idx 是相對 index，保留的 Patch 不會對應到圖中對的位置
    
4. **分析視覺化的圖，這些變化代表著什麼 ?** 10%
黑色的部分是prune掉的patch，黑色的地方大部分都是背景和主體沒有關係，影響辨識的程度不高
    
    