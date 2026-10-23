import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------
# 1. 初始化模型與載入 Hugging Face 預訓練權重
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# 建立 U-Net (EfficientNet-B4) 模型結構
model = smp.Unet(
    encoder_name="efficientnet-b4",
    encoder_weights=None,  # 不需載入 ImageNet，待會直接載入 SIIM 專屬權重
    in_channels=3,
    classes=1
)

# 從 Hugging Face 下載預訓練權重
weights_path = hf_hub_download(
    repo_id="vasandesh/pneumothorax-unet-efficientnet-b4", 
    filename="pytorch_model.bin"
)

# 載入權重並清理不需要或不符的 key
raw_state_dict = torch.load(weights_path, map_location=device, weights_only=False)
state_dict = {}
for k, v in raw_state_dict.items():
    # 1. 移除 "unet." 前綴
    new_key = k[5:] if k.startswith("unet.") else k
    
    # 2. 過濾掉模型結構中沒有的 cls_head 參數
    if new_key.startswith("cls_head."):
        continue
        
    state_dict[new_key] = v

# 嚴格模式設定為 strict=False，避免其他微小差異造成報錯
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print("模型與預訓練權重載入成功！")
# ---------------------------------------------------------
# 2. 影像預處理 Pipeline (配合模型的輸入尺寸與標準化)
# ---------------------------------------------------------
# 注意：該模型訓練時採用的尺寸通常為 512x512 或 256x256
IMG_SIZE = 512

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def predict_pneumothorax(image_input, threshold=0.5):
    """
    輸入: PIL Image 或 NumPy Array (RGB)
    輸出: 氣胸機率、是否判斷有氣胸、分割 Mask
    """
    # 預處理
    if isinstance(image_input, str):
        image_pil = Image.open(image_input).convert("RGB")
        image_np = np.array(image_pil)
    else:
        image_np = np.array(image_input)
        
    augmented = transform(image=image_np)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        output = model(input_tensor)
        # 透過 Sigmoid 轉為 0~1 的機率地圖
        pred_mask_prob = torch.sigmoid(output).squeeze().cpu().numpy()

    # 計算整張 X 光片存在氣胸的概率 (取 Mask 中的最大機率或高於閾值的區域)
    has_pneumothorax = np.max(pred_mask_prob) > threshold
    pred_mask_binary = (pred_mask_prob > threshold).astype(np.uint8)

    return image_np, pred_mask_prob, pred_mask_binary, has_pneumothorax

# ---------------------------------------------------------
# 3. 執行測試與繪圖顯示結果
# ---------------------------------------------------------
if __name__ == "__main__":
    # 替換為你的 X 光影像路徑 (PNG/JPG)
    test_img_path = r"C:\Users\noral\Desktop\Pneumothorax\siim-acr-pneumothorax\png_images\0_test_1_.png"

    try:
        orig_img, prob_map, binary_mask, is_positive = predict_pneumothorax(test_img_path)

        print("=" * 40)
        print(f"診斷結果: {'【有氣胸 Pneumothorax Detected】' if is_positive else '【無氣胸 Normal / No Pneumothorax】'}")
        print("=" * 40)

        # 視覺化呈現
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Chest X-Ray")
        plt.imshow(orig_img)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Probability Heatmap")
        plt.imshow(prob_map, cmap="jet")
        plt.colorbar()
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Segmented Mask")
        plt.imshow(orig_img)
        plt.imshow(binary_mask, alpha=0.4, cmap="Reds") # 疊加紅色氣胸區域
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"請替換正確的影像路徑。錯誤訊息: {e}")
