from ultralytics import YOLO
from collections import Counter
import os

# ====================== 配置 ======================
MODEL_PATH   = r"D:\1000\EzYOLO-main\runs\detect\runs\train\exp_2\weights\best.pt" # 使用原始字符串避免转义
IMAGE_FOLDER = r"D:\1000\pliers_images" # 使用原始字符串
CONF_THRESH  = 0.4

# 标准配置
TOOL_STD = 1
DIE_STD = 5
# ====================================================

model = YOLO(MODEL_PATH)

print("===== 配件齐全检测（1 tool + 5 die）=====\n")

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(IMAGE_FOLDER, filename)
        results = model(img_path, conf=CONF_THRESH, verbose=False)

        detected = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            if cls_id == 1:
                detected.append("tool")
            elif cls_id == 0:
                detected.append("die")

        cnt = Counter(detected)
        tool_num = cnt.get("tool", 0)
        die_num = cnt.get("die", 0)

        # 计算差额
        tool_diff = tool_num - TOOL_STD
        die_diff = die_num - DIE_STD

        # 状态描述
        status = "✅ 配件齐全" if (tool_num == TOOL_STD and die_num == DIE_STD) else "❌ 不齐全"

        # 显示缺多少
        tool_msg = f"tool: {tool_num}，{'多%d' % abs(tool_diff) if tool_diff>0 else '少%d' % abs(tool_diff)}" if tool_diff !=0 else f"tool: {tool_num}（正常）"
        die_msg = f"die: {die_num}，{'多%d' % abs(die_diff) if die_diff>0 else '少%d' % abs(die_diff)}" if die_diff !=0 else f"die: {die_num}（正常）"

        print(f"[{filename}] {status}")
        print(f"   → {tool_msg}")
        print(f"   → {die_msg}\n")