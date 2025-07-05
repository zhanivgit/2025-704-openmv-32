import sensor
import time
import ml
# --- 初始化摄像头 ---
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA) # 恢复为QVGA分辨率
sensor.skip_frames(time=2000)
# --- 加载数字识别模型 ---
model = ml.Model("trained.tflite", load_to_fb=True)
norm = ml.Normalization(scale=(0, 1.0))
print("成功加载TFLite模型。")

# --- 红色十字检测的阈值 ---
# 调整红色范围，使其更容易被检测到
red_threshold = (0, 100, 20, 127, 0, 127)

clock = time.clock()
last_print_time = 0

# --- 状态机 ---
STATE_LEARNING, STATE_DETECTING = 0, 1
current_state = STATE_LEARNING
target_number = None

# --- 学习确认机制 ---
LEARNING_CONFIRMATION_FRAMES = 20 # 需要连续识别20次相同才确认
learning_buffer = []

print("请将要学习的数字放在摄像头中央...")

while True:
    clock.tick()
    img = sensor.snapshot()

    # --- 状态机逻辑 ---
    if current_state == STATE_LEARNING:
        # 步骤 1: 在屏幕中央寻找数字进行学习
        img_w, img_h = img.width(), img.height()
        learn_roi_rect = (img_w // 2 - 50, img_h // 2 - 30, 100, 100) # 恢复学习ROI尺寸和偏移量
        img.draw_rectangle(learn_roi_rect, color=(255, 255, 0)) # 黄色学习框

        # 从ROI中提取图像并进行处理
        roi_img = img.copy(roi=learn_roi_rect)
        processed_img = roi_img.binary([(0, 60)]).dilate(2)
        
        # 运行数字识别
        input_data = [norm(processed_img)]
        result = model.predict(input_data)[0].flatten().tolist()
        confidence = max(result)
        predicted_number = result.index(confidence)

        # 步骤 2: 检查识别置信度并进行学习确认
        if confidence > 0.8:
            learning_buffer.append(predicted_number)
            if len(learning_buffer) > LEARNING_CONFIRMATION_FRAMES:
                learning_buffer.pop(0)

            # 检查是否连续N次识别到相同数字
            if len(learning_buffer) == LEARNING_CONFIRMATION_FRAMES and all(n == learning_buffer[0] for n in learning_buffer):
                target_number = learning_buffer[0]
                current_state = STATE_DETECTING
                print("已稳定识别并学习数字: %d. 现在开始寻找红色十字." % target_number)
        else:
            learning_buffer.clear() # 置信度不够则清空，防止错误累积

    elif current_state == STATE_DETECTING:
        # 步骤 1: 寻找红色十字
        # 降低像素和面积阈值，使其更容易被检测到
        blobs = img.find_blobs([red_threshold], pixels_threshold=200, area_threshold=200, merge=True)
        if not blobs:
            continue # 没有找到十字，继续下一帧

        cross_blob = max(blobs, key=lambda b: b.area())
        img.draw_cross(cross_blob.cx(), cross_blob.cy(), color=(0, 0, 0), size=10)

        # --- 新增边界检查 ---
        img_w, img_h = img.width(), img.height()
        # 检查十字是否足够远离边缘，以容纳左右两个70x70的ROI
        # 左侧ROI的x起始点为 cx - 80，右侧ROI的x结束点为 cx + 10 + 70 = cx + 80
        # 两个ROI的y结束点为 cy + 10 + 70 = cy + 80
        if (cross_blob.cx() < 80 or
            cross_blob.cx() + 80 > img_w or
            cross_blob.cy() + 80 > img_h):
            # 如果十字太靠近边缘，无法安全定义ROI，则跳过当前帧的数字识别
            print("警告: 红色十字太靠近图像边缘，跳过数字识别。")
            continue
        # --- 边界检查结束 ---

        if target_number is None:
            continue # 没有目标数字，继续下一帧

        # 步骤 2: 在十字周围识别目标数字
        # 定义相对于十字中心的ROI - 调整间距
        roi_definitions = {
            "left_num": (cross_blob.cx() - 75, cross_blob.cy() + 10, 70, 70), # 左侧，调整间距和尺寸
            "right_num": (cross_blob.cx() + 5, cross_blob.cy() + 10, 70, 70) # 右侧，调整间距和尺寸
        }

        # 绘制ROI框 (黑色)
        for r_name, r_rect in roi_definitions.items():
            img.draw_rectangle(r_rect, color=(0, 0, 0)) # 黑色框标记ROI

        for r_name, r_rect in roi_definitions.items():
            # 从ROI中提取图像并进行处理
            roi_img = img.copy(roi=r_rect)
            processed_img = roi_img.binary([(0, 60)]) # 恢复为原始二值化方法
            
            input_data = [norm(processed_img)]
            result = model.predict(input_data)[0].flatten().tolist()
            confidence = max(result) # 获取当前识别的置信度
            predicted_number = result.index(confidence)

            # 步骤 3: 核心逻辑: 只在我给定的数字匹配且置信度足够高时才响应
            if predicted_number == target_number and confidence > 0.7: # 新增置信度阈值
                img.draw_rectangle(r_rect, color=(0, 255, 0)) # 绿色框标记ROI
                img.draw_string(r_rect[0], r_rect[1] - 15, str(predicted_number), color=(0, 255, 0), scale=2)
                
                if time.ticks_diff(time.ticks_ms(), last_print_time) > 1000:
                    last_print_time = time.ticks_ms()
                    print("在 %s 区域找到目标数字: %d (置信度: %.2f)" % (r_name, predicted_number, confidence)) # 打印置信度
            else:
                # 如果不匹配或置信度不够，绘制红色框以示未识别
                img.draw_rectangle(r_rect, color=(255, 0, 0)) # 红色框标记未识别
