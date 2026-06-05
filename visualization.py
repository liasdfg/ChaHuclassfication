import torch
import torch.nn as nn
from torchvision import  models
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from rembg import remove
import albumentations as A
def get_model(model_name,class_num): #获取所需模型
    if model_name=="resnet34":
        model = models.resnet34(pretrained=True)
        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in,class_num)
    elif model_name=="googlenet":
        model = models.googlenet(pretrained=True)
        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in,class_num)
    elif model_name=="vgg19":
        model = models.vgg19(pretrained=True)
        num_in = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_in,class_num)
    else:
        model = None
        print("不支持", {model_name})
    return model

def process_image(image_path, transform, normalize):
    image = Image.open(image_path).convert('RGB')
    image_rmbg = remove(image)
    mask = image_rmbg.split()[3]
    image_np = np.array(image)
    mask_np = np.array(mask)
    # 裁剪原图
    nonzero = np.argwhere(mask_np > 0)
    if len(nonzero) > 0:
        ymin, xmin = nonzero.min(axis=0)
        ymax, xmax = nonzero.max(axis=0)
        h, w = mask_np.shape
        ymin = max(0, ymin - 15)
        xmin = max(0, xmin - 15)
        ymax = min(h, ymax + 15)
        xmax = min(w, xmax + 15)
        image_np = image_np[ymin:ymax, xmin:xmax]
        mask_np = mask_np[ymin:ymax, xmin:xmax]
    mask_np = mask_np / 255.0

    if transform:
        transformed = transform(image=image_np, mask=mask_np)
        image_np = transformed['image']
        mask_np = transformed['mask']
    res_image = (image_np * mask_np[..., np.newaxis]).astype(np.uint8)
    res_image = normalize(image=res_image)['image']
    res_image = np.transpose(res_image, (2, 0, 1))
    image_tensor = torch.tensor(res_image, dtype=torch.float32).unsqueeze(0)
    return image_tensor, image

if __name__ == "__main__":
    class_type = "natural shape type" #更改决定预测的类
    folder_path = "./picture_test"   #测试图片的路径，需要测试的图片放在文件夹里即可
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    map_path = "./label_json/label_map(" + class_type + ").json"
    with open(map_path,'r') as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    num_classes = len(label_map)
    print("标签数量",num_classes)
    model = get_model("resnet34",num_classes)    #更换模型
    model.load_state_dict(torch.load(f"./model_save/best_model({class_type}).pt",
                                     map_location=device))

    model.to(device)
    model.eval()
    test_transform = A.Compose([
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0, 0, 0), mask_value=0)
    ])
    normalize_op = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    image_paths = []
    for ex in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ex)))
    if not image_paths:
        print("文件夹中没有图片")
        exit()
    for image_path in image_paths:
        try:
            img_tensor, img_org = process_image(image_path, test_transform, normalize_op)
            img_tensor = img_tensor.to(device)

            # import torchvision.transforms as T # 图片可视化
            # debug_tensor = img_tensor[0].cpu().clone()
            # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            # debug_tensor = debug_tensor * std + mean
            # debug_tensor = torch.clamp(debug_tensor, 0, 1)
            # debug_img = T.ToPILImage()(debug_tensor)
            # debug_filename = f"debug_model_input_{os.path.basename(image_path)}"
            # debug_img.save(debug_filename)
            # print(f"已保存模型实际看到的输入图: {debug_filename}")

            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]

            predicted_idx = torch.argmax(probabilities).item()
            predicted_class = label_map[predicted_idx]
            confidence = probabilities[predicted_idx].item()

            class_perf = [(label_map[i], probabilities[i].item()) for i in range(num_classes)]
            class_perf.sort(key=lambda x: x[1])

            categories = [x[0] for x in class_perf]
            probs = [x[1] for x in class_perf]

            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(np.array(img_org))
            plt.title(f"Predicted: {predicted_class} ({confidence:.2%})", fontsize=14)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            bars = plt.barh(categories, probs, color='skyblue')
            plt.xlabel('Probability')
            plt.title('Class Probabilities', fontsize=14)
            plt.xlim(0, 1.1)
            plt.grid(axis='x', linestyle='--', alpha=0.7)

            for bar, cat in zip(bars, categories):
                if cat == predicted_class:
                    bar.set_color('orange')
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.2%}',
                         va='center', ha='left')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"处理图片 {image_path} 出错: {e}")