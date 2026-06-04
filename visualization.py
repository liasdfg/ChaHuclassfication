import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from rembg import remove
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

def process_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_rmbg = remove(image)
    background = Image.new("RGB", image_rmbg.size, (0, 0, 0))
    background.paste(image_rmbg, mask=image_rmbg.split()[3])
    final_image = background
    return transform(final_image).unsqueeze(0), image

if __name__ == "__main__":
    class_type = "natural shape type"
    folder_path = "./picture_test"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    map_path = "./label_json/label_map(" + class_type + ").json"
    with open(map_path,'r') as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    num_classes = len(label_map)
    model = get_model("resnet34",num_classes)
    model.load_state_dict(torch.load(f"./model_save/best_model({class_type}).pt",
                                     map_location=device))

    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    image_paths = []
    for ex in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ex)))
    if not image_paths:
        print("文件夹中没有图片")
        exit()
    for image_path in image_paths:
        try:
            img_tensor, img_org = process_image(image_path, transform)
            img_tensor = img_tensor.to(device)
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