import fapp
import histogram
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os
from PIL import Image

def show_images(image_path, match_image_path):
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = np.array(image)

        plt.subplot(2, 2, 1)
        plt.title("orignal image")
        plt.imshow(image)
        plt.axis("off")

        image_max = os.path.join(os.path.dirname(image), match_image_path[0])
        image_max = Image.open(image_max)
        image_max = image_max.resize((224, 224))
        image_max = np.array(image_max)
        plt.subplot(2, 2, 2)
        plt.title("match image top 1")
        plt.imshow(image_max)
        plt.axis("off")

        k = len(match_image_path)

        for i, match_image in enumerate(match_image_path):
            image_1 = Image.open(os.path.join(os.path.dirname(image), match_image))
            image_1 = image_1.resize((224, 224))
            image_1 = np.array(image_1)
            plt.subplot(2, k, k + i + 1)
            plt.title(f"match image {i+1}")
            plt.imshow(image_1)
            plt.axis("off")
        image_number = image_path.split("_front")[0]
        plt.tight_layout()
        #  plt.show()
        plt.savefig(f"show_image_{image_number}.pdf", format='pdf')

def main(file1, file2):
    """主函数"""
    
    # 读取图像
    img1 = cv2.imread(file1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(file2, cv2.IMREAD_COLOR)
    
    if img1 is None or img2 is None:
        print("无法加载图像，请检查文件路径！")
        return
    
    # 特征检测和匹配
    keypoints1, keypoints2, knnMatches = fapp.featuresDetectAndMatch(img1, img2)
    putativeMatches = fapp.ratio_match(knnMatches, 0.7518)
    
    # print(f"Putative Matches = {len(putativeMatches)}")
    # showMatchResult(img1, img2, keypoints1, keypoints2, putativeMatches, "PutativeMatches")
    
    # 运行 FAPP 获取最终匹配
    finalMatches = fapp.runFAPP(keypoints1, keypoints2, putativeMatches)
    
    # print(f"Final Matches = {len(finalMatches)}")
    similarity = len(finalMatches) / min(len(keypoints1), len(keypoints2))
    # showMatchResult(img1, img2, keypoints1, keypoints2, finalMatches, "FinalMatches")


    correlation = histogram.compare_histograms_correlation(img1, img2)
    # correlation = (correlation + 1) / 2  # 将相关性范围调整到 [0, 1]
    
    # 绘制并显示直方图
    # plot_histogram(img1, "Image 1")
    # plot_histogram(img2, "Image 2")
    
    # 显示直方图
    # plt.show()
    return similarity, correlation

if __name__ == "__main__":
    
    image_path = "images"  # 图片路径
    images_list = os.listdir(image_path)
    image_list = [os.path.join(image_path, image_name) for image_name in images_list if "front" in image_name]
    k = 5
    correct_num = 0
    for i in range(len(image_list)):
        similarity = []
        for j in range(len(image_list)):
            s1, s2 = main(image_list[i], image_list[j])
            similarity.append(0.5 * (s1 + s2))
        sorted_pairs = sorted(zip(similarity, images_list), reverse=True)
        sorted_similarities, sorted_names = zip(*sorted_pairs)
        sorted_similarities = list(sorted_similarities)
        sorted_names = list(sorted_names)
        top_k = sorted_names[:k]
        print(f"与 {image_list[i]} 最相似的前 {k} 张图片:{top_k}-----相似度: {sorted_similarities[:k]}")

        if i in [0, 10, 100, 1000, 10000, len(image_list) - 1]:
            show_images(image_list[i], top_k)

        if image_list[i] == top_k[0]:
            correct_num += 1
    accuracy = correct_num / len(image_list)
    print(f"准确率: {accuracy:.2%}")





