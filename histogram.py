import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os
from PIL import Image

def calc_histogram(image, channel):
    """
    计算单通道直方图并归一化
    """
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def plot_histogram(image, title):
    """
    绘制并显示图像的 RGB 颜色直方图
    """
    # 分离通道
    channels = cv2.split(image)
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10, 4))
    plt.title(f"{title} Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Normalized Frequency")
    
    # 绘制每个通道的直方图
    for i, (channel, color) in enumerate(zip(channels, colors)):
        hist = calc_histogram(channel, 0)
        plt.plot(hist, color=color, label=f"{'BGR'[i]} Channel")
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def compare_histograms_correlation(img1, img2):
    """
    比较两张图片的颜色直方图，使用相关性
    """
    # 分离通道
    img1_channels = cv2.split(img1)
    img2_channels = cv2.split(img2)
    
    total_correlation = 0.0
    
    # 对每个通道（RGB）计算直方图并比较
    for i in range(3):
        hist1 = calc_histogram(img1_channels[i], 0)
        hist2 = calc_histogram(img2_channels[i], 0)
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        total_correlation += correlation
    
    # 返回平均相关性
    return total_correlation / 3.0