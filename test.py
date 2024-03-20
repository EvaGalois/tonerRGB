from scipy.optimize import minimize
from pprint import pprint
import numpy as np

# 通过客户提供的样片，通过此程序输入进想要的颜色（RGB）值，然后由这个数据库里面的十五中色粉选择出最佳的三四种色粉配置比例。
def rgb_to_cmyk(rgb):
    # Normalize RGB values to [0, 1]
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    # Calculate CMYK values
    k = 1 - max(r, g, b)
    c = (1 - r - k) / (1 - k) if k != 1 else 0
    m = (1 - g - k) / (1 - k) if k != 1 else 0
    y = (1 - b - k) / (1 - k) if k != 1 else 0
    # Convert CMYK values from [0, 1] to [0, 100]
    return round(c * 100), round(m * 100), round(y * 100), round(k * 100)

def rgb_to_cmyk_simple(rgb):
    c = 1 - rgb[0] / 255.0
    m = 1 - rgb[1] / 255.0
    y = 1 - rgb[2] / 255.0
    k = min(c, m, y)
    if k == 1:
        return 0, 0, 0, 1
    c = (c - k) / (1 - k)
    m = (m - k) / (1 - k)
    y = (y - k) / (1 - k)
    return c, m, y, k

pigments_rgb = {"白色色粉": [[(180, 194, 207), (195, 207, 219), (184, 201, 217), (202, 218, 234), (190, 206, 222)], [(239, 242, 247), (250, 251, 253), (251, 252, 253), (248, 252, 253), (248, 252, 255)]],
"二氧化钛": [[(180, 202, 220), (199, 216, 232), (195, 213, 233), (211, 232, 253), (210, 232, 253)], [(235, 240, 244), (242, 246, 249), (246, 250, 253), (234, 235, 240), (248, 252, 255)]],
"红色色粉": [[(188, 48, 57), (168, 45, 38), (200, 55, 52), (195, 54, 36), (192, 57, 63)], [(251, 57, 68), (244, 50, 58), (255, 83, 77), (255, 68, 80), (255, 76, 85)]],
"黄色色粉": [[(77, 115, 58), (76, 100, 52), (98, 128, 68), (93, 120, 43), (100, 128, 51)], [(255, 255, 40), (254, 255, 59), (250, 251, 62), (252, 252, 30), (255, 254, 0)]],
"蓝色色粉": [[(38, 34, 85), (37, 37, 91), (40, 40, 80), (25, 26, 74), (43, 34, 89)], [(42, 160, 230), (32, 184, 247), (10, 182, 246), (2, 163, 235), (0, 140, 225)]],
"绿色色粉": [[(27, 42, 45), (31, 46, 53), (35, 53, 57), (30, 40, 50), (40, 60, 50)], [(108, 232, 198), (89, 223, 270), (88, 230, 174), (81, 215, 156), (76, 228, 171)]],
"酞菁蓝": [[(49, 39, 76), (37, 39, 78), (51, 45, 83), (54, 45, 90), (37, 30, 97)], [(36, 169, 248), (28, 174, 234), (5, 173, 238), (0, 164, 244), (2, 149, 226)]],
"酞菁绿": [[(51, 58, 76), (37, 51, 64), (27, 43, 59), (29, 40, 62), (31, 46, 65)], [(91, 255, 200), (54, 226, 186), (75, 213, 180), (31, 206, 163), (31, 202, 158)]],
"炭黑": [[(35, 34, 39), (43, 42, 48), (44, 44, 46), (37, 37, 45), (43, 35, 48)], [(61, 57, 58), (42, 35, 42), (26, 24, 25), (24, 15, 18), (2, 2, 2)]],
"氧化铁黑TP370B": [[(73, 60, 54), (67, 56, 52), (66, 54, 54), (72, 56, 56), (62, 51, 45)], [(126, 77, 62), (93, 59, 49), (101, 59, 43), (101, 53, 15), (68, 19, 4)]],
"氧化铁黑335200": [[(43, 40, 49), (44, 39, 45), (42, 38, 37), (38, 39, 43), (42, 39, 48)], [(69, 66, 73), (72, 69, 64), (63, 55, 66), (42, 31, 37), (28, 13, 20)]],
"氧化铁黑335201": [[(53, 47, 51), (39, 37, 38), (53, 47, 49), (54, 50, 49), (49, 47, 50)], [(50, 41, 46), (56, 46, 45), (23, 12, 10), (26, 14, 18), (1, 1, 1)]],
"氧化铁红4125": [[(187, 76, 59), (186, 74, 60), (190, 70, 63), (195, 79, 66), (208, 78, 62)], [(219, 66, 48), (230, 68, 47), (226, 55, 27), (206, 31, 4), (197, 11, 0)]],
"氧化铁红": [[(200, 76, 64), (202, 77, 71), (191, 73, 63), (197, 78, 70), (209, 75, 64)], [(231, 70, 62), (241, 67, 40), (204, 20, 8), (215, 22, 4), (219, 24, 4)]],
"氧化铁黄": [[(181, 54, 99), (185, 150, 92), (175, 146, 88), (208, 162, 100), (216, 166, 97)], [(255, 194, 81), (250, 179, 73), (240, 176, 53), (255, 176, 39), (252, 156, 33)]],
"永固紫": [[(47, 41, 43), (44, 38, 48), (44, 41, 48), (37, 33, 34), (45, 45, 45)], [(92, 62, 196), (89, 57, 200), (44, 5, 186), (24, 1, 154), (15, 3, 139)]]}

pigments_cmyk = {}
for pigment, backgrounds in pigments_rgb.items():
    pigments_cmyk[pigment] = [ [rgb_to_cmyk(rgb) for rgb in background] for background in backgrounds ]
easy_cmyk = {}
for pigment, backgrounds in pigments_cmyk.items():
    easy_cmyk[pigment] = []
    easy_cmyk[pigment].append(backgrounds[0][4])
    easy_cmyk[pigment].append(backgrounds[1][4])
print(easy_cmyk)
def mix_cmyk(cmyk_values, proportions):
    """根据给定的比例混合CMYK颜色"""
    mixed = np.dot(proportions, cmyk_values)
    return mixed / np.sum(proportions)

def objective(proportions, target_cmyk, pigment_cmyk_values):
    """优化目标函数：计算混合CMYK颜色与目标颜色的欧氏距离"""
    mixed_cmyk = mix_cmyk(pigment_cmyk_values, proportions)
    return np.linalg.norm(mixed_cmyk - target_cmyk)

def find_mix_proportions(target_cmyk, pigment_cmyk_values):
    """找到最佳的色粉混合比例"""
    # 初始比例
    initial_proportions = np.ones(len(pigment_cmyk_values)) / len(pigment_cmyk_values)#疑问~~~~~~~~~初始化比例为什么要这样设置[0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625]
    # 比例约束：每种色粉比例在0和1之间，总和为1
    bounds = [(0, 1) for _ in range(len(pigment_cmyk_values))]
    print("bounds:",bounds)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # 使用最小化函数求解
    result = minimize(objective, initial_proportions, args=(target_cmyk, pigment_cmyk_values),
                      method='SLSQP', bounds=bounds, constraints=cons)
    if result.success:
        return result.x
    else:
        raise Exception("Optimization failed")

colors = []
for white_cmyk, value in easy_cmyk.items():
    colors.append(value[1])
# print("假设取白色底色的color：",colors)
# 示例：目标CMYK颜色和色粉的CMYK值
target_cmyk = np.array([20, 40, 60, 10])
pigment_cmyk_values = np.array(
    colors
)

# 找到最佳的色粉混合比例
best_proportions = find_mix_proportions(target_cmyk, pigment_cmyk_values)
sum=sum(best_proportions)
print("sum:",sum)
print("Best mix proportions:", best_proportions)

