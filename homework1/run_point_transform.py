import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换
#定义权重函数
def wei(p, v, a=1):
    v = np.linalg.norm(p-v)
    v = v ** (2*a)
    return 1/v

#定义一个二维向量的垂直向量
def orth(p):
    q = np.array([-p[1], p[0]])
    return q


def point_guided_deformation(image, target_pts, source_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    #构建网格点
    target_pts = target_pts[:, ::-1]
    source_pts = source_pts[:, ::-1]
    h, w = image.shape[0:2]
    grid_h = np.arange(0, h, 10)
    grid_w = np.arange(0, w, 10)
    if grid_h[-1] != h - 1:
        grid_h = np.append(grid_h, h-1)
    if grid_w[-1] != w - 1:
        grid_w = np.append(grid_w, w-1)
    num_h, num_w = grid_h.size, grid_w.size
    #控制点数量
    n=len(target_pts)

    #预计算
    W, V = {}, {}
    for i in range(num_h):
        for j in range(num_w):
            V[(i,j)] = np.array([grid_h[i], grid_w[j]])#格点坐标
            for k in range(n):
                W[(i,j,k)] = wei(np.array(source_pts[k]), np.array([grid_h[i], grid_w[j]]))#权重大小

    P, Q = {}, {}
    for k in range(n):
        P[(k)] = np.array(source_pts[k]) #控制点
        Q[(k)] = np.array(target_pts[k]) #目标点


    #预计算
    P_star, Q_star, P_hat, Q_hat = {}, {}, {}, {}
    for i in range(num_h):
        for j in range(num_w):
            a, p_star, q_star = 0, np.zeros(2), np.zeros(2)
            for k in range(n):
                a = a + W[(i,j,k)]
                p_star = p_star + W[(i,j,k)]*P[(k)]
                q_star = q_star + W[(i,j,k)]*Q[(k)]
            P_star[(i,j)] = p_star / a
            Q_star[(i,j)] = q_star / a
            for k in range(n):
                P_hat[(i, j, k)] = P[(k)] - P_star[(i, j)]
                Q_hat[(i, j, k)] = Q[(k)] - Q_star[(i, j)]

    #print('W : ', W)
    #print('V : ', V)
    #print('P : ', P)
    #print('Q : ', Q)
    # print('P_star : ', P_star)
    # print('P_hat : ', P_hat)

    #预计算
    A = {}
    for i in range(num_h):
        for j in range(num_w):
            for k in range(n):
                A[(i, j, k)] = W[(i, j, k)] * np.dot(np.vstack(([[P_hat[(i, j, k)]], [-orth(P_hat[(i, j ,k)])]])), \
                                                     np.vstack(([[V[(i, j)] - P_star[i, j]], [-orth(V[(i,j)] - P_star[(i, j)])]])).T )

    #print('A : ', A)

    F_r, F_r_norm = {}, {}
    V_target = {} #改变后的格点坐标
    for i in range(num_h):
        for j in range(num_w):
            a = np.zeros(2)
            for k in range(n):
                a = a + np.dot(Q_hat[(i, j, k)], A[(i, j, k)])
            #print(a)
            F_r[(i, j)] = a
            F_r_norm[(i, j)] = np.linalg.norm(a)
            V_target[(i, j)] = np.linalg.norm(V[(i, j)] - P_star[(i, j)]) * F_r[(i, j)] / F_r_norm[(i, j)] + Q_star[(i, j)]

    #print(V_target)
    #根据得到的网格点变换后的坐标，用双线性插值计算网格
    warped_image = np.array(image) * 0 #初始化

    #定义一个正方形的双线性插值
    def binter(p00, p10, p01, q00, q10, q01, q11, image, warped_image):
        a = p10[0] - p00[0]
        b = p01[1] - p00[1]
        for i in range(int(p00[0]), int(p10[0])):
            for j in range(int(p00[1]), int(p01[1])):
                a0, a1, b0, b1 = (i - p00[0]) / a, (p10[0] - i) / a, (j - p00[1]) / b, (p01[1] - j) / b
                v = b1 * a0 * q10 + b0 * a0 * q11 + b1 * a1 *q00 + b0 * a1 * q01
                x, y = v[0], v[1]
                if x >= 0 and x < image.shape[0] and y >= 0 and y < image.shape[1]:
                   warped_image[i, j] = image[int(x), int(y)]
        return warped_image
    
    for i in range(num_h - 1):
        for j in range(num_w - 1):
            warped_image = binter(V[(i, j)], V[(i + 1, j)], V[(i, j + 1)], V_target[(i, j)], V_target[(i + 1, j)], \
                   V_target[(i, j + 1)], V_target[(i + 1, j + 1)], image, warped_image)

    ### FILL: 基于MLS or RBF 实现 image warping


    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
