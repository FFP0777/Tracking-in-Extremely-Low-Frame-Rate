import cv2
import os

def images_to_video(image_folder, video_name, fps):
    # 獲取圖片列表
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # 確保按名稱順序排序

    # 獲取圖片尺寸
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # 初始化視頻寫入器
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # 釋放視頻寫入器
    video.release()

if __name__ == "__main__":
    image_folder = r"D:\vehicle_mtmc\datasets\0902_150000_15190013"  # 替換為你的圖片文件夾路徑
    video_name = "output_video.mp4"  # 輸出視頻文件名
    fps = 1  # 設置視頻的幀率

    images_to_video(image_folder, video_name, fps)
