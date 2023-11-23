from PIL import Image

def resize_and_save_image(input_path, output_path, new_size=(84, 84)):
    try:
        # 打开图像文件
        with Image.open(input_path) as img:
            # 调整图像大小
            resized_img = img.resize(new_size)
            # 保存调整大小后的图像
            resized_img.save(output_path)
            print(f"图像已成功调整并保存到 {output_path}")
    except Exception as e:
        print(f"出现错误: {e}")

# 示例使用方法
input_image_path = '/home/dsz/Documents/cjq/few-shot-master/data/cifar/test/bed/bed_s_000024.png'  # 替换为你的输入图像路径
output_image_path = 'bed.png'  # 替换为你的输出图像路径
resize_and_save_image(input_image_path, output_image_path)
