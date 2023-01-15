# 图片识别
from aip import AipOcr
# 时间模块
import time
# 网页获取
import requests
# 操作系统接口模块
import os

# 获取文件夹中所有图片


def get_image():
    images = []  # 存储文件夹内所有文件的路径（包括子目录内的文件）
    for root, dirs, files in os.walk(image_path):
        path = [os.path.join(root, name) for name in files]
        images.extend(path)
    return images


def Image_Excel(APP_ID, API_KEY, SECRET_KEY):
   #  调用百度AI接口
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    # 循环遍历文件家中图片
    images = get_image()
    for i, image in enumerate(images):
        # 以二进制方式打开图片
        img_open = open(image, 'rb')
        # 读取图片
        img_read = img_open.read()
        # 调用表格识别模块识别图片
        table = client.tableRecognitionAsync(img_read)
        # 获取请求ID
        request_id = table['result'][0]['request_id']
        # 获取表格处理结果
        result = client.getTableRecognitionResult(request_id)
        # 处理状态是“已完成”，获取下载地址
        while result['result']['ret_msg'] != '已完成':
            time.sleep(2)  # 暂停2秒再刷新
            result = client.getTableRecognitionResult(request_id)
        download_url = result['result']['result_data']
        print(download_url)
        # 获取表格数据
        excel_data = requests.get(download_url)
        # 根据图片名字命名表格名称
        xlsx_name = 'photo_'+str(i+1) + ".xls"
        # xlsx_name = image.split(".")[0] + ".xls"

        # 新建excel文件
        xlsx = open(xlsx_name, 'wb')
        # 将数据写入excel文件并保存
        xlsx.write(excel_data.content)


if __name__ == '__main__':
    image_path = r'C:\Users\YUyixiong\Desktop\1'
    APP_ID = "23044761"
    API_KEY = "zCLERFdytz8bn38mPrM9ceWs"
    SECRET_KEY = "ESFdhhG7dO9QkyeFvSOQDtbK2tsX28Il"
    Image_Excel(APP_ID, API_KEY, SECRET_KEY)
