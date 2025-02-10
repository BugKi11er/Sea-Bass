import datetime
import os
import random
import string
import threading
import time
import cv2
import yaml
from PIL import Image
from AliyunOss import AliyunOss
from RepVGG import RepVGG
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import imageio as iio
from myutils.CSocket import CSocket
from myutils.handleData import del_file

with open('cfg/config.yml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 使用配置中的参数
db_username = config['db']['username']
db_password = config['db']['password']
db_host = config['db']['host']
db_port = config['db']['port']
db_name = config['db']['name']
db_charset = config['db']['charset']

num_classes = config['model']['num_classes']
weights_path = config['model']['weights_path']
json_path = config['model']['json_path']
use_se = config['model']['use_se']
att_type = config['model']['att_type']
device = config['model']['device']

cap_url = config['cap']['url']

# 获取阿里云OSS的配置信息
access_key_id = config['aliyun_oss']['access_key_id']
access_key_secret = config['aliyun_oss']['access_key_secret']
bucket_name = config['aliyun_oss']['bucket_name']
endpoint = config['aliyun_oss']['endpoint']

# 从配置文件中获取连接参数
host = config['connection']['host']
port = config['connection']['port']

# 是否删除video下的文件
is_Delete = config['file_management']['is_Delete']

# 构造数据库连接字符串
db_connection_string = f"mssql+pymssql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}?charset={db_charset}"

# # 创建数据库引擎
# engine = create_engine(db_connection_string)

# # 创建会话类
# Session = sessionmaker(bind=engine)
# session = Session()

# 定义映射类
# Base = declarative_base()
#
#
# class FeedingInfo(Base):
#     __tablename__ = 'Feeding_info'
#
#     id = Column(Integer, primary_key=True)
#     create_datatime = Column(DateTime)
#     feeding_count = Column(String(50))
#     no_eating_count = Column(String(50))
#     pic_path = Column(String(100))
#     is_feeding = Column(String(50))


is_listener = False  # 全局变量，用于标记是否收到指定消息


def generate_file_name(outfile_name):
    timestamp = int(time.time())
    random_str = ''.join(random.choice(string.ascii_letters) for i in range(6))
    return f"{outfile_name}/video_{timestamp}_{random_str}.mp4"

# 定义全局变量
cap_flag = False

# 定义全局变量
start_time = None  # 清空计时器起始时间

def receive_thread(client):
    """
    用于多线程；死循环，监听 C# 发送的消息
    """
    global is_listener  # 使用全局变量
    global start_time  # 使用全局变量
    global start_time   # 定义计时器起始时间
    while True:
        global cap_flag
        try:
            response = client.receive()
            # print(response)
            if b"Ready to Feeding" == response:
                if cap_flag:
                    client.send("ok")
                    is_listener = True  # 如果收到特定消息，设置全局变量
                    start_time = time.time()  # 更新计时器起始时间
                    print("接受到投饵机的信号开始检测")
                    print("计时器起始时间：", start_time)
                else:
                    print("摄像头未准备好,进行重试")
                    client.send("False")
            if b"Feeding Finished" == response:
                is_listener = False  # 如果收到特定消息，设置全局变量
                start_time = None  # 清空计时器起始时间
                print("接受到投饵机的信号停止检测")

        except Exception as e:
            print("socket err: ", e)
            break



if __name__ == '__main__':

    RepVGG = RepVGG(num_classes=num_classes, weights_path=weights_path,
                    json_path=json_path, use_se=use_se, att_type=att_type, device=device)

    url = cap_url

    cap = cv2.VideoCapture(url)

    # 设置视频宽高和帧率
    # 获取视频宽度
    frame_width = ((1920 - 437) // 2) // 16 * 16

    # 获取视频高度
    frame_height = ((1037) // 2) // 16 * 16

    # 获取视频宽度
    frame_detect_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2

    # 获取视频高度
    frame_detect_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

    read_frame = 0

    frame = None  # 全局变量用来存储图片

    objectsFish = []  # 保存实体数据

    SAVE_VIDEO_INTERVAL = 15  # 批量插入数据库的个数

    fish_status_dict = {"0": 0, "1": 0}  # 计数器

    # 创建视频编码器对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = 25.0

    # 设置输出文件夹根路径
    # outfile_name = "D:/vscodeProject/fish/renren-fast-vue/static/video"
    outfile_name = "video"

    if not os.path.exists(outfile_name):
        os.mkdir(outfile_name)
    # 创建VideoWriter对象，用于将帧写入视频文件

    file_name = generate_file_name(outfile_name)

    out = iio.get_writer(file_name, format='ffmpeg', mode='I', fps=fps, codec='libx264', pixelformat='yuv420p')

    # 定义计数器
    count = 0

    # 创建客户端套接字
    client = CSocket(host, port)

    # 创建一个新线程来接收消息
    t = threading.Thread(target=receive_thread, args=(client,))
    t.start()

    # 创建AliyunOss对象
    aliyunOss = AliyunOss(access_key_id=access_key_id, access_key_secret=access_key_secret,
                          bucket_name=bucket_name, endpoint=endpoint)

    flag = False

    # 初始化帧计数器
    frame_count = 0

    while True:

        ret, frame = cap.read()
        cap_flag = ret
        # 如果读到的帧数不为空，那么就继续读取，如果为空，就退出
        if not ret:
            cap = cv2.VideoCapture(url)
            print("丢失帧")
            continue
        # 每隔三帧进行检测
        if frame_count % 3 == 0:
            if is_listener:
                # 如果超过4分钟还没有收到 Feeding Finished 消息，设置 is_listener 为 False
                if start_time and time.time() - start_time > 6 * 60:
                    is_listener = False
                    start_time = None
                    print("检测超时，停止检测当前时间")
                    print("is_listener={}".format(is_listener))
                flag = True
                frame_detect = cv2.resize(frame, (frame_detect_width, frame_detect_height))
                frame = frame[0:1037, 437:, ]
                frame = cv2.resize(frame, (frame_width, frame_height))
                # print('开始检测')
                # print(current_frame_time)
                # frame_detctct = frame[35:522, 215:, ]

                img = Image.fromarray(cv2.cvtColor(frame_detect, cv2.COLOR_BGRA2RGBA))
                img = img.convert('RGB')

                # img = img[]

                predict_cla, predict_prob = RepVGG.infer(img)

                predict_cla = str(predict_cla)

                if predict_cla in fish_status_dict:
                    current_value = fish_status_dict.get(predict_cla)
                    current_value = current_value + 1
                    fish_status_dict[predict_cla] = current_value
                # print(fish_status_dict)
                # print("检测个数{}".format(len(objectsFish)))
                # start_time = time.time()
                frame = RepVGG.plot_one_img(frame, predict_cla, predict_prob)
                # end_time = time.time()
                # print(end_time-start_time)
                out.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                count = count + 1

                if (count + 1) % (25 * 30) == 0:  # 每25*60帧保存一次
                    file_video_path = file_name
                    # print(file_video_path)
                    count = 0
                    out.close()
                    aliyunOss_file_video_path = aliyunOss.put_object_from_file(file_video_path, file_video_path)
                    if is_Delete:
                        del_file(file_name)
                    file_name = generate_file_name(outfile_name)
                    out = iio.get_writer(file_name, format='ffmpeg', mode='I', fps=fps, codec='libx264',
                                         pixelformat='yuv420p')
                    fish_status_ingestion = fish_status_dict.get("0")
                    fish_status_no_ingestion = fish_status_dict.get("1")
                    current_frame_time = datetime.datetime.now()
                    current_frame_time_str = current_frame_time.strftime('%Y-%m-%d %H:%M:%S.%f')
                    fish_status_ingestion = str(fish_status_ingestion)
                    fish_status_no_ingestion = str(fish_status_no_ingestion)
                    # 发送从c#端的数据
                    fish_status = fish_status_ingestion + "_" + fish_status_no_ingestion + "_" + current_frame_time_str
                    # 插阿里云数据库
                    # feedingInfo = FeedingInfo(create_datatime=current_frame_time,
                    #                           feeding_count=fish_status_ingestion,
                    #                           no_eating_count=fish_status_no_ingestion,
                    #                           pic_path=aliyunOss_file_video_path)
                    # # 插入本地数据库
                    # feedingInfo = FeedingInfo(create_datatime=current_frame_time,
                    #                           feeding_count=fish_status_ingestion,
                    #                           no_eating_count=fish_status_no_ingestion,
                    #                           pic_path=file_video_path)
                    # 插入数据库
                    # session.add(feedingInfo)
                    # session.commit()
                    # # 关闭会话
                    # session.close()
                    print("插入成功--->创建时间：{}, 摄食个数：{}, 非摄食个数：{}，图片路径：{}".format(current_frame_time, fish_status_ingestion, fish_status_no_ingestion, aliyunOss_file_video_path))
                    fish_status_dict = {"0": 0, "1": 0}  # 重置计数器
                    client.send(fish_status)
            else:
                if flag:
                    out.close()
                    if is_Delete:
                        del_file(file_name)
                    flag = False
                    # 创建VideoWriter对象，用于将帧写入视频文件
                    file_name = generate_file_name(outfile_name)
                    out = iio.get_writer(file_name, format='ffmpeg', mode='I', fps=fps, codec='libx264',
                                         pixelformat='yuv420p')
                    count = 0
        # 帧计数器加一
        frame_count += 1
        # 当帧计数器达到 3 时重置为 0
        if frame_count == 3:
            frame_count = 0
    # 完成后释放资源
    out.close()

    cap.release()

    cv2.destroyAllWindows()
