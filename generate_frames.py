import cv2
import os

# 递归方法访问文件并按照文件夹及文件名称新建对应的新文件夹
# 转换avi文件为jpg图，当前程序缺点：视频切割的图片有重复图片存在。

def close_VideoCapture(cap):
    cap.release()

def load_avi_file_sid_cid_time(video_filename):
    try:

        cap = cv2.VideoCapture(video_filename)
        if cap.isOpened():
            return cap
        else:
            print("load avi file failed!\n")
            return None
    except:
        return None
def search_avi_and_get_frame(path1,savepath):

    for filename in os.listdir(path1):
        if os.path.isdir(os.path.join(path1,filename)):
            path1_ = os.path.join(path1,filename)
            savepath_ = os.path.join(savepath,filename)
            search_avi_and_get_frame(path1_,savepath_)
            print('is dir',filename)
        elif filename[0] == '.':
            print('is . continue')
            continue
        else:
            print(filename)
            firstname,lastname =filename.split('.')[0],filename.split('.')[1]
            # print(firstname,lastname)
            if lastname == 'avi':
                # try:
                    # new_dir = savepath+'\\' + newmakedirname
                    newmakedir = os.path.join(savepath,firstname)
                    # newmakedir = os.path.join(new_dir, firstname)
                    if not os.path.exists(newmakedir):
                        os.makedirs(newmakedir)
                    cap = load_avi_file_sid_cid_time(os.path.join(path1,filename))
                    startframe = 1
                    skipframe = 1
                    while cap:
                        # 以帧率的形式进行抽帧
                        cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
                        ret, frame = cap.read()
                        if ret:
                            cv2.imwrite(os.path.join(newmakedir,'{}.jpg'.format(startframe)), frame)
                            startframe += skipframe
                        else:
                            close_VideoCapture(cap)
                            break
                # except:
                #     print('errorfile',filename)

if __name__ == '__main__':
    # avi文件存放的根目录
    path_ = '300VW_Dataset_2015_12_14/'
    #需要存放jpg图的根目录
    savepath_ = '300VW_Out/'
    for dir in os.listdir(path_):
        if os.path.isdir(os.path.join(path_,dir)):
            if not os.path.exists(os.path.join(savepath_,dir)):
                os.makedirs(os.path.join(savepath_,dir))
            search_avi_and_get_frame(os.path.join(path_,dir),os.path.join(savepath_,dir))



