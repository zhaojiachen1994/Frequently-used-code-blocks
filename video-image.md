<details> 
    <summary><strong>   遍历文件夹下所有视频、图像数据   </strong></summary>

```python
video_path = f"{path}/raw_videos"
files = [x for x in os.listdir(video_path) if x.endswith(".MP4")]
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   python 调用ffmpeg 压缩视频分辨率   </strong></summary>
    scale=-1:480, scale=640:480, scale=width:hight
    
```python
video_path = f"{path}/raw_videos"
files = [x for x in os.listdir(video_path) if x.endswith(".MP4")]
for file in files:
    outfile = f"{file[:-4]}_640{file[-4:]}"
    compress = f"ffmpeg -i {raw_video_path}/{file} -strict -2 -vf scale=-1:480 {out_video_path}/{outfile}"
    isRun = os.system(compress)
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   调用moviepy提取音频   </strong></summary>
    
```python
audio_path = f"{path}/audio"
os.makedirs(audio_path, exist_ok=True)
videofiles = [x for x in os.listdir(f"{path}/640_videos") if x.endswith(".MP4")]
for v in videofiles:
    audio_clip = AudioFileClip(f"{path}/640_videos/{v}")
    audio_clip.write_audiofile(f"{audio_path}/{v[:-4]}.wav")
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   按时间提取视频片段   </strong></summary>
    
```python
def extract_frag(path, p_start, t):
    """
        path: 
        p_start: the start time point
        t: the time length
    """
    video_path = f"{path}/sync_640_videos"
    files = [x for x in os.listdir(video_path) if x.endswith(".MP4")]
    target_path = f"{path}/ball_640_videos"

    fps = 60.0
    width = 640  # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = 480  # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    p_start = fps * p_start
    t = int(fps * t)

    for i, file in enumerate(files):
        print(f"extracting {file}")
        cap = cv2.VideoCapture(f"{video_path}/{file}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, p_start)
        writer = VideoWriter(f"{target_path}/{file}", resolution=(width, height), fps=fps)
        for j in range(t):
            _, img = cap.read()
            writer.write(img[:, :, ::-1])
        writer.close()
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   视频文件批量重命名   </strong></summary>
    1_abc.mp4 -> 01_abc.mp4
    
```python
def rename_videos(path):
    video_path = f"{path}/sync_videos"
    videos = [x for x in os.listdir(video_path) if x.endswith(".MP4")]
    for video in videos:
        ind = video.split('_', 1)
        ind[0] = f"{int(ind[0]):02d}"
        tar = "_".join(ind)
        os.rename(f"{video_path}/{video}", f"{video_path}/{tar}")
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   提取同步帧   </strong></summary>
    
```python
def extract_sync_images(path, i_frame=7205):
    video_path = f"{path}/sync_640_videos"
    videos = sorted([x for x in os.listdir(video_path) if x.endswith(".MP4")])
    for i, video in enumerate(videos):
        ic(video)
        cap = cv2.VideoCapture(f"{video_path}/{video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        ret, frame = cap.read()
        cv2.imwrite(f"{path}/temp_640/view{i + 1}_frame{i_frame}.jpg", frame)
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   将图片拼接成大图片   </strong></summary>
    images size: [x, y] = [height, width] = [480, 640], big image shape: [3, 8] 3 rows, 8 columns 
    
```python
video_path = f"{path}/sync_640_videos"
videos = sorted([x for x in os.listdir(video_path) if x.endswith(".MP4")])
images = []
for i, video in enumerate(videos):
    cap = cv2.VideoCapture(f"{video_path}/{video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    ret, frame = cap.read()
    images.append(frame)
big_image = np.zeros((1440, 7680, 3), dtype=np.uint8)  # Resizing the big image with 480*3=1440 and 640*8=7680
for i in range(3):
    for j in range(8):
        x = i * 480
        y = j * 640
        big_image[x:x + 480, y:y + 640, :] = images[i * 8 + j]
cv2.imwrite(f"{path}/temp_640/allview_frame{i_frame}.jpg", big_image)
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   Resize the image   </strong></summary>
    
```python
import cv2
 
img = cv2.imread('/home/img/python.png', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   计算视频时长   </strong></summary>
    
```python
import cv2
cap = cv2.VideoCapture(video_file)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
seconds = round(frames / fps)
video_time = datetime.timedelta(seconds=seconds)
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   按照时间间隔提取视频帧   </strong></summary>
    
```python
c = 0
cap = cv2.VideoCapture(video_path)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
width, height = 640, 480
rval = cap.isOpened()
while rval:
    rval, frame = cap.read()
    if rval and c % inter == 0:
        frame = cv2.resize(frame, [width, height], interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{target_path}/{c:04d}.jpg", frame)
    c = c + 1
cap.release()
```
</details>

