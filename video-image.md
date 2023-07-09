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



