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
