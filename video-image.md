<details> 
    <summary><strong>   遍历文件夹下所有视频、图像数据   </strong></summary>

```python
video_path = f"{path}/raw_videos"
files = [x for x in os.listdir(video_path) if x.endswith(".MP4")]
```
</details>
