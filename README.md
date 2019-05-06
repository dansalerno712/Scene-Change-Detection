# Dynamic Scene Change Detection

## SceneDetector
```python
SceneDetector(self, input_path, output_dir)
```
A class to implement scene change detection based on this
paper, with some additions
https://www.researchgate.net/profile/Anastasios_Dimou/publication/249863763_SCENE_CHANGE_DETECTION_FOR_H264_USING_DYNAMIC_THRESHOLD_TECHNIQUES/links/00b4952d3b4ae69753000000/SCENE-CHANGE-DETECTION-FOR-H264-USING-DYNAMIC-THRESHOLD-TECHNIQUES.pdf

#### Attributes:  
    input_path (string): The path to a video file 

    output_dir (string): The directory to save output files  

### detect
```python
SceneDetector.detect(self, window_size=20, method='SAD', a=-1, b=2, c=2, s=0.02, k=20, display=False, output=False)
```
Detects scene changes in a video

#### Args:
    window_size (int, optional): How many frames should be kept in the sliding window array  

    method (str, optional): Which image similarity method should be used Options: SAD, SSD, MAD, CORR  

    a (float, optional): a coefficient from the paper  

    b (float, optional): b coefficient from the paper  

    c (float, optional): c coefficient from the paper 

    s (float, optional): s coefficient from the paper 

    k (int, optional): k coefficient from the paper AKA how many frames to use the decay threshold for after detecting a change

    display (bool, optional): Whether or not to display data while/after detecting. If true, this will show
    a) The current frame from the video
    b) The preceding frame and the frame that a change
    was detected on
    c) A graph plotting frame similarity and threshold data

    output (bool, optional): whether or not to output data.
    This will output a text file saying between which frames
    a scene change was found

