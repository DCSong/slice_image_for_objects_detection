# slice_image_for_objects_detection
Slicing high resolution image with overlap for objects detection.  
  
  
# cut_image.py  
#### 适用于目标检测等问题中的高分辨率图片切片，可自定义切割大小和重叠尺寸（已考虑目标越界等情况）    
  
### 参数设置      
##### 图片、标签根目录和切割图片、标签保存目录（标签都为txt格式）  
image_root_path = './train_imgs'  
label_root_path = './/train-txt-labels'  
image_save_path = './images'  
label_save_path = './labels'    
    
##### 图片切片大小和重叠尺寸   
cut_h, cut_w = 640, 640  
overlap = 640 // 5    
    
##### 是否在切片时滤去完全没有目标的原图（背景类）    
dropBackground = False    
  
##### 是否在切片时滤去没有目标的切片图  
dropSubBackground = True    
      
      
# detect_cut_images.py  
##### 代码本为[瓷砖表面瑕疵检测](https://tianchi.aliyun.com/competition/entrance/531846/introduction)所写，但稍加修改后也适用于其他情况    
##### 该推断代码基于YOLOv5的推断代码修改而来，在推断时对测试图片使用与上述切片代码相同的逻辑进行切片    
##### 注：该推断代码尚未经过严格测试，可能存在未知BUG    
   
