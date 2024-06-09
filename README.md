<h1 align="center"> AI_CUP MTMC Tracking</h1>

基於https://github.com/regob/vehicle_mtmc 去修改 使其能跟蹤低幀率(fps:1)的資料集，由與此作者提供的輸入只能是影片檔，所以要使用我的轉檔.py 把test相片資料集做轉換。

安裝:

pip install cython "numpy>=1.18.5,<1.23.0"

安裝:

pip install -r requirements.txt

之後執行缺甚麼就安裝甚麼即可
  

使用方式 

python mot\run_tracker.py --config examples\mot_1.yaml


mot_1.yaml檔案形式為

OUTPUT_DIR: "output/resnet50_ibn/1/0903_125957_131610"

MOT:

  VIDEO: r"D:\vehicle_mtmc\datasets\race\0903_125957_131610.mp4"
  
  REID_MODEL_OPTS: r"D:\vehicle_mtmc\reid\vehicle_reid\model\resnet50_ibn11\opts.yaml"
  
  REID_MODEL_CKPT: r"D:\vehicle_mtmc\reid\vehicle_reid\model\resnet50_ibn11\net_44.pth"
  
  DETECTOR: "yolov5x6"      //不需動
  
  TRACKER: "bytetrack_iou"  //只能用這個我改過的跟蹤模型處裡低幀率資料 
  
  SHOW: false               //運行時顯示跟蹤效果
  
  VIDEO_OUTPUT: true        //要不要保存相關結果

  
  選定要跟蹤的影片及藥使用的ReID model 即可


也可去直接下載https://github.com/regob/vehicle_mtmc

將mot內的tracker.py更換成我的tracker.py。

將byte_track內的byte_tracker.py及matching.py更換成我的。

將config/defaults.py內的

# minimum number of bounding boxes per track
C.MOT.MIN_FRAMES = 10 改成為1

即可運行低幀率追蹤
  
