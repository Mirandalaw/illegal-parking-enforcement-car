# illegal-parking-enforcement-
During two weeks, i want to train image(carNumber) if u want to train anything image u could do this way!!


1.status 
  Graphic RTX 3060
 
 : opencv 4.2.0 , CUDA 11.2 cuDNN 8.2.0 , yolov4 , Ubuntu 18.04


2. darnknet install
 -git clone https://github.com/AlexeyAB/darknet.git
 -Makefile(setting info) GPU = 1, CUDNN = 1, CUDNN_HALF = 0 , OPENCV = 1,LIBSO = 1 
 - darknet$ make
 - Want to test i should download yolov4.weights Download
 - https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
 - image : ./darknet detect cfg/yolov4.cfg yolov4.weights data/dog.jpg
 - video : ./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output 동영상.mp4
 - webcam : ./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -c 0
   *error : "cuDNN Error: CUDNN_STATUS_EXECUTION_FAILED" we should do "sudo reboot"
  
  ***important*** (custom data train)
  
  1. mkdir ws &&cd ws &&mkdir img
   
  2. u want to train data that u can save data-set
 
  3. image anotation tool ( YOLO_mark )
     - git clone https://github.com/AlexeyAB/Yolo_mark.git
     - cd Yolo_mark
     - camke .
     - make
     - gedit obj.names
     - cat obj.names (u can see : carNumber)
     - ./yolo_mark ~/ws/img/ ./train.txt ./obj.names


  4. darknet/cfg/yolov4.cfg copy and rename (yolo-obj.cfg)


     -in ws : mkdir backup
     
     
     -mkdir data
     
     
     -gedit data/obj.data -> classes = 1, train = data/train.txt, valid = data/train.txt, names = data/obj.names backup = backup/
     
     
     -copy : Yolo_mark/train.txt -> to ws/data
     
           : Yolo_mark/obj.names -> to ws/data
           
           
     -modify : yolo-obj.cfg -> batch = 64, subdivisions = 64 width,height =416
     
             : max_batches = 2000 (max_batches = class * classCount)
             
             : steps : (max_batches's 80%),(max_batches's 90%)
             
             : ctrl + f -> yolo u can find (1/3) classes =1 modify && filters = 18(filters=(classes + 5)*3)
             
             
             
     -download : yolov4's pretrain model  & ws/ (save here) 
       wget  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137 
       
       
 5. train start : ~/ws$./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137 


 7. complete train : ./darknet detector test data/obj.data yolo-obj.cfg backup/yolo-obj_final.weights carNumber.jpg 


 참고 :  https://webnautes.tistory.com/1482
