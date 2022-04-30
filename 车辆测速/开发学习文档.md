## 一、问题定义

**单目测速：**多目标追踪 + 单目测距 + 速度公式



## 二、算法设计

### 2.1 策略一（已实现）

#### 2.1.1 原理

目标检测并追踪视频中车辆的车尾（假定摄像头安装在单行道上），根据连续两帧的检测框计算得到像素距离。然后通过预先计算的 ppm (pixel per meter) ——道路不同，其值不同——得到实际距离；然后利用FPS计算得到速度，并显示在视频中。

- 相关技术
  - CascadeClassifier目标检测器（训练车尾数据集）
  - dlib中的correlation_tracker跟踪器

#### 2.1.2 算法流程图



<img src="D:\学期计划\2021_暑假\车辆测速\img\流程图.PNG" alt="流程图" style="zoom: 80%;" />

- 变量说明

  

  | 变量名        | 变量类型 | 变量含义                                                     |
  | ------------- | -------- | ------------------------------------------------------------ |
  | frameCounter  | int      | 当前所在帧                                                   |
  | currentCarID  | int      | 新创建车辆跟踪器的ID；可统计共创建跟踪器个数                 |
  | carTracker    | dict     | 每个carID的Tracker;例如{0：<_dlib_pybind11.correlation_tracker object at 0x000002562877FB30>} |
  | carLocation1  | dict     | 上一帧中每个carID的位置[x,y,w,h]；例如{0：[x,y,w,h]}         |
  | carLocation2  | dict     | 下一帧中每个carID的位置[x,y,w,h]；例如{0：[x,y,w,h]}         |
  | speed         | array    | 每个carID当前的速度（不会记录以前的速度；会记录删除的carID的最后速度） |
  | carIDtoDelete | array    | 跟踪器update后置信度小于7的carID(会被carTracker，carLocation1，carLocation2 删除) |

#### 2.1.3 结果

```python
python speed_check.py
```

![output](D:\学期计划\2021_暑假\车辆测速\img\output.gif)

#### 2.1.4 代码

```python
import cv2
import dlib
import time
import threading
import math

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('cars.mp4')

WIDTH = 1280
HEIGHT = 720


def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	# ppm = location2[2] / carWidht
	ppm = 8.8
	d_meters = d_pixels / ppm
	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	fps = 18
	speed = d_meters * fps * 3.6
	return speed
	

def trackMultipleObjects():
	rectangleColor = (0, 255, 0)
	frameCounter = 0
	currentCarID = 0
	fps = 0
	
	carTracker = {}
	carNumbers = {}
	carLocation1 = {}
	carLocation2 = {}
	speed = [None] * 1000
	
	# Write output to video file
	# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH,HEIGHT))


	while True:
		start_time = time.time()
		rc, image = video.read()
		# 检查是否到达视频文件的末尾
		if type(image) == type(None):
			break
		# 转换帧的大小，以加快处理速度
		image = cv2.resize(image, (WIDTH, HEIGHT))
		resultImage = image.copy()
		
		frameCounter = frameCounter + 1
		
		carIDtoDelete = []

		for carID in carTracker.keys():
			trackingQuality = carTracker[carID].update(image)
			
			if trackingQuality < 7:
				carIDtoDelete.append(carID)
				
		for carID in carIDtoDelete:
			print ('Removing carID ' + str(carID) + ' from list of trackers.')
			print ('Removing carID ' + str(carID) + ' previous location.')
			print ('Removing carID ' + str(carID) + ' current location.')
			carTracker.pop(carID, None)
			carLocation1.pop(carID, None)
			carLocation2.pop(carID, None)
		# frameCounter为10的倍数时执行
		# 每10帧执行一次目标检测
		if not (frameCounter % 10):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# 检测视频中的车辆，并用vector保存车辆的坐标、大小（用矩形表示）
			# x,y表示第n帧第i个运动目标外接矩形的中心横坐标和纵坐标位置，该坐标可以大致描述车辆目标所在的位置。
			# w,h表示第n帧第i个运动目标外接矩形的宽度和长度，可以描述车辆目标的大小
			cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
			
			for (_x, _y, _w, _h) in cars:
				x = int(_x)
				y = int(_y)
				w = int(_w)
				h = int(_h)
			
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None
			
				for carID in carTracker.keys():
					trackedPosition = carTracker[carID].get_position()
					
					t_x = int(trackedPosition.left())
					t_y = int(trackedPosition.top())
					t_w = int(trackedPosition.width())
					t_h = int(trackedPosition.height())
					
					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h

					# 找到匹配的ID
					# 改进方法：各种距离
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
				
				if matchCarID is None:
					print ('Creating new tracker ' + str(currentCarID))
					
					tracker = dlib.correlation_tracker()  # 创建跟踪类
					tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h)) # 设置图片中要跟踪物体的框
					
					carTracker[currentCarID] = tracker
					carLocation1[currentCarID] = [x, y, w, h]

					currentCarID = currentCarID + 1
		
		#cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)


		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].get_position() # 得到跟踪到的目标在当前帧的位置
					
			t_x = int(trackedPosition.left())
			t_y = int(trackedPosition.top())
			t_w = int(trackedPosition.width())
			t_h = int(trackedPosition.height())
			
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
			
			# speed estimation
			carLocation2[carID] = [t_x, t_y, t_w, t_h]  # 追踪出的位置
		
		end_time = time.time()
		
		if not (end_time == start_time):
			fps = 1.0/(end_time - start_time)
		
		#cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


		for i in carLocation1.keys():	
			if frameCounter % 1 == 0:
				[x1, y1, w1, h1] = carLocation1[i]
				[x2, y2, w2, h2] = carLocation2[i]
		
				# print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
				carLocation1[i] = [x2, y2, w2, h2]
		
				# print 'new previous location: ' + str(carLocation1[i])
				if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
					if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:  # 太近或太远则不用检测
						speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

					#if y1 > 275 and y1 < 285:
					if speed[i] != None and y1 >= 180:
						cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
					
					#print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

					#else:
					#	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

						#print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
		cv2.imshow('result', resultImage)
		# Write the frame into the file 'output.avi'
		#out.write(resultImage)


		if cv2.waitKey(33) == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	trackMultipleObjects()

```



### 2.2 策略二（未实现）—— 三角形相似

- **情况1：平行行驶**

  ![6](D:\学期计划\2021_暑假\车辆测速\img\6.PNG)

  ![7](D:\学期计划\2021_暑假\车辆测速\img\7.PNG)

- **情况2.1：垂直行驶——投影平面与运动方向垂直**

  ![8](D:\学期计划\2021_暑假\车辆测速\img\8.PNG)

- **情况2.2：垂直行驶——投影平面与运动方向有夹角**

  ![9](D:\学期计划\2021_暑假\车辆测速\img\9.PNG)

### 2.3 策略三（未实现）—— 函数拟合

![10](D:\学期计划\2021_暑假\车辆测速\img\10.PNG)

![11](D:\学期计划\2021_暑假\车辆测速\img\11.PNG)

### 2.4 策略四（未实现）—— 相对对极几何

![12](D:\学期计划\2021_暑假\车辆测速\img\12.PNG)

