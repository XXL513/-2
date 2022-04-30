# Mater OpenCV

## Chapter 01 - Setting Up OpenCV

### Project Structure

```
# project structure

sampleproject/
|
|----.gitignore
|---- sampleproject.py
|---- LICENSE
|---- README.rst
|---- requirements.txt
|---- setup.py
|---- tests.py
```

*README(.rst or .md)* is used to register the main properties of the project. [template](https://github.com/dbader/readme-template)

- What your project does
- How to install it
- Example usage
- How to set up the dev environment
- How to ship a change
- Change log
- License and author info		

# First Python and OpenCV project

```
# project structure

helloopencv/
|
|---- images/
|
|----.gitignore
|---- helloopencv.py
|---- LICENSE
|---- README.rst
|---- requirements.txt
|---- setup.py
|---- helloopencvtests.py
```

- *setup.py*

```python
# setup.py
# https://github.com/pypa/sampleproject/blob/master/setup.py 有更多信息
from setuptools import setup

setup(
    name='helloopencv',
    version='0.1',
    py_modules=["helloopencv"],
    license='MIT',
    description='An example python opencv project',
    long_description=open('README.rst', encoding='utf-8').read(),
    install_requires=['numpy','opencv-contrib-python'],
    url='https://github.com/albertofernandez',
    author='Alberto Fernandez',
    author_email='fernandezvillan.alberto@gmail.com'
)
```

```
# install the project in our system

C:\...\helloopencv>python setup.py install
# 运行后目录结构变成
helloopencv/
|---- build/
|---- dist/
|---- helloopencv.egg-info/
|---- images/
|
|----.gitignore
|---- helloopencv.py
|---- LICENSE
|---- README.rst
|---- requirements.txt
|---- setup.py
|---- helloopencvtests.py
```

- 3种应用

```
# 第一种：import
C:\...\helloopencv>python
>>> import helloopencv
helloopencv.py is being imported into another module
>>> helloopencv.show_message()
'this function returns a message'

# 第二种：直接execute（此种不需要python setup.py install）
C:\...\helloopencv>python helloopencv.py
helloopencv.py is being run directly

## 这两种应用方式输出不一样，查看helloopencv.py即可

# 第三种：利用helloopencvtests.py进行unit testing
C:\...\helloopencv>py.test -s -v helloopencvtests.py

============ test session starts========================
platform win32 -- Python 3.7.4, pytest-5.2.1, py-1.8.0, pluggy-0.13.0 -- G:\anaconda\python.exe
cachedir: .pytest_cache
rootdir: D:\学期计划\2021_暑假\Mastering-OpenCV-4-with-Python-master\Mastering-OpenCV-4-with-Python-master\Chapter01\02-minimal-opencv-python-project\helloopencv
plugins: arraydiff-0.3, doctestplus-0.4.0, openfiles-0.4.0, remotedata-0.3.2
collecting ... hellopencv.py is being imported into another module
collected 4 items

helloopencvtests.py::test_show_message testing show_message
PASSED
helloopencvtests.py::test_load_image testing load_image
PASSED
helloopencvtests.py::test_write_image_to_disk testing write_image_to_disk
PASSED
helloopencvtests.py::test_convert_to_grayscale testing test_convert_to_grayscale
PASSED

=============== 4 passed in 0.13s ========================

```

-----------------------------------------------------2021/7/3日完成-----------------------------------------------------------------

## Chapter 02 - Image Basic in OpenCV

### The goal of computer vision

- A new representation(for example, a new image)
- A decision (for example, perform a concrete task)
- A new result (for example, correct classification of the image)
- Some useful information extraction (for example, object detection)

### The problems of computer vision

- Ambiguous images
- 外界影响 ： illumination, weather, reflections, and movements ......
-  Objects in the image may also be occluded by other objects 

 *You can also classify your test images in connection with the main difficulty they have to easily detect the weak points of your algorithm.* 

### Image-processing steps

*Three steps :*

1. Get the image to work with
2.  Process the image by applying image-processing techniques to achieve the required functionality 
   - Low-level process: takes an image as the input and then outputs another image, such as *Noise removal, Image sharpening, illumination normalization, perspective correction*
   - Mid-level process:  takes the preprocessed image to output some kind of representation of the image. 
   - High-level process:  takes this vector of numbers (usually called **attributes**) and outputs the final result, such as *face recognition, emotion recognition*.
3.  Show the result of the processing step 

### Image formulation

- grayscale  image
- color image
- black and white image

 *PPI = width(pixels)/width of image* *(inches)* = *height(pixels)/height of image* *(inches)* 

*坐标系*

<img src="D:\学期计划\2021_暑假\img\坐标系.png" alt="坐标系" style="zoom: 25%;" />

### Accessing and manipulating pixels in OpenCV with BGR images

*OpenCV loads the color images so that **the blue channel is the first, the green channel is the second, and the red channel is the third*** 

```python
# read an image
img = cv2.imread('logo.png')

# get the dimensions
# color image: a tuple of number of rows, colums, channels
# grayscale: a tuple of number of rows, colums
# ** it can be used to check if loaded image is color or grayscale image.
dimensions = img.shape
# Or
(h, w, c) = img.shape

# get the size of the image (height × width × channels)
total_number_of_elements= img.size

# get the datatype of the image
image_type = img.dtype

# display an image
cv2.imshow('original image', img)

# wait the value indicated by the argument(in milliseconds)
# cv2.waitKey(0): the programs waits indefinitely until a keyboard event is produced
cv2.waitKey(0)

# Get the value of the pixel (x=40, y=6):
(b, g, r) = img[6, 40]
# Or
b = img[6, 40, 0],g = img[6, 40, 1],r = img[6, 40, 2]

# modify the value of the pixel
img[6, 40] = (0, 0, 255)

# get a certain region of the image
top_left_corner = img[0:50, 0:50]
```

### Accessing and manipulating pixels in OpenCV with grayscale images

```python
# read an image
# Second argument is a flag specifying the way the image should be read.
gray_img = cv2.imread('logo.png', cv2.IMREAD_GRAYSCALE)

# Get the value of the pixel (x=40, y=6):
i = gray_img[6, 40]

# Set the pixel to black:
gray_img[6, 40] = 0
```

### BGR order in OpenCV and convert it

*Other python packages use the RGB color format rather than BGR, such as Matplotlib.*

```python
## how to convert the format

# load the image
img_OpenCV = cv2.imread('logo.png') 
# split the loaded image into its three channels (b,g,r)
b, g, r = cv2.split(img_OpenCV)
# Or
B = img_OpenCV[:,:,0],G = img_OpenCV[:,:,1],R = img_OpenCV[:,:,2]
# merge again the three channels but in the RGB format
img_matplotlib = cv2.merge([r,g,b])
# Or 
img_RGB = img_OpenCV[:,:,::-1]
# Show both images (img_OpenCV and img_matplotlib) using matplotlib
# This will show the image in wrong color:
plt.subplot(121)
plt.imshow(img_OpenCV)
plt.title('img_OpenCV')
# This will show the image in true color:
plt.subplot(122)
plt.imshow(img_matplotlib)
plt.title('img_matplotlib')
plt.show()
# Show both images (img_OpenCV and img_matplotlib) using cv2.imshow()
# This will show the image in true color:
cv2.imshow('bgr image', img_OpenCV)
# This will show the image in wrong color:
cv2.imshow('rgb image', img_matplotlib)
cv2.waitKey(0)
cv2.destroyAllWindows()
# To stack horizontally (img_OpenCV to the left of img_matplotlib):
img_concats = np.concatenate((img_OpenCV, img_matplotlib), axis=1)
```

----------------------------------------------------- 2021/7/4日完成 -----------------------------------------------------------------

## Chapter 03 - Handling Files and Images

<img src="D:\学期计划\2021_暑假\img\process.png" alt="process" style="zoom:25%;" />

 *Command-line arguments are a common and simple way to parameterize the execution of programs.* 

### sys.argv

<img src="D:\学期计划\2021_暑假\img\sys.png" alt="sys" style="zoom:25%;" />

 *sysargv_python.py* 

```python
# to handle command-line arguments, python uses sys.argv
import sys
print("the name of the script being processed is:'{}'".format(sys.argv[0]))
print("the number of arguments of the script is: '{}'".format(len(sys.argv)))
print("the arguments of the script are: '{}'".format(str(sys.argv)))
```

```
# python script_name first_arg
D:>python sysargv_python.py opencv
```

### argparse - command-line option and argument parsing

*argparse library handles command-line arguments in a systematic way.*

- First, the program determines what arguments it requires.
- Second, *argparse* will work out how to parse these arguments to *sys.argv*.
- Also,  *argparse* produces help and usage messages, and issues errors when invalid arguments are provided. 

 

*argparse_positional_arguments.py* 

```python
import argparse

# first, create the ArgumentParser object
parser = argparse.ArgumentParser()

# add a positional argument
parser.add_argument("first_argument", help="this is the string text in connection with first_argument")

# parse arguments
args = parser.parse_args()

# get and print the first argument of the script
print(args.first_argument)
```

```
# output
# 1
D:\...\mastering>python argparse_positional_arguments.py
usage: argparse_positional_arguments.py [-h] first_argument
argparse_positional_arguments.py: error: the following arguments are required: first_argument

# 2
D:\...\mastering>python argparse_positional_arguments.py -h
usage: argparse_positional_arguments.py [-h] first_argument
positional arguments:
  first_argument  this is the string text in connection with first_argument
optional arguments:
  -h, --help      show this help message and exit
  
# 3
D:\...\mastering>python argparse_positional_arguments.py 6
6

# 4
D:\...\mastering>python argparse_positional_arguments.py -h 6
usage: argparse_positional_arguments.py [-h] first_argument
positional arguments:
  first_argument  this is the string text in connection with first_argument
optional arguments:
  -h, --help      show this help message and exit
 
## 可以看到加上-h后无论是否给出参数，输出结果都一样
## 代码种help信息只有在加上-h后才会输出
## 因此一般选择 2 和 3，2用来提示变量是什么，3用来显示输出结果
```

but,  *argparse* treats the options we give it as strings.  So, if the parameter is not a string, the *type* option should be established.

```python
parser.add_argument("first_number", help="first number to be added", type=int)
parser.add_argument("second_number", help="second number to be added", type=int)
# Additionally, the arguments can be stored in a dictionary calling vars() function
args_dict = vars(parser.parse_args())
print("args_dict dictionary: '{}'".format(args_dict))
```

```
args_dict dictionary: '{'first_number': 5, 'second_number': 10}'
```

### Reading images in OpenCV

```python
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('img_path',help='the path of the image')

args = parser.parse_args()

img = cv2.imread(args.img_path)
img2 = cv2.imread(vars(args)['img_path'])

cv2.imshow('img1',img)
cv2.imshow('img2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Reading and writing images in OpenCV

```python
# convert the image into grayscale
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('img_path_input',help='the input path of image')
parser.add_argument('img_path_output',help='the output path of image')

args = vars(parser.parse_args())

img_color = cv2.imread(args['img_path_input'])
# ** 注意是COLOR_BGR2GRAY不是COLOR_RGB2GRAY
img_gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
cv2.imwrite(args['img_path_output'],img_gray)
cv2.imshow('color image', img_color)
cv2.imshow('gray image', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Reading camera frames and save

```python
# If you have connected a webcam to your computer, it has an index of 0. 
# Additionally, if you have a second camera, you can select it by passing 1. 
# As you can see, the type of this parameter is int. 
parser.add_argument("index_camera", help="index of the camera to read from", type=int)
args = parser.parse_args()
# First, create a VideoCapture object to read from the camera (pass 0):
capture = cv2.VideoCapture(args.index_camera)

# read a video file:
parser.add_argument("video_path", help="path to the video file")
args = parser.parse_args()
capture = cv2.VideoCapture(args.video_path)

# read a IP camera
# cv2.VideoCapture("http://217.126.89.102:8010/axis-cgi/mjpg/video.cgi")
parser.add_argument("ip_url", help="IP URL to connect")
args = parser.parse_args()
capture = cv2.VideoCapture(args.ip_url)

# To check whether the connection has been established correctly
# return a bool
capture.isOpened()

# To capture footage frame by frame from the camera
# return a bool which indicates whether the frame has been correctly read from the capture object. 
capture.read()

# To access some properties of the capture object
# If we call a property that is not supported, the returned value will be 0:
capture.get(property_identifier)

# save some frames to disk when something interesting happens
cv2.imwrite()
```

```python
# convert capture image into grayscale
# and save
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('index_camera', help='the index of camera', type=int)
args = parser.parse_args()

capture = cv2.VideoCapture(args.index_camera)

capture_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
capture_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
capture_fps = capture.get(cv2.CAP_PROP_FPS)

print("the width of the capture: '{}'".format(capture_width))
print("the height of the capture: '{}'".format(capture_height))
print("the fps of the capture: '{}'".format(capture_fps))

if capture.isOpened() is False:
    print('open error')
    
frame_index = 0
# Read until video is completed, or 'q' is pressed
while capture.isOpened():
    ret, img_color = capture.read()
    if ret is True:
        cv2.imshow('color image', img_color)

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray image', img_gray)
		
        # 键盘输入c保存当前帧
        if cv2.waitKey(20) & 0xff == ord('c'):
            img_color_name = 'img_color_{}.png'.format(frame_index)
            img_gray_name = 'img_gray_{}.png'.format(frame_index)
            cv2.imwrite(img_color_name, img_color)
            cv2.imwrite(img_gray_name, img_gray)
            frame_index += 1
            
        # 键盘输入q退出
        # 当cv2.waitKey(20)返回正常时，cv2.waitKey(20) & 0xff = cv2.waitKey(1)
        # cv2.waitKey(20) & 0xff防止异常错误
        if cv2.waitKey(20) & 0xff == ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
capture.release()
```

### Writing a video file

#### frames per second (fps)

- 每秒帧数
- 虽然fps越高越好，但算法每秒应该处理的帧数取决于待解决的特定问题
- 如何计算：

```python
import time
process_start = time.time()
.
.
.
process_end = time.time()
process_time = process_end - process_start
cal_fps = 1.0 / process_time
```

#### Considerations for writing a video file

- 视频代码是用来压缩和解压数字视频的软件，压缩视频格式遵循一定的标准规范。OpenCV提供 *FOURCC* 来指定视频编码器。典型的编码器有DIVX, XVID, X264和MJPG，但使用特定编码器依赖于平台是否安装
- 典型的视频文件格式有AVI, MP4, MOV, WMV
- 视频文件格式与FOURCC的正确组合并不简单，可能需要尝试

在OpenCV中创建视频时需要考虑的因素：

<img src="D:\学期计划\2021_暑假\img\consderations.png" alt="consderations" style="zoom:33%;" />

```python
# 摄像头采集视频灰度化后保存
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("output_video_path", help="path to the video file to write")
args = parser.parse_args()

capture = cv2.VideoCapture(0)

capture_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
capture_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
capture_fps = capture.get(cv2.CAP_PROP_FPS)

print("the width of the capture: '{}'".format(capture_width))
print("the height of the capture: '{}'".format(capture_height))
print("the fps of the capture: '{}'".format(capture_fps))

# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# The last argument is False so that we can write the video in grayscale. If we want to create the video in color, this last argument should be True:
out_gray = cv2.VideoWriter(args.output_video_path, fourcc,int(capture_fps),(int(capture_width), int(capture_height)), False)

if capture.isOpened() is False:
    print('open error')

while capture.isOpened():
    ret, img_color = capture.read()
    if ret is True:
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        out_gray.write(img_gray)
        cv2.imshow('gray image', img_gray)

        if cv2.waitKey(20) & 0xff == ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
capture.release()
out_gray.release()
```

### Playing with video capture properties

#### Getting all the properties from the video capture object

```python
import cv2
import argparse


def decode_fourcc(fourcc):
    """Decodes the fourcc value to get the four chars identifying it"""

    # Convert to int:
    fourcc_int = int(fourcc)

    # We print the int value of fourcc
    print("int value of fourcc: '{}'".format(fourcc_int))

    # We can also perform this in one line:
    # return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    fourcc_decode = ""
    for i in range(4):
        int_value = fourcc_int >> 8 * i & 0xFF
        print("int_value: '{}'".format(int_value))
        fourcc_decode += chr(int_value)
    return fourcc_decode


# We first create the ArgumentParser object
# The created object 'parser' will have the necessary information
# to parse the command-line arguments into data types.
parser = argparse.ArgumentParser()

# We add 'video_path' argument using add_argument() including a help.
parser.add_argument("video_path", help="path to the video file")
args = parser.parse_args()

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
capture = cv2.VideoCapture(args.video_path)

# Get and print these values:
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS)))
print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))
print("CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))
print("CAP_PROP_FOURCC  : '{}'".format(decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))))
print("CAP_PROP_FRAME_COUNT  : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
print("CAP_PROP_MODE : '{}'".format(capture.get(cv2.CAP_PROP_MODE)))
print("CAP_PROP_BRIGHTNESS : '{}'".format(capture.get(cv2.CAP_PROP_BRIGHTNESS)))
print("CAP_PROP_CONTRAST : '{}'".format(capture.get(cv2.CAP_PROP_CONTRAST)))
print("CAP_PROP_SATURATION : '{}'".format(capture.get(cv2.CAP_PROP_SATURATION)))
print("CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE)))
print("CAP_PROP_GAIN  : '{}'".format(capture.get(cv2.CAP_PROP_GAIN)))
print("CAP_PROP_EXPOSURE : '{}'".format(capture.get(cv2.CAP_PROP_EXPOSURE)))
print("CAP_PROP_CONVERT_RGB : '{}'".format(capture.get(cv2.CAP_PROP_CONVERT_RGB)))
print("CAP_PROP_RECTIFICATION : '{}'".format(capture.get(cv2.CAP_PROP_RECTIFICATION)))
print("CAP_PROP_ISO_SPEED : '{}'".format(capture.get(cv2.CAP_PROP_ISO_SPEED)))
print("CAP_PROP_BUFFERSIZE : '{}'".format(capture.get(cv2.CAP_PROP_BUFFERSIZE)))

# Check if camera opened successfully
if capture.isOpened() is False:
    print("Error opening video stream or file")

# Read until video is completed
while capture.isOpened():
    # Capture frame-by-frame
    ret, frame = capture.read()

    if ret is True:

        # Print current frame number per iteration
        print("CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))

        # Get the timestamp of the current frame in milliseconds
        print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))

        # Display the resulting frame
        cv2.imshow('Original frame', frame)

        # Convert the frame to grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the grayscale frame
        cv2.imshow('Grayscale frame', gray_frame)

        # Press q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# Release everything:
capture.release()
cv2.destroyAllWindows()

```

#### Play a video backwards and save it

```python
import cv2
import argparse

def decode_fourcc(fourcc):
    """Decodes the fourcc value to get the four chars identifying it"""
    fourcc_int = int(fourcc)
    print("int value of fourcc: '{}'".format(fourcc_int))
    return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

parser = argparse.ArgumentParser()
parser.add_argument("video_path", help="path to the video file")
parser.add_argument("output_video_path", help="path to the video file to write")
args = parser.parse_args()

capture = cv2.VideoCapture(args.video_path)

# Get some properties of VideoCapture (frame width, frame height and frames per second (fps)):
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)
codec = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
print("codec: '{}'".format(codec))

fourcc = cv2.VideoWriter_fourcc(*codec)
out = cv2.VideoWriter(args.output_video_path, fourcc, int(fps), (int(frame_width), int(frame_height)), True)

# Check if camera opened successfully
if capture.isOpened() is False:
    print("Error opening video stream or file")

# We get the index of the last frame of the video file
frame_index = capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1

while capture.isOpened() and frame_index >= 0:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = capture.read()

    if ret is True:
        out.write(frame)
        frame_index = frame_index - 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# Release everything:
capture.release()
out.release()
cv2.destroyAllWindows()
```

----------------------------------------------------- 2021/7/5日完成 -----------------------------------------------------------------

## Chapter 04 - Constructing Basic Shapes in OpenCV

Draw basic shapes on the image in order to do the following:

- Show some intermediate results of your algorithm
- Show the final results of your algorithm
- Show some debugging information

通常建一个 *常量文件* 定义颜色或 *定义字典* 等，用于其他文件引用，例如：

```python
# constant.py
"""
Common colors triplets (BGR space) to use in OpenCV
"""
BLUE = (255, 0, 0) # 常量用大写
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (125, 125, 125)
DARK_GRAY = (50, 50, 50) # 用下划线隔开
LIGHT_GRAY = (220, 220, 220)

# 定义字典
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255), 'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(), 'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

# other.py
import constant

# Get red color
print("red:'{}'".format(constant.RED))
```

利用matplotlib显示图片，因此定义函数如下：

```python
def show_with_matplotlib(img, title):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB:
    img_RGB = img[:, :, ::-1]

    # Show the image using matplotlib:
    plt.imshow(img_RGB)
    plt.title(title)
    plt.show()
```

### Basic shapes - lines, rectangles, and circles

```python
# create the canvas to draw: 400 x 400 pixels, 3 channels, uint8
image = np.zeros((400,400,3), dtype="uint8")

# set background color
image[:] = colors['light_gray']

# 1. lines
cv2.line(image, (0,0),(400,400),colors['green'],3)
show_with_matplotlib(image, 'cv2.line()')
# clean the canvas
image[:] = colors['light_gray']

# 2. rectangles
cv2.rectangle(image, (10, 50), (60, 300), colors['green'], 3) # 长方形框
cv2.rectangle(image, (80, 50), (130, 300), colors['blue'], -1) # 填充
show_with_matplotlib(image, 'cv2.rectangle()')
# clean the canvas
image[:] = colors['light_gray']

# 3. circles
cv2.circle(image, (50, 50), 20, colors['green'], 3) # 圆形框
cv2.circle(image, (100, 100), 30, colors['blue'], -1) # 填充
show_with_matplotlib(image, 'cv2.circle()')
```

### Advanced shapes

```python
# 1. 剪切矩阵内部的直线
# Draw a rectangle and a line:
cv2.line(image, (0, 0), (300, 300), colors['green'], 3)
cv2.rectangle(image, (0, 0), (100, 100), colors['blue'], 3)
# We call the function cv2.clipLine():
ret, p1, p2 = cv2.clipLine((0, 0, 100, 100), (0, 0), (300, 300))
# cv2.clipLine() returns False if the line is outside the rectangle
# And returns True otherwise
if ret:
    cv2.line(image, p1, p2, colors['yellow'], 3)

# Show image:
show_with_matplotlib(image, 'cv2.clipLine()')

# 2. 创建箭头
# cv.arrowedLine(img, pt1, pt2, color, thickness=1, lineType=8, shift=0, tipLength=0.1)
cv2.arrowedLine(image, (50, 50), (200, 50), colors['red'], 3, 8, 0, 0.1)
cv2.arrowedLine(image, (50, 120), (200, 120), colors['green'], 3, cv2.LINE_AA, 0, 0.3)
cv2.arrowedLine(image, (50, 200), (200, 200), colors['blue'], 3, 8, 0, 0.3)

# 3. 创建椭圆
cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=8, shift=0)

# 4. 创建多边形
# cv2.polylines(image, pts, isClosed, color, thickness=1, lineType=8, shift=0)
# These points define a triangle:
pts = np.array([[250, 5], [220, 80], [280, 80]], np.int32)
# Reshape to shape (number_vertex, 1, 2)
pts = pts.reshape((-1, 1, 2))
# Print the shapes: this line is not necessary, only for visualization:
print("shape of pts '{}'".format(pts.shape))
# Draw this polygon with True option:
cv2.polylines(image, [pts], True, colors['green'], 3)
```

另外，对于有 *shift* 参数的绘图函数，是指其绘制精度可以达到亚像素级别，使用方式如下：

```python
def draw_float_circle(img, center, radius, color, thickness=1, lineType=8, shift=4):
    """Wrapper function to draw float-coordinate circles"""

    factor = 2 ** shift
    center = (int(round(center[0] * factor)), int(round(center[1] * factor)))
    radius = int(round(radius * factor))
    cv2.circle(img, center, radius, color, thickness, lineType, shift)
   
draw_float_circle(image, (299, 299), 300, colors['red'], 1, 8, 0)
draw_float_circle(image, (299.9, 299.9), 300, colors['green'], 1, 8, 1)
draw_float_circle(image, (299.99, 299.99), 300, colors['blue'], 1, 8, 2)
draw_float_circle(image, (299.999, 299.999), 300, colors['yellow'], 1, 8, 3)
```

### Drawing text

```python
cv2.putText( img, text, org, fontFace, fontScale, color, thickness=1, lineType= 8, bottomLeftOrigin=False)

# image.fill(255) 将上述背景改为白色

# 其他相关函数
# 返回font Scale (fontScale)
retval = cv2.getFontScaleFromHeight(fontFace, pixelHeight, thickness=1)
# get the text size
retval, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
```

### Dynamic drawing with mouse events

"""
	windowName: name
	onMouse: callback function
	param: additional information
"""
cv2.setMouseCallback(windowName, onMouse, param=None)

```python
# First, create the callback function
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("event: EVENT_LBUTTONDBLCLK")
        cv2.circle(image, (x, y), 10, colors['magenta'], -1)

    if event == cv2.EVENT_MOUSEMOVE:
        print("event: EVENT_MOUSEMOVE")

    if event == cv2.EVENT_LBUTTONUP:
        print("event: EVENT_LBUTTONUP")

    if event == cv2.EVENT_LBUTTONDOWN:
        print("event: EVENT_LBUTTONDOWN")
        
# Second, create a named window where the mouse callback will be established
cv2.namedWindow('Image mouse')

# Finanlly, set the mouse callback function 
cv2.setMouseCallback('Image mouse', draw_circle)
```

Practice 1: 

You can do the following:

- Add a circle using the double left-click
- Delete the last added circle using a simple left-click
- Delete all circles using the double right-click

```python
import cv2
import numpy as np

# Dictionary containing some colors:
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}


# This is the mouse callback function:
def draw_text():
    # We set the position to be used for drawing text:
    menu_pos = (10, 500)
    menu_pos2 = (10, 525)
    menu_pos3 = (10, 550)
    menu_pos4 = (10, 575)

    # Write the menu:
    cv2.putText(image, 'Double left click: add a circle', menu_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
    cv2.putText(image, 'Simple right click: delete last circle', menu_pos2, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255))
    cv2.putText(image, 'Double right click: delete all circle', menu_pos3, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255))
    cv2.putText(image, 'Press \'q\' to exit', menu_pos4, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))


# mouse callback function
def draw_circle(event, x, y, flags, param):
    """Mouse callback function"""

    global circles
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # Add the circle with coordinates x,y
        print("event: EVENT_LBUTTONDBLCLK")
        circles.append((x, y))
    if event == cv2.EVENT_RBUTTONDBLCLK:
        # Delete all circles (clean the screen)
        print("event: EVENT_RBUTTONDBLCLK")
        circles[:] = []
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Delete last added circle
        print("event: EVENT_RBUTTONDOWN")
        try:
            circles.pop()
        except (IndexError):
            print("no circles to delete")
    if event == cv2.EVENT_MOUSEMOVE:
        print("event: EVENT_MOUSEMOVE")
    if event == cv2.EVENT_LBUTTONUP:
        print("event: EVENT_LBUTTONUP")
    if event == cv2.EVENT_LBUTTONDOWN:
        print("event: EVENT_LBUTTONDOWN")


# Structure to hold the created circles:
circles = []

# We create the canvas to draw: 600 x 600 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set the background to black using np.zeros():
image = np.zeros((600, 600, 3), dtype="uint8")

# We create a named window where the mouse callback will be established:
cv2.namedWindow('Image mouse')

# We set the mouse callback function to 'draw_circle':
cv2.setMouseCallback('Image mouse', draw_circle)

# We draw the menu:
draw_text()

# We get a copy with only the text printed in it:
clone = image.copy()

while True:

    # We 'reset' the image (to get only the image with the printed text):
    image = clone.copy()

    # We draw now only the current circles:
    for pos in circles:
        # We print the circle (filled) with a  fixed radius (30):
        cv2.circle(image, pos, 30, colors['blue'], -1)

    # Show image 'Image mouse':
    cv2.imshow('Image mouse', image)

    # Continue until 'q' is pressed:
    if cv2.waitKey(400) & 0xFF == ord('q'):
        break

# Destroy all generated windows:
cv2.destroyAllWindows()

```

Practice 2:

时钟

<img src="D:\学期计划\2021_暑假\img\clock.png" alt="clock" style="zoom:50%;" />

```python
import cv2
import numpy as np
import datetime
import math


def array_to_tuple(arr):
    return tuple(arr.reshape(1, -1)[0])


# Dictionary containing some colors
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

# We create the canvas to draw: 640 x 640 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set background to black using np.zeros()
image = np.zeros((640, 640, 3), dtype="uint8")

# If you want another background color you can do the following:
image[:] = colors['light_gray']

# Coordinates to define the origin for the hour markings:
hours_orig = np.array(
    [(620, 320), (580, 470), (470, 580), (320, 620), (170, 580), (60, 470), (20, 320), (60, 170), (169, 61), (319, 20),
     (469, 60), (579, 169)])

# Coordinates to define the destiny for the hour markings:
hours_dest = np.array(
    [(600, 320), (563, 460), (460, 562), (320, 600), (180, 563), (78, 460), (40, 320), (77, 180), (179, 78), (319, 40),
     (459, 77), (562, 179)])

# We draw the hour markings:
for i in range(0, 12):
    cv2.line(image, array_to_tuple(hours_orig[i]), array_to_tuple(hours_dest[i]), colors['black'], 3)

# We draw a big circle, corresponding to the shape of the analog clock
cv2.circle(image, (320, 320), 310, colors['dark_gray'], 8)

# We draw the rectangle containig the text and the text "Mastering OpenCV 4 with Python":
cv2.rectangle(image, (150, 175), (490, 270), colors['dark_gray'], -1)
cv2.putText(image, "Mastering OpenCV 4", (150, 200), 1, 2, colors['light_gray'], 1, cv2.LINE_AA)
cv2.putText(image, "with Python", (210, 250), 1, 2, colors['light_gray'], 1, cv2.LINE_AA)

# We make a copy of the image with the "static" information
image_original = image.copy()

# Now, we draw the "dynamic" information:
while True:
    # Get current date:
    date_time_now = datetime.datetime.now()
    # Get current time from the date:
    time_now = date_time_now.time()
    # Get current hour-minute-second from the time:
    hour = math.fmod(time_now.hour, 12)
    minute = time_now.minute
    second = time_now.second

    print("hour:'{}' minute:'{}' second: '{}'".format(hour, minute, second))

    # Get the hour, minute and second angles:
    second_angle = math.fmod(second * 6 + 270, 360)
    minute_angle = math.fmod(minute * 6 + 270, 360)
    hour_angle = math.fmod((hour * 30) + (minute / 2) + 270, 360)

    print("hour_angle:'{}' minute_angle:'{}' second_angle: '{}'".format(hour_angle, minute_angle, second_angle))

    # Draw the lines corresponding to the hour, minute and second needles
    second_x = round(320 + 310 * math.cos(second_angle * 3.14 / 180))
    second_y = round(320 + 310 * math.sin(second_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (second_x, second_y), colors['blue'], 2)

    minute_x = round(320 + 260 * math.cos(minute_angle * 3.14 / 180))
    minute_y = round(320 + 260 * math.sin(minute_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (minute_x, minute_y), colors['blue'], 8)

    hour_x = round(320 + 220 * math.cos(hour_angle * 3.14 / 180))
    hour_y = round(320 + 220 * math.sin(hour_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (hour_x, hour_y), colors['blue'], 10)

    # Finally, a small circle, corresponding to the point where the three needles joint, is drawn:
    cv2.circle(image, (320, 320), 10, colors['dark_gray'], -1)

    # Show image:
    cv2.imshow("clock", image)

    # We get the image with the static information:
    image = image_original.copy()

    # A wait of 500 milliseconds is performed (to see the displayed image)
    # Press q on keyboard to exit the program:
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

# Release everything:
cv2.destroyAllWindows()

```

 ----------------------------------------------------- 2021/7/6日完成 -----------------------------------------------------------------

## Chapter 05 - Image Processing Techniques

### 通道的拆分（split）与合并（merge）

- *cv2.split()* : split the source multichannel image into several single-channel images
- *cv2.merge()* : merge several single-channel images into a multichannel image

```python
(b,g,r) = cv2.split(image)
# but, cv2.split() is a time-consuming operation
# otherwise, we can use the Numpy to work with specific channels
b = image[:,:,0],g = image[:,:,1],r = image[:,:,2]

image_copy = cv2.merge((b,g,r))

# 令某一通道为0
image_without_blue = image.copy()
image_without_blue[:,:,0] = 0
```

### 图像的几何变换

- cv2.warpAffine() : transforms the source image by using the following *2 x 3* *M* transformation matrix:

  ​                        dis(x,y) = src(M11x + M12y + M13, M21x + M22y + M23)

- cv2.warpPerspective() :  transforms the source image using the following *3 x 3* transformation matrix: 

  dis(x,y) = src((M11x + M12y + M13)/(M31x + M32y + M33), (M21x + M22y + M23)/(M31xM32y + M33))

```python
# Get the height and width of the image:
height, width = image.shape[:2]

## 1.缩放图像
dst_image = cv2.resize()

## 2.平移图像
# first,create the 2×3 transformation matrix
# x：x方向平移的像素数，y：y方向平移的像素数
M = np.float32([[1,0,x],[0,1,y]])
# second,use cv2.warpAffine(image,M,(width,height)) to transform
cv2.warpAffine(image,M,(width,height))

## 3.旋转图像
# first, use cv.getRotationMatrix2D() to build the 2x3 rotation matrix
# (width / 2.0, height / 2.0)：旋转中心，180：旋转180°，1：比例因子
M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 180, 1)
# second, use cv2.warpAffine(image,M,(width,height)) to rotate
dst_image = cv2.warpAffine(image, M, (width, height))

## 4.图像的仿射变换：一种二维坐标到二维坐标之间的线性变换
# first, use cv2.getAffineTransform() to build the 2 x 3 matrix
# pts_1：原图中的三个点 pts_2：变换后的三个点
pts_1 = np.float32([[135, 45], [385, 45], [135, 230]])
pts_2 = np.float32([[135, 45], [385, 45], [150, 230]])
M = cv2.getAffineTransform(pts_1, pts_2)
# second, use cv2.warpAffine(image,M,(width,height))
dst_image = cv2.warpAffine(image_points, M, (width, height))

## 5.透视变换：将图像投影到一个新的视平面
# first, use cv2.getPerspectiveTransform() to build the 3 x 3 matrix
# pts_1：原图中的四个点 pts_2：变换后的四个点
pts_1 = np.float32([[450, 65], [517, 65], [431, 164], [552, 164]])
pts_2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts_1, pts_2)
# second, use cv2.warpPerspective(image,M,(width,height))
dst_image = cv2.warpPerspective(image, M, (300, 300))

## 6.图片裁剪
dst_image = image[80:200, 230:330]
```

### 图片滤波（filter）

噪声类型： [常见的噪声](https://blog.csdn.net/weixin_40446557/article/details/81451651) 

```python
"""
	Smoothing images with different methods
"""

## way 1.使用卷积核
kernel_averaging_5_5 = np.array([[0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04],[0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04]])
# or kernel_averaging_5_5 = np.ones((5, 5), np.float32) / 25
smooth_image_f2D = cv2.filter2D(image, -1, kernel_averaging_5_5)

## way 2.使用 the normalized box filter
smooth_image_b = cv2.blur(image, (10, 10))
# Or
smooth_image_bfi = cv2.boxFilter(image, -1, (10, 10), normalize=True)

## way 3.使用 a Gaussian kernel
smooth_image_gb = cv2.GaussianBlur(image, (9, 9), 0)

## way 4.使用 a median kernel
# 这种滤波器可以用来降低图像的椒盐噪声
smooth_image_mb = cv2.medianBlur(image, 9)

## way 5.使用 a bilateral filter
# 可用于减少噪音，同时保持边缘锋利
smooth_image_bf = cv2.bilateralFilter(image, 5, 10, 10)
```

除此之外，上述还可用于锐化图片

### 常用的卷积核

```python
# 利用cv2.filter2D()实现

# 1.Identify kernel (does not modify the image)
kernel_identity = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
# 2.edge detection
kernel_edge_detection_1 = np.array([[1, 0, -1],
                                    [0, 0, 0],
                                    [-1, 0, 1]])

kernel_edge_detection_2 = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])

kernel_edge_detection_3 = np.array([[-1, -1, -1],
                                    [-1, 8, -1],
                                    [-1, -1, -1]])
# 3.sharpening
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

kernel_unsharp_masking = -1 / 256 * np.array([[1, 4, 6, 4, 1],
                                              [4, 16, 24, 16, 4],
                                              [6, 24, -476, 24, 6],
                                              [4, 16, 24, 16, 4],
                                              [1, 4, 6, 4, 1]])
# 4.smoothing
kernel_blur = 1 / 9 * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

gaussian_blur = 1 / 16 * np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]])
# 5.embossing
kernel_emboss = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]])
# 6.edge detection
sobel_x_kernel = np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])

sobel_y_kernel = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])

outline_kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
```

### 创建卡通图像

```python
"""
Cartoonizing images using both custom and OpenCV functions
"""

# Import required packages:
import cv2
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB:
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def sketch_image(img):
    """Sketches the image applying a laplacian operator to detect the edges"""

    # Convert to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median filter
    img_gray = cv2.medianBlur(img_gray, 5)

    # Detect edges using cv2.Laplacian()
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)

    # Threshold the edges image:
    ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

    return thresholded


def cartonize_image(img, gray_mode=False):
    """Cartoonizes the image applying cv2.bilateralFilter()"""

    # Get the sketch:
    thresholded = sketch_image(img)

    # Apply bilateral filter with "big numbers" to get the cartoonized effect:
    filtered = cv2.bilateralFilter(img, 10, 250, 250)

    # Perform 'bitwise and' with the thresholded img as mask in order to set these values to the output
    cartoonized = cv2.bitwise_and(filtered, filtered, mask=thresholded)

    if gray_mode:
        return cv2.cvtColor(cartoonized, cv2.COLOR_BGR2GRAY)

    return cartoonized


# Create the dimensions of the figure and set title:
plt.figure(figsize=(14, 6))
plt.suptitle("Cartoonizing images", fontsize=14, fontweight='bold')

# Load image:
image = cv2.imread('cat.jpg')

# Call the created functions for sketching and cartoonizing images:
custom_sketch_image = sketch_image(image)
custom_cartonized_image = cartonize_image(image)
custom_cartonized_image_gray = cartonize_image(image, True)

# Call the OpenCV functions to get a similar output:
sketch_gray, sketch_color = cv2.pencilSketch(image, sigma_s=30, sigma_r=0.1, shade_factor=0.1)
stylizated_image = cv2.stylization(image, sigma_s=60, sigma_r=0.07)

# Display all the resulting images:
show_with_matplotlib(image, "image", 1)
show_with_matplotlib(cv2.cvtColor(custom_sketch_image, cv2.COLOR_GRAY2BGR), "custom sketch", 2)
show_with_matplotlib(cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR), "sketch gray cv2.pencilSketch()", 3)
show_with_matplotlib(sketch_color, "sketch color cv2.pencilSketch()", 4)
show_with_matplotlib(stylizated_image, "cartoonized cv2.stylization()", 5)
show_with_matplotlib(custom_cartonized_image, "custom cartoonized", 6)
show_with_matplotlib(cv2.cvtColor(custom_cartonized_image_gray, cv2.COLOR_GRAY2BGR), "custom cartoonized gray", 7)

# Show the created image:
plt.show()

```

<img src="D:\学期计划\2021_暑假\img\cartoon.png" alt="cartoon" style="zoom:50%;" />

### 图像运算

#### 饱和运算

通过限制运算可以采用的最大值和最小值，将运算限制在一个固定的范围内。在图像处理里经常有(比如说增加亮度)两种灰度值运算后要判断值是否大于255或小于0，根据结果再取255或0 

例如

```python
x = np.uint8([250])
y = np.uint8([50])

# OpenCV addition: values are clipped to ensure they never fall outside the range [0,255]
# 250+50 = 300 => 255:
result_opencv = cv2.add(x, y)
print("cv2.add(x:'{}' , y:'{}') = '{}'".format(x, y, result_opencv))

# NumPy addition: values wrap around
# 250+50 = 300 % 256 = 44:
result_numpy = x + y
print("x:'{}' + y:'{}' = '{}'".format(x, y, result_numpy))
```

```
cv2.add(x:'[250]' , y:'[50]') = '[[255]]'
x:'[250]' + y:'[50]' = '[44]
```

#### 图像加减法

```python
# cv2.add()的用法
M = np.ones(image.shape, dtype="uint8") * 60
added_image = cv2.add(image, M)

# cv2.subtract()的用法
subtracted_image = cv2.subtract(image, M)

# When we add a value, the image will be lighter, 
# and when we subtract a value, it will be darker
```

#### 计算图像梯度

```python
# x方向梯度
# the depth of the output is set to CV_16S to avoid overflow
# CV_16S = one channel of 2-byte signed integers (16-bit signed integers)
gradient_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0, 3)
# y方向梯度
gradient_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1, 3)

# Conversion to an unsigned 8-bit type:
abs_gradient_x = cv2.convertScaleAbs(gradient_x)
abs_gradient_y = cv2.convertScaleAbs(gradient_y)

# Combine the two images using the same weight:
sobel_image = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)
```

#### 按位运算

```python
# Bitwise OR
bitwise_or = cv2.bitwise_or(img_1, img_2)

# Bitwise AND
bitwise_and = cv2.bitwise_and(img_1, img_2)

# Bitwise XOR
bitwise_xor = cv2.bitwise_xor(img_1, img_2)

# Bitwise NOT
bitwise_not_1 = cv2.bitwise_not(img_1)
```

<img src="D:\学期计划\2021_暑假\img\Bitwise.png" alt="Bitwise" style="zoom:50%;" />

<img src="D:\学期计划\2021_暑假\img\Bitwise2.png" alt="Bitwise2" style="zoom:50%;" />

### 图像形态变换

- 形态变换是一些基于图像形状的简单操作，通常在二进制图像上执行。
- 两个输入：原始图像，决定操作性质的内核

#### 常见内核创建

```python
"""
	kernel_type: 内核形状
				 矩形：cv2.MORPH_RECT
                 十字形：cv2.MORPH_CROSS
                 椭圆形：cv2.MORPH_ELLIPSE
	kernel_size: 内核大小
"""
cv2.getStructuringElement(kernel_type, kernel_size)

```



#### 扩张（Dilation）

- 内核区域存在像素值为1，整个区域就置1

```python
dilation = cv2.dilate(image, kernel, iterations=1)
```

#### 侵蚀（Erosion）

- 内核区域存在的像素值全为1，整个区域才置1，否则置为0

```python
erosion = cv2.erode(image, kernel, iterations=1)
```

#### 开场（Opening）

```python
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
```

#### 闭幕（Closing）

```python
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
```

#### 形态梯度（Morphological gradient ）

```python
morph_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
```

#### 高顶礼帽（Top Hat ）

```python
top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
```

#### 黑帽（Black Hat ）

```python
black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
```

<img src="D:\学期计划\2021_暑假\img\Morpho.png" alt="Morpho" style="zoom:50%;" />

### 颜色空间

- OpenCV中提供超过150种颜色空间， BGR, RGB, HSV等
- 利用cv2.cvtColor() 进行转换

应用1：在不同颜色空间里进行皮肤分割

```python
"""
Skin segmentation algorithms based on different color spaces
"""

# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Name and path of the images to load:
image_names = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg']
path = 'skin_test_imgs'


# Load all test images building the relative path using 'os.path.join'
def load_all_test_images():
    """Loads all the test images and returns the created array containing the loaded images"""

    skin_images = []
    for index_image, name_image in enumerate(image_names):
        # Build the relative path where the current image is:
        image_path = os.path.join(path, name_image)
        # print("image_path: '{}'".format(image_path))
        # Read the image and add it (append) to the structure 'skin_images'
        skin_images.append(cv2.imread(image_path))
    # Return all the loaded test images:
    return skin_images


# Show all the images of the array creating the name for each one
def show_images(array_img, title, pos):
    """Shows all the images contained in the array"""

    for index_image, image in enumerate(array_img):
        show_with_matplotlib(image, title + "_" + str(index_image + 1), pos + index_image)


# Shows the image 'color_img' in the indicated position 'pos'
def show_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(5, 6, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Lower and upper boundaries for the HSV skin segmentation method:
lower_hsv = np.array([0, 48, 80], dtype="uint8")
upper_hsv = np.array([20, 255, 255], dtype="uint8")


# Skin detector based on the HSV color space
def skin_detector_hsv(bgr_image):
    """Skin segmentation algorithm based on the HSV color space"""

    # Convert image from BGR to HSV color space:
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Find region with skin tone in HSV image:
    skin_region = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    return skin_region


# Lower and upper boundaries for the HSV skin segmentation method:
lower_hsv_2 = np.array([0, 50, 0], dtype="uint8")
upper_hsv_2 = np.array([120, 150, 255], dtype="uint8")


# Skin detector based on the HSV color space
def skin_detector_hsv_2(bgr_image):
    """Skin segmentation algorithm based on the HSV color space"""

    # Convert image from BGR to HSV color space:
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Find region with skin tone in HSV image:
    skin_region = cv2.inRange(hsv_image, lower_hsv_2, upper_hsv_2)
    return skin_region


# Lower and upper boundaries for the YCrCb skin segmentation method:
# Values taken for the publication: 'Face Segmentation Using Skin-Color Map in Videophone Applications'
# The same values appear in the publication 'Skin segmentation using multiple thresholding'
# (Cb in [77, 127]) and (Cr in [133, 173])
lower_ycrcb = np.array([0, 133, 77], dtype="uint8")
upper_ycrcb = np.array([255, 173, 127], dtype="uint8")


# Skin detector based on the YCrCb color space
def skin_detector_ycrcb(bgr_image):
    """Skin segmentation algorithm based on the YCrCb color space.
    See 'Face Segmentation Using Skin-Color Map in Videophone Applications'"""

    # Convert image from BGR to YCrCb color space:
    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skin_region = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)
    return skin_region


# Values are taken from: 'RGB-H-CbCr Skin Colour Model for Human Face Detection'
# (R > 95) AND (G > 40) AND (B > 20) AND (max{R, G, B} − min{R, G, B} > 15) AND (|R − G| > 15) AND (R > G) AND (R > B)
# (R > 220) AND (G > 210) AND (B > 170) AND (|R − G| ≤ 15) AND (R > B) AND (G > B)
def bgr_skin(b, g, r):
    """Rule for skin pixel segmentation based on the paper 'RGB-H-CbCr Skin Colour Model for Human Face Detection'"""

    e1 = bool((r > 95) and (g > 40) and (b > 20) and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (
            abs(int(r) - int(g)) > 15) and (r > g) and (r > b))
    e2 = bool((r > 220) and (g > 210) and (b > 170) and (abs(int(r) - int(g)) <= 15) and (r > b) and (g > b))
    return e1 or e2


# Skin detector based on the BGR color space
def skin_detector_bgr(bgr_image):
    """Skin segmentation based on the RGB color space"""

    h = bgr_image.shape[0]
    w = bgr_image.shape[1]

    # We crete the result image with back background
    res = np.zeros((h, w, 1), dtype="uint8")

    # Only 'skin pixels' will be set to white (255) in the res image:
    for y in range(0, h):
        for x in range(0, w):
            (b, g, r) = bgr_image[y, x]
            if bgr_skin(b, g, r):
                res[y, x] = 255

    return res


# Implemented skin detectors to be used:
skin_detectors = {
    'ycrcb': skin_detector_ycrcb,
    'hsv': skin_detector_hsv,
    'hsv_2': skin_detector_hsv_2,
    'bgr': skin_detector_bgr
}


# Apply the 'skin_detector' to all the images in the array
def apply_skin_detector(array_img, skin_detector):
    """Applies the specific 'skin_detector' to all the images in the array"""

    skin_detector_result = []
    for index_image, image in enumerate(array_img):
        detected_skin = skin_detectors[skin_detector](image)
        bgr = cv2.cvtColor(detected_skin, cv2.COLOR_GRAY2BGR)
        skin_detector_result.append(bgr)
    return skin_detector_result


# create a figure() object with appropriate size and set title:
plt.figure(figsize=(15, 8))
plt.suptitle("Skin segmentation using different color spaces", fontsize=14, fontweight='bold')

# Show the skin_detectors dictionary
# This is only for debugging purposes
for i, (k, v) in enumerate(skin_detectors.items()):
    print("index: '{}', key: '{}', value: '{}'".format(i, k, v))

# We load all the test images:
test_images = load_all_test_images()

# We plot the test images:
show_images(test_images, "test img", 1)

# For each skin detector we apply and show all the test images:
show_images(apply_skin_detector(test_images, 'ycrcb'), "ycrcb", 7)
show_images(apply_skin_detector(test_images, 'hsv'), "hsv", 13)
show_images(apply_skin_detector(test_images, 'hsv_2'), "hsv_2", 19)
show_images(apply_skin_detector(test_images, 'bgr'), "bgr", 25)

# Show the created image:
plt.show()

```

<img src="D:\学期计划\2021_暑假\img\skinseg.png" alt="skinseg" style="zoom:50%;" />

### 伪彩色等效图片

在许多计算机视觉应用中，算法的输出是灰度图像。然而，人眼并不善于观察灰度图像的变化。它们对彩色图像的变化更为敏感，因此常用的方法是将灰度图像变换（重着色）为伪彩色等效图像。

```python
img_COLORMAP_HSV = cv2.applyColorMap(gray_img, cv2.COLORMAP_HSV)
```

<img src="D:\学期计划\2021_暑假\img\colormaps.png" alt="colormaps" style="zoom:50%;" />



#### 自定义色彩映射

第一种方法：定义一个颜色映射，将0到255个灰度值映射到256个颜色

```python
lut = np.zeros((256, 1, 3), dtype=np.uint8)
lut[0,0,0]=[...]
lut[0,0,1]=[...]
lut[0,0,2]=[...]
im_color = cv2.applyColorMap(im_gray, lut)
```

第二种方法：只提供一些关键颜色，然后插值以获得构建查找表所需要的所有颜色

## Chapter 06 - Constructing and Building Histogram

- 直方图便于更好理解图像内容的强大技术，例如相机实时显示正在捕获的场景中的直方图以便调整相机捕获时的参数
- 图像直方图是一种反映图像色调分布的直方图，它为每个色调值绘制像素数，每个色调值的像素数也被称为频率，例如 *h(80) = number of pixels with intensity 80*. 

<img src="D:\学期计划\2021_暑假\img\Grayscalehist.png" alt="Grayscalehist" style="zoom: 25%;" />

- bins :  Each of these 256 values is called a **bin** in histogram terminology in the previous screenshot
- range:   This is the range of intensity values we want to measure. 

### Grayscale histograms

```python
## Grayscale histograms
"""
	images:represents the source image of type unit8 or float32 provides as a list(example,[gray_img])
	channels:represents the index of channel provided as a list(example,[0] for grayscale images,or [0],[1],[2]for multi-channel images)
	mask:是否使用mask image
	histSize:represents the number of bins provided as a list (for example, [256]).
	ranges:represents the range of intensity values we want to measure (for example, [0,256]).
"""
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

```

### Use histogram to detect image brightness issues

The definition of the brightness of a  grayscale image:

<img src="D:\学期计划\2021_暑假\img\Brightness.png" alt="Brightness" style="zoom:25%;" />

 Here, *I(x, y)* is the tone value for a specific pixel of the image. 

- If the average tone of an image is high (for example, 220), this means that most pixels of the image will be very close to the white color. 
- If the average tone on an image is low (for example, 30) this means that most pixels of the image will be very close to the black color. 

```python
# Add 35 to every pixel on the grayscale image (the result will look lighter) and calculate histogram
M = np.ones(gray_image.shape, dtype="uint8") * 35
added_image = cv2.add(gray_image, M)
hist_added_image = cv2.calcHist([added_image], [0], None, [256], [0, 256])

# Subtract 35 from every pixel (the result will look darker) and calculate histogram
subtracted_image = cv2.subtract(gray_image, M)
hist_subtracted_image = cv2.calcHist([subtracted_image], [0], None, [256], [0, 256])
```

<img src="D:\学期计划\2021_暑假\img\Grayscalehisto.png" alt="Grayscalehisto" style="zoom:33%;" />

### Grayscale histograms with a mask

```python
mask = np.zeros(gray_image.shape[:2], np.uint8)
mask[30:190, 30:190] = 255
hist_mask = cv2.calcHist([gray_image], [0], mask, [256], [0, 256])
```

<img src="D:\学期计划\2021_暑假\img\maskedhistogram.png" alt="maskedhistogram" style="zoom: 50%;" />

原图中存在 some small black and white circles with 0 and 255，因此原图的直方图在0和255存在像素数。

mask之后，黑色外围不被计算进直方图，因此直方图在0和255上像素数为0.

### Color histogram

```python
def hist_color_img(img):
    """Calculates the histogram for a three-channel image"""

    histr = []
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
    return histr

def show_hist_with_matplotlib_rgb(hist, title, pos, color):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(2, 3, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])

    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)
```

<img src="D:\学期计划\2021_暑假\img\colorhist.png" alt="colorhist" style="zoom:50%;" />

### Comparing OpenCV, NumPy, and Matplotlib histogram

- cv2.calcHist() provided by OpenCV
- np.histogram() provided by NumPy
- plt.hist() provided by Matplotlib

<img src="D:\学期计划\2021_暑假\img\compare.png" alt="compare" style="zoom:50%;" />

### Histogram equalization

直方图均衡化的作用：亮度标准化，同时增加图像的对比度

#### Grayscale histogram equalization

```python
image = cv2.imread('lenna.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image_eq = cv2.equalizeHist(gray_image
```

<img src="D:\学期计划\2021_暑假\img\grayscalequal.png" alt="grayscalequal" style="zoom: 50%;" />

由图可知，均衡化后的三幅图，它们的直方图相似

#### Color histogram equalization

对于彩色图像的均衡化直方图，若对每个通道均衡化效果并不好，代码和结果如下：

```python
def equalize_hist_color(img):
    """Equalize the image splitting the image applying cv2.equalizeHist() to each channel and merging the results"""

    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
    eq_channels.append(cv2.equalizeHist(ch))

    eq_image = cv2.merge(eq_channels)
    return eq_image
```

<img src="D:\学期计划\2021_暑假\img\colorhisteqnot.png" alt="colorhisteqnot" style="zoom: 33%;" />

可以看出图片变化很大。这是由于均衡化是在每个通道上独立进行的，但merge时会产生新的色差。

因此，较好的办法是：将BGR图像转化为其他颜色空间（包含 luminance/intensity channel ，例如 Yuv, Lab, HSV, and HSL ），然后只对luminance 通道进行均衡化。

```python
def equalize_hist_color_hsv(img):
    """Equalize the image splitting the image after HSV conversion and applying cv2.equalizeHist()
    to the V channel, merging the channels and convert back to the BGR color space
    """

    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image
```

<img src="D:\学期计划\2021_暑假\img\colorhisteq.png" alt="colorhisteq" style="zoom:33%;" />

### Contrast Limited Adaptive Histogram Equalization

对比度受限的自适应直方图均衡化（CLAHE)的目的：通过限制对比度来解决*自适应直方图均衡化*（AHE）过度放大均匀区域噪声的问题，可提高图像对比度。

```python
## grayscale image
clahe = cv2.createCLAHE(clipLimit=2.0)
# Apply CLAHE to the grayscale image varying clipLimit parameter:
gray_image_clahe = clahe.apply(gray_image)

## color image
def equalize_clahe_color_hsv(img):
    """Equalize the image splitting it after conversion to HSV and applying CLAHE
    to the V channel and merging the channels and convert back to BGR
    """

    cla = cv2.createCLAHE(clipLimit=4.0)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cla.apply(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image
```

<img src="D:\学期计划\2021_暑假\img\CLAHE.png" alt="CLAHE" style="zoom: 33%;" />



并且，使用CLAHE得到的直方图效果较好，因此， **CLAHE is commonly used as the first step in many computer vision applications （ for example, face processing, among others ）**

<img src="D:\学期计划\2021_暑假\img\CLAHEhis.png" alt="CLAHEhis" style="zoom:33%;" />

### Histogram comparison

- cv2.compareHist() 可用于获取两个直方图的匹配程度，但直方图仅显示统计信息，而不显示像素位置，因此不可使用比较图像。
- 常见的图像比较方法：将图像分成一定数量区域（通常是相同大小），计算每个区域直方图，最后将所有直方图连接起来以创建图像的特征表示。

```python
cv2.compareHist(H1, H2, method)
```

method:

-  cv2.HISTCMP_CORREL:  计算两个直方图的相关性，返回[-1, 1]，-1是根本不匹配，1是完全匹配
-  cv2.HISTCMP_CHISQR: 计算两个直方图之间的卡方距离，返回[0，unbounded]，0是完全匹配
-  cv2.HISTCMP_INTERSECT: 计算两个直方图的交集，若直方图被标准化，返回[0，1], 1是完全匹配，0是完全不匹配
-  cv2.HISTCMP_BHATTACHARYYA: 计算两个直方图的巴氏距离，返回[0, 1]，0是完全匹配，1是完全不匹配

## Chapter 07 - Thresholding Techniques

- 图像阈值分割是一种简单而有效的图像分割方法，它根据像素的灰度值对图像进行分割，从而将图像分为前景和背景。
- 图像分割的目的是将一幅图像的表示形式修改为另一种更易于处理的表示形式
- 图像分割通常根据对象的某些属性（颜色、边缘或直方图）从背景中提取对象。如果像素强度小于阈值，最简单的阈值化方法就是将原图像中的每个像素替换为黑色像素；若大于阈值则替换为白色

```python
"""
	src: 目标图片（type为cv2.THRESH_OTSU和cv2.THRESH_TRIANGLE应为单通道图片）
	thresh：阈值
	maxval:最大值（type为cv2.THRESH_BINARY和cv2.THRESH_BINARY_INV才设置）
	type:
"""
ret1,thresh1 = cv2.threshold(src, thresh, maxval, type, dst=None)
```

Different types are as follows: 

- cv2.THRESH_BINARY

<img src="D:\学期计划\2021_暑假\img\BINARY.png" alt="BINARY" style="zoom:8%;" />

- cv2.THRESH_BINARY_INV

<img src="D:\学期计划\2021_暑假\img\BINARYINV.png" alt="BINARYINV" style="zoom:8%;" />

- cv2.THRESH_TRUNC

<img src="D:\学期计划\2021_暑假\img\TRUNC.png" alt="TRUNC" style="zoom:8%;" />

- cv2.THRESH_TOZERO

<img src="D:\学期计划\2021_暑假\img\TOZERO.png" alt="TOZERO" style="zoom:8%;" />

- cv2.THRESH_TOZERO_INV

<img src="D:\学期计划\2021_暑假\img\TOZEROINV.png" alt="TOZEROINV" style="zoom:8%;" />

- cv2.THRESH_OTSU：与上述某一个结合，计算并返回最佳阈值

- cv2.THRESH_TRIANGLE：同上

### Adaptive thresholding

<img src="D:\学期计划\2021_暑假\img\thresholdingea.png" alt="thresholdingea" style="zoom: 50%;" />



- 上述是利用cv2.THRESH_BINARY方法，采用不同阈值获取的。如果阈值很低，则阈值图像中缺少一些数字；如果阈值高，则有一些数字被黑色像素遮挡。再加上光照因素，建立全局阈值是困难的。
- 自适应阈值算法可以在一定程度上解决上述问题

```python
"""
	src:目标图像
	maxValue：最大值
	adaptiveMethod：自适应阈值算法
	thresholdType：cv2.THRESH_BINARY 或 cv2.THRESH_BINARY_INV
	blockSize：设置用于计算像素阈值的邻域区域的大小，可以为3，5，7....
	C：从平均值或加权平均值中减去的常数（取决于adaptiveMethod）
"""
adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst
```

adaptiveMethod：

-  cv2.ADAPTIVE_THRESH_MEAN_C:  (x, y)处的阈值T(x, y) = x的邻域的平均值 - C
-  cv2.ADAPTIVE_THRESH_GAUSSIAN_C:  (x, y)处的阈值T(x, y) = x的邻域的加权平均值 - C



thresholdType：

- cv2.THRESH_BINARY

<img src="D:\学期计划\2021_暑假\img\ADBINART.png" alt="ADBINART" style="zoom:8%;" />

- cv2.THRESH_BINARY_INV: 上述取反

```python
## 自适应阈值进行数字检测
# 先滤波，a bilateral filter is applied because we want to keep the edges sharp
gray_image = cv2.bilateralFilter(gray_image, 15, 25, 25)
# 自适应阈值分割
thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
```

<img src="D:\学期计划\2021_暑假\img\ada.png" alt="ada" style="zoom:50%;" />

### Otsu's thresholding algorithm

- 虽然自适应阈值算法不用确定阈值，但*blockSize*和*C*仍是需要通过实验确定的参数。
- Otsu's thresholding algorithm 可以不确定参数，并且是处理 *双峰图像*  的好方法

```python
ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
# cv2.THRESH_OTSU可以与cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, and cv2.THRESH_TOZERO_INV组合
# ret1: thresholded value
# th1: thresholded image
```

<img src="D:\学期计划\2021_暑假\img\Otsu.png" alt="Otsu" style="zoom:50%;" />

- 对于带有噪声的图片，若直接使用阈值算法，结果仍带有噪音。利用过滤方法后（此处是Gaussian filter），结果较好。
- 注意顺序：若阈值操作后再使用过滤方法，效果如下。上述是先过滤再阈值操作。

- <img src="D:\学期计划\2021_暑假\img\shift.PNG" alt="shift" style="zoom:50%;" />

- 由直方图可知，该图为双峰图像。

### The triangle binarization algorithm

Another automatic thresholding algorithm ，使用方法同上

```python
ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
```

<img src="D:\学期计划\2021_暑假\img\tran.png" alt="tran" style="zoom:50%;" />

### Thresholding color images

```python
## 两种方法产生的结果相同
## 但对于BGR图片使用阈值算法可能会产生奇怪的结果
# way 1
ret1, thresh1 = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
# way 2
(b, g, r) = cv2.split(image)
ret2, thresh2 = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
ret4, thresh4 = cv2.threshold(r, 150, 255, cv2.THRESH_BINARY)
bgr_thresh = cv2.merge((thresh2, thresh3, thresh4))
```

### Thresholding with scikit-image

```python
from skimage.filters import (threshold_otsu, threshold_triangle, threshold_niblack, threshold_sauvola)
from skimage import img_as_ubyte

# Trying Otsu's scikit-image algorithm:
thresh_otsu = threshold_otsu(gray_image)
binary_otsu = gray_image > thresh_otsu
binary_otsu = img_as_ubyte(binary_otsu)  # Convert to uint8 data type，易于显示

# Trying Niblack's scikit-image algorithm:
thresh_niblack = threshold_niblack(gray_image, window_size=25, k=0.8)
binary_niblack = gray_image > thresh_niblack
binary_niblack = img_as_ubyte(binary_niblack)

# Trying Sauvola's scikit-image algorithm:
thresh_sauvola = threshold_sauvola(gray_image, window_size=25)
binary_sauvola = gray_image > thresh_sauvola
binary_sauvola = img_as_ubyte(binary_sauvola)

# Trying triangle scikit-image algorithm:
thresh_triangle = threshold_triangle(gray_image)
binary_triangle = gray_image > thresh_triangle
binary_triangle = img_as_ubyte(binary_triangle)
```

-  Otsu and triangle are global thresholding techniques, while Niblack and Sauvola are local thresholding techniques. 
-  当背景不均匀时，局部阈值被认为是一种较好的方法
- 更多阈值算法可在文档中查找 http://scikit-image.org/docs/dev/api/api.html. 



## Chapter 08 - Contour Detection, Filtering, and Drawing

- 轮廓（Contour）：定义图像中对象边界的一系列点。轮廓线传递了物体边界的关键信息和物体形状
- 可以作为image descriptors（例如SIFT、Fourier descriptors）的基础
- 可用于形状分析和目标检测识别
- 有时候直接对真实照片进行轮廓检测，由于真实照片的轮廓包含很多点，不利于debug，因此常用以下函数进行应用前的debug

```python
def get_one_contour():
    """Returns a 'fixed' contour"""

    cnts = [np.array(
        [[[600, 320]], [[563, 460]], [[460, 562]], [[320, 600]], [[180, 563]], [[78, 460]], [[40, 320]], [[77, 180]],
         [[179, 78]], [[319, 40]], [[459, 77]], [[562, 179]]], dtype=np.int32)]
    return cnts

def draw_contour_points(img, cnts, color):
    """Draw all points from a list of contours"""

    for cnt in cnts:
        squeeze = np.squeeze(cnt)
        for p in squeeze:
            # print(p)
            p = array_to_tuple(p)
            # print(p)
            cv2.circle(img, p, 10, color, -1)

    return img


def draw_contour_outline(img, cnts, color, thickness=1):
    """Draws contours outlines of each contour"""

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)
        
# Get a sample contours:
contours = get_one_contour()
print("contour shape: '{}'".format(contours[0].shape))
print("'detected' contours: '{}' ".format(len(contours)))
```

### Detect contours

函数介绍：

```python
## detect contours in binary images 
cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
```

- 获取binary images 的方法：1）图片本身就是 2）threshold operations后
- 此函数返回：检测到的轮廓，每个轮廓包含定义边界的所有检测点
- mode：
  -  cv2.RETR_EXTERNAL ：仅输出外部轮廓
  -  cv2.RETR_LIST ：输出所有轮廓，没有任何层次关系
  -  cv2.RETR_TREE ：通过建立层次关系输出所有轮廓
- hierarchy：包含层次关系。例如对第i个轮廓 contours[i]， hierarchy[i, j]（j∈[0, 3]）解释如下：
  - hierarchy[i, 0]： Index of the next contour at the same hierarchical level 
  - hierarchy[i, 1]： Index of the previous contour at the same hierarchical level 
  - hierarchy[i, 2]： Index of the first child contour 
  - hierarchy[i, 3]： Index of the parent contour 
  - hierarchy[i, 3] = 负数说明不存上述关系
- method：
  -  cv2.CHAIN_APPROX_NONE ：不执行压缩存储所有边界点
  -  cv2.CHAIN_APPROX_SIMPLE ：压缩轮廓的水平、垂线和对角线，只保留断点，如矩形的四个点
  -  cv2.CHAIN_APPROX_TC89_L1：根据显著性进行压缩
  -  cv2.CHAIN_APPROX_TC89_KCOS ：同上

```python
# Load the image and convert it to grayscale:
image = build_sample_image_2()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get a ginary image:
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
image_approx_none = image.copy()

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Draw the contours in the previously created images:
draw_contour_points(image_approx_none, contours, (255, 255, 255))
```

### Image moments(图像矩)

- 矩：函数表达式，具有定量度量作用
- 图像矩：图像像素强度的加权平均值或某种函数，可以描述检测到的轮廓的某些特性（例如，物体的质心或面积等）

```python
retval = cv.moments(array[, binaryImage])

# 例1
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
M = cv2.moments(contours[0])
# 输出 因为mu00=m00, nu00=1, nu10=mu10=mu01=mu10=0，所以没有 输出
{'m00': 235283.0, 'm10': 75282991.16666666, 'm01': 75279680.83333333, 'm20': 28496148988.333332, 'm11': 24089788592.25, 'm02': 28492341886.0, 'm30': 11939291123446.25, 'm21': 9118893653727.8, 'm12': 9117775940692.967, 'm03': 11936167227424.852, 'mu20': 4408013598.184406, 'mu11': 2712402.277420044, 'mu02': 4406324849.628765, 'mu30': 595042037.7265625, 'mu21': -292162222.4824219, 'mu12': -592577546.1586914, 'mu03': 294852334.5449219, 'nu20': 0.07962727021646843, 'nu11': 4.8997396280458296e-05, 'nu02': 0.07959676431294238, 'nu30': 2.2160077537124397e-05, 'nu21': -1.0880470778779139e-05, 'nu12': -2.2068296922023203e-05, 'nu03': 1.0980653771087236e-05}

# 例2
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
M = cv2.moments(thresh, True)
```

-  The spatial moments （空间矩）：

<img src="D:\学期计划\2021_暑假\img\The spatial moments.png" alt="The spatial moments" style="zoom:8%;" />

-  The central moments （中心矩）：

<img src="D:\学期计划\2021_暑假\img\The central moments.png" alt="The central moments" style="zoom:8%;" />

<img src="D:\学期计划\2021_暑假\img\mean.png" alt="mean" style="zoom:8%;" />

-  Normalized central moments 

<img src="D:\学期计划\2021_暑假\img\Normalized central moments.png" alt="Normalized central moments" style="zoom:8%;" />

根据定义，中心距对于平移是不变的，因此适合描述物体的形状。

中心矩和空间矩依赖于对象的大小，不是比例不变的。

归一化中心矩具有平移和缩放不变性。

#### Some object features based on moments

```python
M = cv2.moments(contours[0])

## 1.计算轮廓面积
cv2.contourArea(contours[0])
# Or
M['m00']

## 2.计算轮廓质心
print("center X : '{}'".format(round(M['m10'] / M['m00'])))
print("center Y : '{}'".format(round(M['m01'] / M['m00']))
      
## 3.计算轮廓的圆度（roundness - 一个轮廓接近园轮廓的程度）
def roundness(contour, moments):
    """Calculates the roundness of a contour"""

    length = cv2.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * np.pi)
    return k
      
## 4.计算轮廓的偏心率（eccentricity - 轮廓的伸长程度）
def eccentricity_from_ellipse(contour):
    """Calculates the eccentricity fitting an ellipse from a contour"""

    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)

    a = ma / 2
    b = MA / 2

    ecc = np.sqrt(a ** 2 - b ** 2) / a
    return ecc

def eccentricity_from_moments(moments):
    """Calculates the eccentricity from the moments of the contour"""

    a1 = (moments['mu20'] + moments['mu02']) / 2
    a2 = np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2) / 2
    ecc = np.sqrt(1 - (a1 - a2) / (a1 + a2))
    return ecc
      
## 5.轮廓边界矩形的宽高比
def aspect_ratio(contour):
    """Returns the aspect ratio of the contour based on the dimensions of the bounding rect"""

    x, y, w, h = cv2.boundingRect(contour)
    res = float(w) / h
    return res
```

#### Hu moment invariants

Hu moment 具有平移、缩放和旋转不变性

```python
"""
	m: the moments calculated with cv2.moments()
	return hu: the seven Hu invariant moments
"""
cv2.HuMoments(m[, hu]) → hu

```

#### Zernike moments

the *mahotas* package provides the *zernike_moments()* function 

```python
mahotas.features.zernike_moments(im, radius, degree=8, cm={center_of_mass(im)})
```

### More functionality related to contours

<img src="D:\学期计划\2021_暑假\img\relatecon.png" alt="relatecon" style="zoom:50%;" />

```python
# 1.minimal bounding rectangle
x, y, w, h = cv2.boundingRect(contours[0])

# 2.minimal rotated rectangle
rotated_rect = cv2.minAreaRect(contours[0])

# 3.获取旋转矩阵的四个顶点
box = cv2.boxPoints(rotated_rect)

# 4.the minimal circle (return the center and radius)
(x,y),radius = cv2.minEnclosingCircle(contours[0])

# 5.the ellipse with the minimum least square errors
ellipse = cv2.fitEllipse(contours[0])

# 6.计算给定轮廓的轮廓近似值（基于给定的精度）
approx = cv2.approxPolyDP(contours[0], epsilon, True)

# 7.calculates the four extreme points 
def extreme_points(contour):
    """Returns extreme points of the contour"""

    index_min_x = contour[:, :, 0].argmin()
    index_min_y = contour[:, :, 1].argmin()
    index_max_x = contour[:, :, 0].argmax()
    index_max_y = contour[:, :, 1].argmax()

    extreme_left = tuple(contour[index_min_x][0])
    extreme_right = tuple(contour[index_max_x][0])
    extreme_top = tuple(contour[index_min_y][0])
    extreme_bottom = tuple(contour[index_max_y][0])

    return extreme_left, extreme_right, extreme_top, extreme_bottom

```

### contours 排序

```python
def sort_contours_size(cnts):
    """ Sort contours based on the size"""

    cnts_sizes = [cv2.contourArea(contour) for contour in cnts]
    (cnts_sizes, cnts) = zip(*sorted(zip(cnts_sizes, cnts)))
    return cnts_sizes, cnts
```

代码解释：

```python
coordinate = ['x', 'y', 'z']
value = [5, 4, 3]
result = zip(coordinate, value)
print(list(result))
c, v =  zip(*zip(coordinate, value))
print('c =', c)
print('v =', v)
************************************** 结果 **************************************
[('x', 5), ('y', 4), ('z', 3)]
c = ('x', 'y', 'z')
v = (5, 4, 3)


coordinate = ['x', 'y', 'z']
value = [5, 4, 3]
print(sorted(zip(value, coordinate)))
c, v = zip(*sorted(zip(value, coordinate)))
print('c =', c)
print('v =', v)
************************************** 结果 **************************************
[(3, 'z'), (4, 'y'), (5, 'x')]
c = (3, 4, 5)
v = ('z', 'y', 'x')
```

### contours 形状识别

```python
def detect_shape(contour):
    """Returns the shape (e.g. 'triangle', 'square') from the contour"""

    detected_shape = '-----'

    # Calculate perimeter of the contour: 周长
    perimeter = cv2.arcLength(contour, True)

    # Get a contour approximation:
    contour_approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

    # Check if the number of vertices is 3. In this case, the contour is a triangle
    if len(contour_approx) == 3:
        detected_shape = 'triangle'

    # Check if the number of vertices is 4. In this case, the contour is a square/rectangle
    elif len(contour_approx) == 4:

        # We calculate the aspect ration from the bounding rect:
        x, y, width, height = cv2.boundingRect(contour_approx)
        aspect_ratio = float(width) / height

        # A square has an aspect ratio close to 1 (comparison chaining is used):
        if 0.90 < aspect_ratio < 1.10:
            detected_shape = "square"
        else:
            detected_shape = "rectangle"

    # Check if the number of vertices is 5. In this case, the contour is a pentagon
    elif len(contour_approx) == 5:
        detected_shape = "pentagon"

    # Check if the number of vertices is 6. In this case, the contour is a hexagon
    elif len(contour_approx) == 6:
        detected_shape = "hexagon"

    # The shape as more than 6 vertices. In this example, we assume that is a circle
    else:
        detected_shape = "circle"

    # return the name of the shape and the found vertices
    return detected_shape, contour_approx
```

<img src="D:\学期计划\2021_暑假\img\1.png" alt="1" style="zoom:50%;" />

### Matching contours

 Hu moment invariants can be used for both object matching and recognition. 

In this section, we are going to see how to match contours based on Hu moment invariants.  

```python
cv2.matchShapes()
```

<img src="D:\学期计划\2021_暑假\img\2.png" alt="2" style="zoom:50%;" />



## Chapter 09 - Augmented Reality

- Location-based augmented reality：通过读取多个传感器数据来检测用户的位置和方向
- recognition-based augmented reality ：通过图像处理技术推断。从图像中获取摄像机的姿态需要找到环境中的已知点之间的对应关系及其对应的摄像机投影，主要方法：
  - 基于标记的姿态估计：使用平面标记（主要是正方形标记）从四个角点计算相机的姿态
  - 基于无标记的姿态估计：使用图像中自然存在的对象进行姿态估计。例如PnP问题求解

### Markerless-based augmented reality

#### Feature detection

- 特征 - Feature : a small pitch in the image. 具有旋转、缩放和照明不变性（尽可能），这样就可以从同一场景不同视角的不同图像中检测出相同的特征。
- 好的特征应是：
  - Repeatable and precise（同一物体不同图像应提取相同的特征）
  - Distinctive to the image（具有不同结构的图像将不具有此特征）
-  algorithms and techniques to detect features in images ：
  - **Harris Corner Detection**
  - **Shi-Tomasi Corner Detection**
  - **Scale Invariant Feature Transform** (**SIFT**)
  - **Speeded-Up Robust Features** (**SURF**)
  - **Features from Accelerated Segment Test** (**FAST**)
  - **Binary Robust Independent Elementary Features**(**BRIEF**)
  - **Oriented FAST and Rotated BRIEF** (**ORB**)

```python
## ORB feature 的使用
# Initiate ORB detector:
orb = cv2.ORB_create()

# Detect the keypoints using ORB:
keypoints = orb.detect(image, None)

# Compute the descriptors of the detected keypoints:
keypoints, descriptors = orb.compute(image, keypoints)

# Draw detected keypoints:
image_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 255), flags=0)
```

#### Feature matching

 OpenCV provides two matchers ：

-  **Brute-Force** (**BF**) **matcher**
-  **Fast Library for Approximate Nearest Neighbors** (**FLANN**) **matcher**

```python
## BF匹配实例
# Initiate ORB detector:
orb = cv2.ORB_create()

# Detect the keypoints and compute the descriptors with ORB:
keypoints_1, descriptors_1 = orb.detectAndCompute(image_query, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(image_scene, None)

# Create BFMatcher object
# First parameter sets the distance measurement (by default it is cv2.NORM_L2)
# The second parameter crossCheck (which is False by default) can be set to True in order to return only
# consistent pairs in the matching process (the two features in both sets should match each other)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors:
bf_matches = bf_matcher.match(descriptors_1, descriptors_2)

# 按照距离升序排序
bf_matches = sorted(bf_matches, key=lambda x: x.distance)

# Draw first 20 matches:
result = cv2.drawMatches(image_query, keypoints_1, image_scene, keypoints_2, bf_matches[:20], None,matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)
```

<img src="D:\学期计划\2021_暑假\img\3.png" alt="3" style="zoom:50%;" />

#### Feature matching and homography computation to find objects

图像匹配后，下一步就是使用*cv2.findHomography()*函数在两个图像中匹配的关键点位置之间找到透视转换，即 homography matrix 。

```python
# Initiate ORB detector:
orb = cv2.ORB_create()

# Detect the keypoints and compute the descriptors with ORB:
keypoints_1, descriptors_1 = orb.detectAndCompute(image_query, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(image_scene, None)

# Create BFMatcher object
# First parameter sets the distance measurement (by default it is cv2.NORM_L2)
# The second parameter crossCheck (which is False by default) can be set to True in order to return only
# consistent pairs in the matching process (the two features in both sets should match each other)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors:
bf_matches = bf_matcher.match(descriptors_1, descriptors_2)

# Sort the matches in the order of their distance:
bf_matches = sorted(bf_matches, key=lambda x: x.distance)
best_matches = bf_matches[:40]

# Extract the matched keypoints:
pts_src = np.float32([keypoints_1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
pts_dst = np.float32([keypoints_2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Find homography matrix:
M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

# Get the corner coordinates of the 'query' image:
h, w = image_query.shape[:2]
pts_corners_src = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

# Perform perspective transform using the previously calculated matrix and the corners of the 'query' image
# to get the corners of the 'detected' object in the 'scene' image:
pts_corners_dst = cv2.perspectiveTransform(pts_corners_src, M)

# Draw corners of the detected object:
img_obj = cv2.polylines(image_scene, [np.int32(pts_corners_dst)], True, (0, 255, 255), 10)

# Draw matches:
img_matching = cv2.drawMatches(image_query, keypoints_1, img_obj, keypoints_2, best_matches, None,matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)
```

<img src="D:\学期计划\2021_暑假\img\4.png" alt="4" style="zoom:50%;" />



### Marker-based augmented reality

#### Creating markers and dictionaries

The first step to consider when creating your marker-based augmented reality application is to print the markers to use. 

```python
# First, create the dictionary object
# DICT_4X4_50 = 0, DICT_4X4_100 = 1, DICT_4X4_250 = 2, DICT_4X4_1000 = 3, 
# DICT_5X5_50 = 4, DICT_5X5_100 = 5, DICT_5X5_250 = 6, DICT_5X5_1000 = 7, 
# DICT_6X6_50 = 8, DICT_6X6_100 = 9, DICT_6X6_250 = 10, DICT_6X6_1000 = 11, 
# DICT_7X7_50 = 12, DICT_7X7_100 = 13, DICT_7X7_250 = 14, and DICT_7X7_1000 = 15.
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

# Second, draw a marker using 'cv2.aruco.drawMarker()'.
aruco_marker_1 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=1)
```

#### Detecting markers

```python
# detect markers
corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)

# Draw detected markers:
frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))
# Draw rejected markers:
frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejected_corners, borderColor=(0, 0, 255))
```

<img src="D:\学期计划\2021_暑假\img\5.png" alt="5" style="zoom:50%;" />

#### Camera calibration

```python
alibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs

"""
Aruco camera calibration
"""

# Import required packages:
import time
import cv2
import numpy as np
import pickle

# Create dictionary and board object:
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
board = cv2.aruco.CharucoBoard_create(3, 3, .025, .0125, dictionary)

# Create board image to be used in the calibration process:
image_board = board.draw((200 * 3, 200 * 3))

# Write calibration board image:
cv2.imwrite('charuco.png', image_board)

# Create VideoCapture object:
cap = cv2.VideoCapture(0)

all_corners = []
all_ids = []
counter = 0
for i in range(300):

    # Read frame from the webcam:
    ret, frame = cap.read()

    # Convert to grayscale:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers:
    res = cv2.aruco.detectMarkers(gray, dictionary)

    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and counter % 3 == 0:
            all_corners.append(res2[1])
            all_ids.append(res2[2])

        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    counter += 1

# Calibration can fail for many reasons:
try:
    cal = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape, None, None)
except:
    cap.release()
    print("Calibration could not be done ...")

# Get the calibration result:
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal

# Save the camera parameters:
f = open('calibration2.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs), f)
f.close()

# Release everything:
cap.release()
cv2.destroyAllWindows()

```

#### Camera pose estimation

```python
cv.aruco.estimatePoseSingleMarkers( corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs[, _objPoints]]] ) ->  rvecs, tvecs, _objPoints
```

<img src="D:\学期计划\2021_暑假\img\6.png" alt="6" style="zoom:50%;" />

#### Camera pose estimation and basic augmentation

利用增强现实应用程序覆盖一些图像、形状或三维模型

结果如下：

<img src="D:\学期计划\2021_暑假\img\7.png" alt="7" style="zoom:50%;" />

#### Camera pose estimation and more advanced augmentation

<img src="D:\学期计划\2021_暑假\img\8.png" alt="8" style="zoom:50%;" />

### Snapchat-based augmented reality

#### Snapchat-based augmented reality OpenCV moustache overlay

1. 检测图像中的所以人脸
2. 迭代图像中所以检测到的人脸，在其区域内搜索鼻子
3. 检测到鼻子后，调整想要覆盖胡子的区域（虽然图中有两个检测到的鼻子，但只有一个覆盖，因为执行检查以了解检测到的鼻子是否有效）

<img src="D:\学期计划\2021_暑假\img\9.png" alt="9" style="zoom:50%;" />

#### Snapchat-based augmented reality OpenCV glasses overlay

 overlay a pair of glasses on the eyes region of the detected face.  

<img src="D:\学期计划\2021_暑假\img\10.png" alt="10" style="zoom:50%;" />

### 二维码检测

```python
# create the QR code detector
qr_code_detector = cv2.QRCodeDetector()

data, bbox, rectified_qr_code = qr_code_detector.detectAndDecode(image)
```

<img src="D:\学期计划\2021_暑假\img\11.png" alt="11" style="zoom:50%;" />

## Chapter 10 - Machine Learning with OpenCV

<img src="D:\学期计划\2021_暑假\img\12.png" alt="12" style="zoom:40%;" />

-  Supervised learning ：使用一组样本训练，每个样本都有相应输出值
  - 偏差和方差：
    - 偏差-bias：欠拟合的标志
    - 方差-variance：过拟合的标志
  - 模型复杂性和训练数据量
  - 输入的维度：高维度可能会使学习变困难
  - 噪声：一般在训练前检测和去除噪声训练样本
- Unsupervised machine learning：样本没有标记输出
  - 目标：对样本集合中的潜在结构或分布进行建模和推断
  - 聚类和降维是两种常用算法
-  semi-supervised learning ：同时使用标记和非标记数据

### k-means clustering

- 目标：将n个样本划分成k个聚类，每个样本都属于具有最近均值的聚类
- 函数： cv2.kmeans() 
- 例子：Color quantization - 颜色量化（减少图像中颜色数量的过程）

<img src="D:\学期计划\2021_暑假\img\13.png" alt="13" style="zoom:50%;" />

### k-nearest neighbor

- 可用于分类和回归
- 过程：训练阶段：KNN存储所有训练样本的特征向量和类标签。分类阶段：未标记向量被分类为最接近要分类的k个训练样本中最频繁的类标签
- 函数： cv2.ml.KNearest_create()  ， train() 和 findNearest()  方法
- 例子：手写数字识别

### Support vector machine

- 通过将训练样本按其指定的类别进行最佳分离，在高维空间中构造一个或一组超平面，如图绿线

<img src="D:\学期计划\2021_暑假\img\14.png" alt="14" style="zoom:50%;" />

- 函数： cv2.ml.SVM_create() 
- 例子：手写数字识别

## Chapter 11 - Face Detection, Tracking, and Recognition

![15](D:\学期计划\2021_暑假\img\15.png)

- Face detection ：a  specific case of object detection ; find both the locations and size of all the faces in an image
- Facial landmarks detection ：a special  case of landmarks detection ; locate the main landmarks in a face
- Face tracking : a special case of object tracking ; find both the locations and sizes of all the moving faces in a video
- Face recognition : a special case of object recognition , where a person is identified or verified from an image or video using the information extracted from the face:
  - Face identification (1:N)
  - Face verification (1:1)
- 常用库：opencv,  dlib, face_recognition, cvlib

### Face detection

#### Face detection with OpenCV

Two approached for face detection:

- Haar cascade based face detectors
- Deep learning-based face detectors

 In this sense, OpenCV provides four cascade classifiers to use for (frontal) face detection: 

-  haarcascade_frontalface_alt.xml (***FA1***) 
-  haarcascade_frontalface_alt2.xml (***FA2***)
-  haarcascade_frontalface_alt_tree.xml (***FAT***)
-  haarcascade_frontalface_default.xml (***FD***)

```python
"""
Face detection using haar feature-based cascade classifiers
"""

# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return image


# Load image and convert to grayscale:
img = cv2.imread("test_face_detection.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load cascade classifiers:
cas_alt2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
cas_default = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detect faces:
faces_alt2 = cas_alt2.detectMultiScale(gray)
faces_default = cas_default.detectMultiScale(gray)
retval, faces_haar_alt2 = cv2.face.getFacesHAAR(img, "haarcascade_frontalface_alt2.xml")
faces_haar_alt2 = np.squeeze(faces_haar_alt2)
retval, faces_haar_default = cv2.face.getFacesHAAR(img, "haarcascade_frontalface_default.xml")
faces_haar_default = np.squeeze(faces_haar_default)

# Draw face detections:
img_faces_alt2 = show_detection(img.copy(), faces_alt2)
img_faces_default = show_detection(img.copy(), faces_default)
img_faces_haar_alt2 = show_detection(img.copy(), faces_haar_alt2)
img_faces_haar_default = show_detection(img.copy(), faces_haar_default)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 8))
plt.suptitle("Face detection using haar feature-based cascade classifiers", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces_alt2, "detectMultiScale(frontalface_alt2): " + str(len(faces_alt2)), 1)
show_img_with_matplotlib(img_faces_default, "detectMultiScale(frontalface_default): " + str(len(faces_default)), 2)
show_img_with_matplotlib(img_faces_haar_alt2, "getFacesHAAR(frontalface_alt2): " + str(len(faces_haar_alt2)), 3)
show_img_with_matplotlib(img_faces_haar_default, "getFacesHAAR(frontalface_default): " + str(len(faces_haar_default)),
                         4)

# Show the Figure:
plt.show()
```

- OpenCV 不仅提供人脸检测，还提供其他的，例如猫脸检测

- 另外，也提供 deep learning-based face detector，有两种models

  - Face detector (FP16)
  - Face detector (UINT8)

  ```python
  """
  Face detection using OpenCV DNN face detector
  """
  
  # Import required packages:
  import cv2
  import numpy as np
  from matplotlib import pyplot as plt
  
  
  def show_img_with_matplotlib(color_img, title, pos):
      """Shows an image using matplotlib capabilities"""
  
      img_RGB = color_img[:, :, ::-1]
  
      ax = plt.subplot(1, 1, pos)
      plt.imshow(img_RGB)
      plt.title(title)
      plt.axis('off')
  
  
  # Load pre-trained model:
  net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
  # net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
  
  # Load image:
  image = cv2.imread("test_face_detection.jpg")
  
  # Get dimensions of the input image (to be used later):
  (h, w) = image.shape[:2]
  
  # Create 4-dimensional blob from image:
  blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)
  
  # Set the blob as input and obtain the detections:
  net.setInput(blob)
  detections = net.forward()
  
  # Initialize the number of detected faces counter detected_faces:
  detected_faces = 0
  
  # Iterate over all detections:
  for i in range(0, detections.shape[2]):
      # Get the confidence (probability) of the current detection:
      confidence = detections[0, 0, i, 2]
  
      # Only consider detections if confidence is greater than a fixed minimum confidence:
      if confidence > 0.7:
          # Increment the number of detected faces:
          detected_faces += 1
          # Get the coordinates of the current detection:
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")
  
          # Draw the detection and the confidence:
          text = "{:.3f}%".format(confidence * 100)
          y = startY - 10 if startY - 10 > 10 else startY + 10
          cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
          cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
  
  # Create the dimensions of the figure and set title:
  fig = plt.figure(figsize=(10, 5))
  plt.suptitle("Face detection using OpenCV DNN face detector", fontsize=14, fontweight='bold')
  fig.patch.set_facecolor('silver')
  
  # Plot the images:
  show_img_with_matplotlib(image, "DNN face detector: " + str(detected_faces), 1)
  
  # Show the Figure:
  plt.show()
  
  ```

  

#### Face detection with dlib

- Use only a few training images to easily train own object detectors:  http://dlib.net/train_object_detector.py.html 

- HOG方法人脸检测

  ```python
  """
  Face detection using dlib frontal face detector, which is based on Histogram of Oriented Gradients (HOG) features
  and a linear classifier in a sliding window detection approach
  """
  
  # Import required packages:
  import cv2
  import dlib
  from matplotlib import pyplot as plt
  
  
  def show_img_with_matplotlib(color_img, title, pos):
      """Shows an image using matplotlib capabilities"""
  
      img_RGB = color_img[:, :, ::-1]
  
      ax = plt.subplot(1, 2, pos)
      plt.imshow(img_RGB)
      plt.title(title)
      plt.axis('off')
  
  
  def show_detection(image, faces):
      """Draws a rectangle over each detected face"""
  
      for face in faces:
          cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 10)
      return image
  
  
  # Load image and convert to grayscale:
  img = cv2.imread("test_face_detection.jpg")
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # Load frontal face detector from dlib:
  detector = dlib.get_frontal_face_detector()
  
  # Detect faces:
  rects_1 = detector(gray, 0)
  rects_2 = detector(gray, 1)
  
  # Draw face detections:
  img_faces_1 = show_detection(img.copy(), rects_1)
  img_faces_2 = show_detection(img.copy(), rects_2)
  
  # Create the dimensions of the figure and set title:
  fig = plt.figure(figsize=(10, 4))
  plt.suptitle("Face detection using dlib frontal face detector", fontsize=14, fontweight='bold')
  fig.patch.set_facecolor('silver')
  
  # Plot the images:
  show_img_with_matplotlib(img_faces_1, "detector(gray, 0): " + str(len(rects_1)), 1)
  show_img_with_matplotlib(img_faces_2, "detector(gray, 1): " + str(len(rects_2)), 2)
  
  # Show the Figure:
  plt.show()
  
  ```

- CNN方法

  ```python
  """
  Face detection using dlib CNN face detector using a pre-trained model (712 KB)
  from http://dlib.net/files/mmod_human_face_detector.dat.bz2.
  """
  
  # Import required packages:
  import cv2
  import dlib
  from matplotlib import pyplot as plt
  
  
  def show_img_with_matplotlib(color_img, title, pos):
      """Shows an image using matplotlib capabilities"""
  
      img_RGB = color_img[:, :, ::-1]
  
      ax = plt.subplot(1, 1, pos)
      plt.imshow(img_RGB)
      plt.title(title)
      plt.axis('off')
  
  
  def show_detection(image, faces):
      """Draws a rectangle over each detected face"""
  
      # faces contains a list of mmod_rectangle objects
      # The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score
      # Therefore, we iterate over the detected mmod_rectangle objects accessing dlib.rect to draw the rectangle
  
      for face in faces:
          cv2.rectangle(image, (face.rect.left(), face.rect.top()), (face.rect.right(), face.rect.bottom()), (255, 0, 0),
                        10)
      return image
  
  
  # Load CNN detector from dlib:
  cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
  
  # Load image and convert to grayscale:
  img = cv2.imread("test_face_detection.jpg")
  
  # Resize the image to attain reasonable speed:
  # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
  
  # Detect faces:
  rects = cnn_face_detector(img, 0)
  
  # Draw face detections:
  img_faces = show_detection(img.copy(), rects)
  
  # Create the dimensions of the figure and set title:
  fig = plt.figure(figsize=(10, 5))
  plt.suptitle("Face detection using dlib CNN face detector", fontsize=14, fontweight='bold')
  fig.patch.set_facecolor('silver')
  
  # Plot the images:
  show_img_with_matplotlib(img_faces, "cnn_face_detector(img, 0): " + str(len(rects)), 1)
  
  # Show the Figure:
  plt.show()
  
  ```

#### Face detection with face_recognition

- HOG方法

  ```python
  """
  Face detection using face_recognition HOG face detector (internally calls dlib HOG face detector)
  """
  
  # Import required packages:
  import cv2
  import face_recognition
  from matplotlib import pyplot as plt
  
  
  def show_img_with_matplotlib(color_img, title, pos):
      """Shows an image using matplotlib capabilities"""
  
      img_RGB = color_img[:, :, ::-1]
  
      ax = plt.subplot(1, 2, pos)
      plt.imshow(img_RGB)
      plt.title(title)
      plt.axis('off')
  
  
  def show_detection(image, faces):
      """Draws a rectangle over each detected face"""
  
      for face in faces:
          top, right, bottom, left = face
          cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 10)
      return image
  
  
  # Load image:
  img = cv2.imread("test_face_detection.jpg")
  
  # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
  rgb = img[:, :, ::-1]
  
  # Perform face detection using face_recognition (internally calls dlib HOG face detector):
  rects_1 = face_recognition.face_locations(rgb, 0, "hog")
  rects_2 = face_recognition.face_locations(rgb, 1, "hog")
  
  # Draw face detections:
  img_faces_1 = show_detection(img.copy(), rects_1)
  img_faces_2 = show_detection(img.copy(), rects_2)
  
  # Create the dimensions of the figure and set title:
  fig = plt.figure(figsize=(10, 4))
  plt.suptitle("Face detection using face_recognition frontal face detector", fontsize=14, fontweight='bold')
  fig.patch.set_facecolor('silver')
  
  # Plot the images:
  show_img_with_matplotlib(img_faces_1, "face_locations(rgb, 0, hog): " + str(len(rects_1)), 1)
  show_img_with_matplotlib(img_faces_2, "face_locations(rgb, 1, hog): " + str(len(rects_2)), 2)
  
  # Show the Figure:
  plt.show()
  
  ```

  

- CNN方法

```python
"""
Face detection using face_recognition CNN face detector (internally calls dlib CNN face detector)
"""

# Import required packages:
import cv2
import face_recognition
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""

    for face in faces:
        top, right, bottom, left = face
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    return image


# Load image and resize:
img = cv2.imread("test_face_detection.jpg")
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
rgb = img[:, :, ::-1]

# Perform face detection using face_recognition (internally using dlib CNN face detection):
rects_1 = face_recognition.face_locations(rgb, 0, "cnn")
rects_2 = face_recognition.face_locations(rgb, 1, "cnn")

# Draw face detections:
img_faces_1 = show_detection(img.copy(), rects_1)
img_faces_2 = show_detection(img.copy(), rects_2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 4))
plt.suptitle("Face detection using face_recognition CNN face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces_1, "face_locations(rgb, 0, cnn): " + str(len(rects_1)), 1)
show_img_with_matplotlib(img_faces_2, "face_locations(rgb, 1, cnn): " + str(len(rects_2)), 2)

# Show the Figure:
plt.show()

```

#### Face detection with cvlib

```python
"""
Face detection using clib face detector (uses DNN OpenCV face detector)
"""

# Import required packages:
import cv2
import cvlib as cv
from matplotlib import pyplot as plt


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""

    for (startX, startY, endX, endY) in faces:
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

    return image


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Load image:
image = cv2.imread("test_face_detection.jpg")

# Detect faces:
faces, confidences = cv.detect_face(image)

# Draw face detections:
img_result = show_detection(image.copy(), faces)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 5))
plt.suptitle("Face detection using cvlib face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_result, "cvlib face detector: " + str(len(faces)), 1)

# Show the Figure:
plt.show()

```

### Detecting facial landmarks

#### Detecting facial landmarks with OpenCV

三种算法：

- **FacemarkLBF**
- **FacemarkKamezi**
- **FacemarkAAM**

注意：在python中应用需要修改opencv源代码，具体修改方式查看原文



#### Detecting facial landmarks with dlib

```python
"""
Detecting facial landmarks using dlib
"""

# Import required packages:
import cv2
import dlib
import numpy as np

# Define what landmarks you want:
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
ALL_POINTS = list(range(0, 68))


def draw_shape_lines_all(np_shape, image):
    """Draws the shape using lines to connect between different parts of the face(e.g. nose, eyes, ...)"""

    draw_shape_lines_range(np_shape, image, JAWLINE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, LEFT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, NOSE_BRIDGE_POINTS)
    draw_shape_lines_range(np_shape, image, LOWER_NOSE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, LEFT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_OUTLINE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_INNER_POINTS, True)


def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    """Draws the shape using lines to connect the different points"""

    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=1, lineType=cv2.LINE_8)


def draw_shape_points_pos_range(np_shape, image, points):
    """Draws the shape using points and position for every landmark filtering by points parameter"""

    np_shape_display = np_shape[points]
    draw_shape_points_pos(np_shape_display, image)


def draw_shape_points_pos(np_shape, image):
    """Draws the shape using points and position for every landmark"""

    for idx, (x, y) in enumerate(np_shape):
        # Draw the positions for every detected landmark:
        cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))

        # Draw a point on every landmark position:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def draw_shape_points_range(np_shape, image, points):
    """Draws the shape using points for every landmark filtering by points parameter"""

    np_shape_display = np_shape[points]
    draw_shape_points(np_shape_display, image)


def draw_shape_points(np_shape, image):
    """Draws the shape using points for every landmark"""

    # Draw a point on every landmark position:
    for (x, y) in np_shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def shape_to_np(dlib_shape, dtype="int"):
    """Converts dlib shape object to numpy array"""

    # Initialize the list of (x,y) coordinates
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)

    # Loop over all facial landmarks and convert them to a tuple with (x,y) coordinates:
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)

    # Return the list of (x,y) coordinates:
    return coordinates


# Name of the two shape predictors:
p = "shape_predictor_68_face_landmarks.dat"
# p = "shape_predictor_5_face_landmarks.dat"

# Initialize frontal face detector and shape predictor:
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Create VideoCapture object to get images from the webcam:
video_capture = cv2.VideoCapture(0)

# You can use a test image for debugging purposes:
test_face = cv2.imread("face_test.png")


while True:

    # Capture frame from the VideoCapture object:
    ret, frame = video_capture.read()

    # Just for debugging purposes:
    # frame = test_face.copy()

    # Convert frame to grayscale:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces:
    rects = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Draw a box around the face:
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)

        # Get the shape using the predictor:
        shape = predictor(gray, rect)

        # Convert the shape to numpy array:
        shape = shape_to_np(shape)

        # Draw all lines connecting the different face parts:
        # draw_shape_lines_all(shape, frame)

        # Draw jaw line:
        # draw_shape_lines_range(shape, frame, JAWLINE_POINTS)

        # Draw all points and their position:
        # draw_shape_points_pos(shape, frame)
        # You can also use:
        # draw_shape_points_pos_range(shape, frame, ALL_POINTS)

        # Draw all shape points:
        draw_shape_points(shape, frame)

        # Draw left eye, right eye and bridge shape points and positions
        # draw_shape_points_pos_range(shape, frame, LEFT_EYE_POINTS + RIGHT_EYE_POINTS + NOSE_BRIDGE_POINTS)

    # Display the resulting frame
    cv2.imshow("Landmarks detection using dlib", frame)

    # Press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release everything:
video_capture.release()
cv2.destroyAllWindows()

```

#### Detecting facial landmarks with face_recognition

```python
"""
Detecting facial landmarks using face_recognition
"""

# Import required packages:
import cv2
import face_recognition
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Load image:
image = cv2.imread("face_test.png")

# Create images to show the results:
image_68 = image.copy()
image_5 = image.copy()

# Convert the image from BGR color (which OpenCV uses) to RGB color:
rgb = image[:, :, ::-1]

# Detect 68 landmarks:
face_landmarks_list_68 = face_recognition.face_landmarks(rgb)

# Print detected landmarks:
print(face_landmarks_list_68)

# Draw all detected landmarks:
for face_landmarks in face_landmarks_list_68:
    for facial_feature in face_landmarks.keys():
        for p in face_landmarks[facial_feature]:
            cv2.circle(image_68, p, 2, (0, 255, 0), -1)

# Detect 5 landmarks:
face_landmarks_list_5 = face_recognition.face_landmarks(rgb, None, "small")

# Print detected landmarks:
print(face_landmarks_list_5)

# Draw all detected landmarks:
for face_landmarks in face_landmarks_list_5:
    for facial_feature in face_landmarks.keys():
        for p in face_landmarks[facial_feature]:
            cv2.circle(image_5, p, 2, (0, 255, 0), -1)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 5))
plt.suptitle("Facial landmarks detection using face_recognition", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(image_68, "68 facial landmarks", 1)
show_img_with_matplotlib(image_5, "5 facial landmarks", 2)

# Show the Figure:
plt.show()

```

### Face tracking

- #### Face tracking with the dlib DCF-based tracker

- #### Object tracking with the dlib DCF-based tracke

### Face recognition

#### Face recognition with OpenCV

Indeed, OpenCV provides three different implementations to use:

- **Eigenfaces**
- **Fisherfaces**
- **Local Binary Patterns Histograms** (**LBPH**)

#### Face recognition with dlib

Making use of the dlib functionality, we can use a pre-trained model to map a face into a 128D descriptor. Afterward, we can use these feature vectors to perform face recognition.  

1) Calculate the 128D descriptor, used to quantify the face : 

```python
"""
This script makes used of dlib library to calculate the 128-dimensional (128D) descriptor to be used for face
recognition. Face recognition model can be downloaded from:
https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2
"""

# Import required packages:
import cv2
import dlib
import numpy as np

# Load shape predictor, face enconder and face detector using dlib library:
pose_predictor_5_point = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()


def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    """Returns the 128D descriptor for each face in the image"""

    # Detect faces:
    face_locations = detector(face_image, number_of_times_to_upsample)
    # Detected landmarks:
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]
    # Calculate the face encoding for every detected face using the detected landmarks for each one:
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


# Load image:
image = cv2.imread("jared_1.jpg")

# Convert image from BGR (OpenCV format) to RGB (dlib format):
rgb = image[:, :, ::-1]

# Calculate the encodings for every face of the image:
encodings = face_encodings(rgb)

# Show the first encoding:
print(encodings[0])

```

2) 欧氏距离计算相似性 （阈值为0.6）

```python
"""
This script makes used of dlib library to calculate the 128D descriptor to be used for face recognition
and compare the faces using some distance metrics
"""

# Import required packages:
import cv2
import dlib
import numpy as np

# Load shape predictor, face enconder and face detector using dlib library:
pose_predictor_5_point = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()


def compare_faces_ordered(encodings, face_names, encoding_to_check):
    """Returns the ordered distances and names when comparing a list of face encodings against a candidate to check"""

    distances = list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    return zip(*sorted(zip(distances, face_names)))


def compare_faces(encodings, encoding_to_check):
    """Returns the distances when comparing a list of face encodings against a candidate to check"""

    return list(np.linalg.norm(encodings - encoding_to_check, axis=1))


def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    """Returns the 128D descriptor for each face in the image"""

    # Detect faces:
    face_locations = detector(face_image, number_of_times_to_upsample)
    # Detected landmarks:
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]
    # Calculate the face encoding for every detected face using the detected landmarks for each one:
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


# Load images:
known_image_1 = cv2.imread("jared_1.jpg")
known_image_2 = cv2.imread("jared_2.jpg")
known_image_3 = cv2.imread("jared_3.jpg")
known_image_4 = cv2.imread("obama.jpg")
unknown_image = cv2.imread("jared_4.jpg")

# Convert image from BGR (OpenCV format) to RGB (dlib format):
known_image_1 = known_image_1[:, :, ::-1]
known_image_2 = known_image_2[:, :, ::-1]
known_image_3 = known_image_3[:, :, ::-1]
known_image_4 = known_image_4[:, :, ::-1]
unknown_image = unknown_image[:, :, ::-1]

# Crate names for each loaded image:
names = ["jared_1.jpg", "jared_2.jpg", "jared_3.jpg", "obama.jpg"]

# Create the encodings:
known_image_1_encoding = face_encodings(known_image_1)[0]
known_image_2_encoding = face_encodings(known_image_2)[0]
known_image_3_encoding = face_encodings(known_image_3)[0]
known_image_4_encoding = face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding, known_image_3_encoding, known_image_4_encoding]
unknown_encoding = face_encodings(unknown_image)[0]

# Compare faces:
computed_distances = compare_faces(known_encodings, unknown_encoding)
computed_distances_ordered, ordered_names = compare_faces_ordered(known_encodings, names, unknown_encoding)

# Print obtained results:
print(computed_distances)
print(computed_distances_ordered)
print(ordered_names)


```

#### Face recognition with face_recognition

## Chapter 12 - Introduction to Deep Learning

### Deep learning overview for computer vision tasks

#### Deep learning characteristics

- 高端基础设备支持，例如GPU优化
- 深度学习在特征检测和提取工作较为简单
- 经验法则：数据量大时，深度学习较优；数据量小时，传统机器学习方法较优

<img src="D:\学期计划\2021_暑假\img\16.png" alt="16" style="zoom:50%;" />

#### Deep learning for image classification

![17](D:\学期计划\2021_暑假\img\17.png)

#### Deep learning for object detection

数据集：

-  **PASCAL Visual Object Classification** (**PASCAL VOC**)  
-  **ImageNet** 
-  **Common Objects in Context** (**COCO**) 



评价指标：

-  **mean Average Precision** (**mAP**) 
-  **Average Precision** (**AP**) 



模型比较：

![18](D:\学期计划\2021_暑假\img\18.png)

### Deep learning in OpenCV

#### Understanding cv2.dnn.blobFromImage()

此函数即为深度学习投入图像训练前的准备工作，包括压缩图片，旋转等

#### OpenCV deep learning classification

代码见文件夹

- AlexNet for image classification
- GoogLeNet for image classification
- ResNet for image classification
- SqueezeNet for image classification

#### OpenCV deep learning object detection

代码见文件夹

- MobileNet-SSD for object detection
- YOLO for object detection

### The TensorFlow library

#### TensorFlow中TensorBoard的使用

不需要给出初始值，用于算法的图示化

#### Linear regression in TensorFlow

文件分为：训练、加载训练结果、模型和参数共同加载



When saving the final model (saver.save(sess, './linear_regression')), four files are created:

- .meta file: Contain the TensorFlow graph
- .data file: Contain the values of the weights, biases, gradients, and all the other variables saved
- .index file: Identify the checkpoint
- checkpoint file: Keeps a record of the latest checkpoint files saved

#### Handwritten digits recognition using TensorFlow

**one-hot encoding：** *labels have been converted from a single number to a vector, whose length is equal to the number of possible classes.*  

### The Keras library

#### Linear regression in Keras

#### Handwritten digit recognition in Keras



## Chapter 13 - Mobile and Web Computer Vision with Python and OpenCV

### Introduction to Flask

```python
from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == "__main__":
    app.run()
```

















