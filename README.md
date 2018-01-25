#my blog [www.biexiabibi.com](http://www.biexiabibi.com)

# facenet_face_regonistant
利用facenet实现检测图片中的人脸，将识别到的人脸向量存入数据库，此外利用post提交一个新图片（也可以提交一个图片地址，参考face_recognition_api.py文件中get_url_imgae函数自行修改），返回数据库中相似的人脸的信息
算法主要分为2个步骤<br/>
1.**提取图片中的人脸 ，并保存到临时目录中**<br/>
2.**将人脸图片转换为128维的向量 ，便于后续求人脸相似度**<br/>

项目主要分为3个步骤<br/>
1.**提交post请求，将uid ugroup pic提交，进行人脸信息保存操作**<br/>
2.**收到请求后将pic进行处理解析为128维向量保存，并跟uid和ugroup保存入库 ，返回数据库插入成功的id**<br/>
3.**提交post请求，将ugroup pic提交人脸查询请求，意思为再ugroup中查看与图片pic相似的人脸**<br/>
4.**收到请求后，处理图片解析图片中所有的人脸，进行按库查询，然后与该图片中所有人脸相似的uid和距离（相似度距离）**<br/>

## 安装准备
#### 安装python包 
按照requirements.txt中的包全部安装即可（其中mysql-connector-python 我采用的yum install 安装的）
如下
`tensorflow==1.2`
`scipy`
`scikit-learn`
`opencv-python`
`h5py`
`matplotlib`
`Pillow`
`requests`
`psutil`
`mysql-connector-python`
`Werkzeug`
`Flask`
`Flask-HTTPAuth`



## 提前建立数据库 
建表语句再database.sql 
（需要提前建立数据库，名字自己定义，本项目数据库名为face_data）
数据库配置在face_mysql.py文件中 
（第12行 配置数据库用户名、密码、地址、数据库地址 本案例配置如下
```python
db = mysql.connector.connect(user='root', password='123456', host='127.0.0.1', database='face_data')
```
）

## 模型准备
本项目是根据[facenet](https://github.com/davidsandberg/facenet)中提取关键的代码，将其进行封装使用
所以需要提交下载facenect提供的模型 [模型地址](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) 需要可访问谷歌

该模型现在保存在[百度网盘](http://pan.baidu.com/s/1i4YhAdB)中  密码：avbl
下载下来后按照models\facenet\20170512-110547  这个目录结构存放即可
百度网盘链接 ： 链接：http://pan.baidu.com/s/1i4YhAdB 密码：avbl

## 如何使用？
首先在服务器上 执行 
``` bash
python   face_recognition_api.py  
```

访问地址是XXXXXX:8088  这个可以配置 （文件face_recognition_api.py 最后代码中有注释）
#### 模拟post请求，如图所示
本项目演示案例是利用谷歌浏览器插件“Postman”进行模拟的post请求
模拟post请求，如图所示
图中依次是插入、查询时的场景
* 插入请求地址为http://127.0.0.1:8088/face/insert
* 查询请求地址为http://127.0.0.1:8088/face/query
请求参数如图所示 文本字段有uid ugroup  文件字段是imagefile

![插入示意图](https://github.com/zangruirui/facenet_regonistant/blob/master/img/insert.png)

![查询示意图1](https://github.com/zangruirui/facenet_regonistant/blob/master/img/query.png)

![查询示意图1](https://github.com/zangruirui/facenet_regonistant/blob/master/img/query1.png)


# 处理文件下的图片提取人脸图片问题
处理一个文件夹下的所有图片，将人脸信息提取出来并保存到单独的文件夹下
执行
``` bash
python   face_recognition_savepic.py  
```
修改文件参数<br/>
1.**images_path：要处理的图片文件夹路径**<br/>
2.**modelpath： 模型存放路径**<br/>
3.**out_path ： 将每个图片中每个人脸转换为128维向量保存到json文件中**<br/>



###  识别图像中的人脸并保存图片案例
例如下图是将test.jpg  和test2.jpg识别人脸
在test.jpg 识别到一个人脸 保存为test_0.jpg
在test2.jpg 识别到2个人脸  保存为test2_0.jpg  test2_1.jpg

####  原图 test.jpg
![原图](https://github.com/zangruirui/facenet_regonistant/blob/master/pic_tmp/test.jpg)
####  识别结果 test_0.jpg
![识别人脸图](https://github.com/zangruirui/facenet_regonistant/blob/master/pic_tmp/test_0.jpg)

####  原图 test2.jpg
![原图](https://github.com/zangruirui/facenet_regonistant/blob/master/pic_tmp/test2.jpg)
####  识别结果 test2_0.jpg   test2_0.jpg
![识别人脸图1](https://github.com/zangruirui/facenet_regonistant/blob/master/pic_tmp/test2_0.jpg)
![识别人脸图2](https://github.com/zangruirui/facenet_regonistant/blob/master/pic_tmp/test2_1.jpg)
