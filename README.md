# facenet_face_regonistant
利用facenet实现检测图片中的人脸，将识别到的人脸向量存入数据库，此外利用post提交一个新图片 返回数据库中相似的人脸的信息

### 安装准备
##### 安装python包 
按照requirements.txt中的包全部安装即可（其中mysql-connector-python 我采用的yum install 安装的）

##### 提前建立数据库 
建表语句再database.sql 
（需要提前建立数据库，名字自己定义）

##### 模型准备
本项目是根据[facenet](https://github.com/davidsandberg/facenet)中提取关键的代码，将其进行封装使用
所以需要提交下载facenect提供的模型 [模型地址](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit) 需要可访问谷歌

该模型现在保存在[百度网盘](http://pan.baidu.com/s/1i4YhAdB)中  密码：avbl
下载下来后按照models\facenet\20170512-110547  这个目录结构存放即可
百度网盘链接 ： 链接：http://pan.baidu.com/s/1i4YhAdB 密码：avbl

### 如何使用？
模拟post请求，如图所示
图中依次是插入、查询时的场景
![](https://github.com/zangruirui/facenet_regonistant/blob/master/img/insert.png)
![](https://github.com/zangruirui/facenet_regonistant/blob/master/img/query.png)
![](https://github.com/zangruirui/facenet_regonistant/blob/master/img/query1.png)

