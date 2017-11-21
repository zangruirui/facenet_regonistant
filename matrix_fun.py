# -*- coding:utf-8 -*-
import numpy as np
import face_mysql


class matrix:
    def __init__(self):
        pass

    # 两个矩阵的欧式距离
    def EuclideanDistances(self, A, B):
        BT = B.transpose()
        # vecProd = A * BT
        vecProd = np.dot(A, BT)
        # print(vecProd)
        SqA = A ** 2
        # print(SqA)
        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
        # print(sumSqAEx)

        SqB = B ** 2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
        SqED = sumSqBEx + sumSqAEx - 2 * vecProd
        SqED[SqED < 0] = 0.0
        ED = np.sqrt(SqED)
        return ED.transpose()
    #
    def get_socres(self, A, ugroup):
        # 设置每次处理的最大数据库记录数
        #如果数据库中记录太多时可分批进行处理
        maxlen = 128

        fmysql = face_mysql.face_mysql()
        results = np.array(fmysql.findall_facejson(ugroup))

        #如果没有数据库找到时 直接返回空的list
        if results.shape[0] == 0: return [],[],[]

        pic_scores_all = []
        #获取数据库中的入库时的图片名称  pic_names在数据库中存的是数组列索引4这个位置
        pic_names = results[:, 4]
        #获取入库时图片对象的uid  pic_uid在数据库中存的是数组列索引2这个位置
        pic_uid = results[:, 2]
        for i in range(0, len(results), maxlen):
            pic_vectors = results[i:i + maxlen, 3]
            # 效率待优化，现在是每行处理
            pic_vectors = [[float(j) for j in i.split(',')] for i in pic_vectors]
            pic_socores = self.EuclideanDistances(A, np.array(pic_vectors))
            pic_socores_list = np.array(pic_socores).tolist()

            pic_scores_all.extend(pic_socores_list)
        pic_scores_all = np.array(pic_scores_all).transpose()

        # 获取距离最近的值
        # np.argsort() 返回排序后的索引
        pic_min_scores = np.amin(pic_scores_all, axis=1)
        pic_min_names = []
        pic_min_uid = []
        for i in range(0, len(pic_min_scores)):
            # 获取最小值的index
            index = np.where(pic_scores_all[i] == pic_min_scores[i])
            # print(int(index[0]))
            # 有多个符合条件的只取第一个
            pic_min_names.append(pic_names[index[0][0]])
            pic_min_uid.append(pic_uid[index[0][0]])
        # print(pic_min_names)
        return pic_min_scores.tolist(), pic_min_names, pic_min_uid


