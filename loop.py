from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import jieba
import pandas as pd
 

def datasets_demo():
    iris = load_iris()
    print("鸢尾花的数据集：\n",iris)
    print("查看数据集描述\n",iris["DESCR"])
    print("查看特征值的名字\n",iris.feature_names)
    print("查看特征值：\n",iris.data,iris.data.shape)
    
    
    #数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print("训练集的特征值\n",x_train,x_train.shape)
    return None

def dict_demo():

    #字典特征抽取
    data=[{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]
    #1.实例化一个转换器类
    transfer=DictVectorizer(sparse=False)
    #2.调用fit_transform()
    data_new=transfer.fit_transform(data)
    print("data_new\n",data_new)
    print("特征名字\n",transfer.get_feature_names())
    return None


def count_demo():
    #文本特征提取：CountVectorizer
    data=["life is short,i like like python","life is too long,i disike python"]
    #1.实例化一个转化器类
    transfer = CountVectorizer()
    #2.调用fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new\n",data_new.toarray())
    print("特征名字：\n",transfer.get_feature_names())
    
    return None






def count_chinese_demo():
    #文本特征提取：CountVectorizer
    data=["我 爱 北京 天安门","天安门 上 太阳 升"]
    #1.实例化一个转化器类
    transfer = CountVectorizer()
    #2.调用fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new\n",data_new.toarray())
    print("特征名字：\n",transfer.get_feature_names())
    
    return None


def cut_word(text):
    #进行中文分词
    a=" ".join(list(jieba.cut(text)))
    print(a)
    return text



def count_chinese_demo2():
    #中文文本特征抽取自动分词
    data = ["一切都像刚睡醒的样子，欣欣然张开了眼。",
    "山朗润起来了，水涨起来了，太阳的脸红起来了。",
    "小草偷偷地从土里钻出来，嫩嫩的，绿绿的。",
    "园子里，田野里，瞧去，一大片一大片满是的。",
    "坐着，躺着，打两个滚， 踢几脚球，赛几趟跑，捉几回迷藏。",
    "风轻悄悄的，草软绵绵的"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
        #print(data_new)
         #1.实例化一个转化器类
    transfer = CountVectorizer()
    #2.调用fit_transform
    data_final=transfer.fit_transform(data_new)
    print("data_new\n",data_final.toarray())
    print("特征名字：\n",transfer.get_feature_names())

    return None



def tfidf_demo():
    ##用TF-IDF的方法进行特征抽取

    data = ["一切都像刚睡醒的样子，欣欣然张开了眼。",
    "山朗润起来了，水涨起来了，太阳的脸红起来了。",
    "小草偷偷地从土里钻出来，嫩嫩的，绿绿的。",
    "园子里，田野里，瞧去，一大片一大片满是的。",
    "坐着，躺着，打两个滚， 踢几脚球，赛几趟跑，捉几回迷藏。",
    "风轻悄悄的，草软绵绵的"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
        #print(data_new)
         #1.实例化一个转化器类
    transfer = TfidfVectorizer()
    #2.调用fit_transform
    data_final=transfer.fit_transform(data_new)
    print("data_new\n",data_final.toarray())
    print("特征名字：\n",transfer.get_feature_names())


    return None




def minmax_demo():

    #1.获取数据
    data=pd.read_csv("dating.txt")
    print("data:\n",data)
    #2.实例化一个转换器类
    transfer=MinMaxScaler(feature_range=[1,3])
    #3.调用fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)



    return None




def stand_demo():

    #1.获取数据
    data=pd.read_csv("dating.txt")
    print("data:\n",data)
    #2.实例化一个转换器类
    transfer=StandardScaler()
    #3.调用fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)



    return None


def variance_demo():
    #低方差特征过滤
    data=pd.read_csv("dating.txt")
    transfer=VarianceThreshold()
    data_new = transfer.fit_transform(data)
    #计算相关系数
    r = pearsonr(data[""],data[""])


if __name__=="__main__":
    #代码1：sklearn数据集的使用
    #datasets_demo()
    #代码2：字典特征提取
    #dict_demo()
    #代码3：文本特征抽取
    #count_demo()
    #代码4：中文文本特征抽取
    #count_chinese_demo()
    #代码5：中文文本特征抽取
    #count_chinese_demo2()
    #代码6：中文切分
    #cut_word("我爱北京天安门")
    #代码7：
    #tfidf_demo()
    #代码8：归一化
    #minmax_demo()
    #代码9：标准化
    #stand_demo()
    #代码10：低方差特征过滤
    variance_demo()
