# 使用docker模拟spark-hadoop集群
本文介绍使用`docker`技术在一台主机上启动3个`container`，这个3个`container`信息如下，相互间组织成一个`spark`集群，并在`spark`集群上运行`cpu`版的`arctern`
| host name |ip address | conainer name |
| ------ | ------ | ------ |
| master | 172.18.0.10 | master
| slave1 | 172.18.0.11 | slave1
| slave2 | 172.18.0.12 | slave2

----

## 创建docker子网
创建一个名为`hadoop`的docker子网
```bash
docker network create --subnet=172.18.0.0/16 hadoop
```

----

## 启动`container`
```bash
docker run -d -ti --name master --hostname master --net hadoop --ip 172.18.0.10 --add-host slave1:172.18.0.11 --add-host slave2:172.18.0.12 ubuntu:16.04 bash
docker run -d -ti --name slave1 --hostname slave1 --net hadoop --ip 172.18.0.11 --add-host master:172.18.0.10 --add-host slave2:172.18.0.12 ubuntu:16.04 bash
docker run -d -ti --name slave2 --hostname slave2 --net hadoop --ip 172.18.0.12 --add-host master:172.18.0.10 --add-host slave1:172.18.0.11 ubuntu:16.04 bash
```

----

## 安装依赖库
进入`docker`并安装依赖库,以`master`为例，`slave1`和`slave2`同此操作
```bash
# 进入 master
docker exec -it master bash
```
以下指令在`master`上执行
```bash
apt update
apt install -y libgl-dev libosmesa6-dev libglu1-mesa-dev wget openjdk-8-jre openssh-server vim
service ssh start
#新建hadoop用户
useradd -m hadoop -s /bin/bash
#修改hadoop用户密码为hadoop
echo -e "hadoop\nhadoop" | passwd hadoop
exit
```

----

## 设置免密码登录
退出`master`的`root`账户，在`host`上执行以下命令，以`hadoop`用户登录`master`
```bash
docker exec -it -u hadoop master bash
```
以下指令在`master`上执行，设置免密码登录,`slave1`和`slave2`同此操作

```bash
#生成ssh-key，用户免密登录
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa

#cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

#需要输入密码，密码为hadooop
ssh-copy-id master
ssh-copy-id slave1
ssh-copy-id slave2
```

----

## 配置`hdfs`
本小节所有操作均在`master`上执行
```bash
#进入hadoop用户目录
cd ~/

#下载hadoop
wget https://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz

#解压hadoop
tar -xvf hadoop-2.7.7.tar.gz
rm -rf hadoop-2.7.7.tar.gz

#创建数据目录
mkdir -p data/hdfs/datanode
mkdir -p data/hdfs/namenode
mkdir -p data/tmp
mkdir -p data/yarn/nodemanager
```
编辑`~/hadoop-2.7.7/etc/hadoop/hadoop-env.sh`，添加如下环境变量
```bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
```

编辑`~/hadoop-2.7.7/etc/hadoop/slaves`，内容如下
```txt
master
slave1
slave2
```

编辑`~/hadoop-2.7.7/etc/hadoop/core-site.xml`，内容如下
```xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
	<property>
		<name>fs.defaultFS</name>
		<value>hdfs://master:9000</value>
	</property>
	<property>
		<name>hadoop.tmp.dir</name>
		<value>file:///home/hadoop/data/tmp</value>
	</property>
</configuration>
```

编辑`~/hadoop-2.7.7/etc/hadoop/hdfs-site.xml`，内容如下
```xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
	<property>
		<name>dfs.replication</name>
		<value>1</value>
	</property>
	<property>
		<name>dfs.datanode.data.dir</name>
		<value>file:///home/hadoop/data/hdfs/datanode</value>
	</property>
	<property>
		<name>dfs.namenode.name.dir</name>
		<value>file:///home/hadoop/data/hdfs/namenode</value>
	</property>
	<property>
		<name>dfs.http.address</name>
		<value>0.0.0.0:50070</value>
	</property>

</configuration>
```

编辑`~/hadoop-2.7.7/etc/hadoop/mapred-site.xml`，内容如下
```xml
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
	<property>
		<name>mapreduce.framework.name</name>
		<value>yarn</value>
	</property>
</configuration>
```

编辑`~/hadoop-2.7.7/etc/hadoop/yarn-site.xml`，内容如下
```xml
<?xml version="1.0"?>

<configuration>

	<property>
		<name>yarn.resourcemanager.hostname</name>
		<value>master</value>
	</property>
	<property>
		<name>yarn.nodemanager.aux-services</name>
		<value>mapreduce_shuffle</value>
	</property>
	<property>
		<name>yarn.nodemanager.local-dirs</name>
		<value>file:///home/hadoop/data/yarn/nodemanager</value>
	</property>

	<property>
		<name>yarn.resourcemanager.address</name>
		<value>master:8032</value>
	</property>

	<property>
		<name>yarn.resourcemanager.scheduler.address</name>
		<value>master:8030</value>
	</property>

	<property>
		<name>yarn.resourcemanager.resource-tracker.address</name>
		<value>master:8031</value>
	</property>

	<property>
		<name>yarn.nodemanger.pmem-check-enabled</name>
		<value>false</value>
	</property>

	<property>
		<name>yarn.nodemanger.vmem-check-enabled</name>
		<value>false</value>
	</property>

	<property>
		<name>yarn.nodemanager.vmem-pmem-ratio</name>
		<value>21</value>
	</property>

	<property>
		<name>yarn.log-aggregation-enable</name>
		<value>true</value>
	</property>

	<property>
		<name>yarn.scheduler.minimum-allocation-mb</name>
		<value>128</value>
	</property>
	<property>
		<name>yarn.scheduler.maximum-allocation-mb</name>
		<value>2048</value>
	</property>
	<property>
		<name>yarn.scheduler.minimum-allocation-vcores</name>
		<value>1</value>
	</property>
	<property>
		<name>yarn.scheduler.maximum-allocation-vcores</name>
		<value>2</value>
	</property>
	<property>
		<name>yarn.nodemanager.resource.memory-mb</name>
		<value>4096</value>
	</property>
	<property>
		<name>yarn.nodemanager.resource.cpu-vcores</name>
		<value>4</value>
	</property>
	
</configuration>

```
将`master`配置复制到`slave1`,`slave2`
```bash
scp -r ~/hadoop-2.7.7 slave1:~/
scp -r ~/hadoop-2.7.7 slave2:~/
scp -r ~/data slave1:~/
scp -r ~/data slave2:~/
```
`slave1`、`slave2`的`~/hadoop-2.7.7/etc/hadoop/yarn-site.xml`需删除以下内容
```xml
	<property>
		<name>yarn.resourcemanager.hostname</name>
		<value>master</value>
	</property>
```

----

## 添加环境变量
`master`,`slave1`,`slave2`均需执行此操作
在`~/.bashrc`中添加环境变量
```bash
export HADOOP_PREFIX=/home/hadoop/hadoop-2.7.7
export HADOOP_HOME=$HADOOP_PREFIX
export HADOOP_COMMON_HOME=$HADOOP_PREFIX
export HADOOP_CONF_DIR=$HADOOP_PREFIX/etc/hadoop
export HADOOP_HDFS_HOME=$HADOOP_PREFIX
export HADOOP_MAPRED_HOME=$HADOOP_PREFIX
export HADOOP_YARN_HOME=$HADOOP_PREFIX
```

----

## 启动`hdfs`
以下指令在`master`节点上执行
```bash
# 格式化，只在第一次时执行即可
~/hadoop-2.7.7/bin/hdfs namenode -format
# 启动hdfs和yarn
~/hadoop-2.7.7/sbin/start-dfs.sh
~/hadoop-2.7.7/sbin/start-yarn.sh
```

----

## 检查`hdfs`是否正确启动
使用`ps ax`观察，`master`内如类似如下
```bash
  PID TTY      STAT   TIME COMMAND
 1854 ?        Sl     2:01 /usr/lib/jvm/java-8-openjdk-amd64/bin/java -Dproc_namenode -Xmx1000m
 2003 ?        Sl     2:07 /usr/lib/jvm/java-8-openjdk-amd64/bin/java -Dproc_datanode -Xmx1000m
 2195 ?        Sl     1:01 /usr/lib/jvm/java-8-openjdk-amd64/bin/java -Dproc_secondarynamenode -Xmx1000m
 2378 ?        Sl     9:38 /usr/lib/jvm/java-8-openjdk-amd64/bin/java -Dproc_resourcemanager -Xmx1000m
 2672 ?        Sl     4:44 /usr/lib/jvm/java-8-openjdk-amd64/bin/java -Dproc_nodemanager -Xmx1000m
```
`slave1`和`slave2`内容类似如下
```bash
  PID TTY      STAT   TIME COMMAND
 6232 ?        Sl     2:04 /usr/lib/jvm/java-8-openjdk-amd64/bin/java -Dproc_datanode -Xmx1000m
 6352 ?        Sl     4:45 /usr/lib/jvm/java-8-openjdk-amd64/bin/java -Dproc_nodemanager -Xmx1000m
```

----

## 在`hdfs`上创建目录
```bash
~/hadoop-2.7.7/bin/hdfs dfs -mkdir /data
~/hadoop-2.7.7/bin/hdfs dfs -mkdir /data/tmp
~/hadoop-2.7.7/bin/hdfs dfs -ls hdfs:/data
```

----

## 安装`conda`
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b
echo "source $HOME/miniconda3/etc/profile.d/conda.sh" >> .bashrc
rm ~/miniconda.sh
```
退出并重新进入`master`，检查`conda`是否安装生效
```bash
hadoop@master:/$ conda env list
# conda environments:
#
base                  *  /home/hadoop/miniconda3
```

----

## 安装`arctern`
创建一个名为`arctern`的`conda`环境，并安装`arctern-spark`
```bash
conda create -y -n arctern -c conda-forge -c arctern-dev arctern-spark
```
检查`arctern`是否成功安装
```bash
conda activate arctern

(arctern) hadoop@master:~$ python
Python 3.7.6 | packaged by conda-forge | (default, Mar  5 2020, 15:27:18) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import arctern_pyspark
>>> exit()
```

按以上步骤在`master`、`slave1`、`slave2`安装`conda`和`arctern`

----

## 安装`spark`
与`hadoop`一样，只需在`master`节点配置`spark`，然后`scp`至`slave1`和`slave2`
```bash
#进入home目录
cd ~/

#下载 spark
wget https://downloads.apache.org/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop2.7.tgz

#解压 spark
#解压hadoop
tar -xvf spark-3.0.0-preview2-bin-hadoop2.7.tgz
rm -rf spark-3.0.0-preview2-bin-hadoop2.7.tgz
```
编辑`~/spark-3.0.0-preview2-bin-hadoop2.7/conf/spark-env.sh`，添加如下内容
```bash
export HADOOP_CONF_DIR=/home/hadoop/hadoop-2.7.7/etc/hadoop
export YARN_CONF_DIR=/home/hadoop/hadoop-2.7.7/etc/hadoop
export PYSPARK_PYTHON=/home/hadoop/miniconda3/envs/arctern/bin/python
```
编辑`~/spark-3.0.0-preview2-bin-hadoop2.7/conf/spark-defaults.conf`，添加如下内容
```txt
spark.executorEnv.PROJ_LIB         /home/hadoop/miniconda3/envs/arctern/share/proj
spark.executorEnv.GDAL_DATA        /home/hadoop/miniconda3/envs/arctern/share/gdal
```
将`master`配置复制到`slave1`,`slave2`
```bash
scp -r ~/spark-3.0.0-preview2-bin-hadoop2.7 slave1:~/
scp -r ~/spark-3.0.0-preview2-bin-hadoop2.7 slave2:~/
```
跑官方示例，检查`spark`是否安装成功
```bash
./spark-3.0.0-preview2-bin-hadoop2.7/bin/spark-submit --class org.apache.spark.examples.SparkPi --master yarn  --deploy-mode cluster ./spark-3.0.0-preview2-bin-hadoop2.7/examples/jars/spark-examples_2.12-3.0.0-preview2.jar 10
```

----

## 测试`arctern`
以下操作只在`master`上执行, 新建`gen.py`文件用于生成测试数据，内容如下
```python
from random import random
cnt=100000
print("idx,pos")
for i in range(0, cnt):
    lng = random()*360 - 180
    lat = random()*180 - 90
    print(i,"point({} {})".format(lng,lat),sep=',')
```
生成测试数据，并将测试数据存入`hdfs`
```bash
python gen.py > ~/pos.csv
~/hadoop-2.7.7/bin/hdfs dfs -put pos.csv hdfs:/data/
~/hadoop-2.7.7/bin/hdfs dfs -ls hdfs:/data/
```
新建`st_transform_test.py`，内容如下
```python
from pyspark.sql import SparkSession
from arctern_pyspark import register_funcs

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("st_transform test") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    register_funcs(spark)

    df=spark.read.format("csv").option("header",True).option("delimiter",",").schema("idx long, pos string").load("hdfs:///data/pos.csv")
    df.printSchema()
    df.createOrReplaceTempView("pos")
    rst = spark.sql("select idx,pos,st_transform(pos, 'epsg:4326', 'epsg:3857') from pos")
    rst.write.mode("append").csv("hdfs:///data/st_transform/")
    spark.stop()
```
向`spark`提交`st_transform_test.py`
```bash
~/spark-3.0.0-preview2-bin-hadoop2.7/bin/spark-submit --master yarn --deploy-mode cluster st_transform_test.py
```
检查`st_transform_test.py`的运行结果
```bash
hadoop@master:~$ ~/hadoop-2.7.7/bin/hdfs dfs -ls hdfs:/data/st_transform
Found 3 items
-rw-r--r--   1 hadoop supergroup          0 2020-03-20 12:21 hdfs:///data/st_transform/_SUCCESS
-rw-r--r--   1 hadoop supergroup    8538736 2020-03-20 12:21 hdfs:///data/st_transform/part-00000-c07a4e2c-965f-406a-adb7-c31a42cb92b6-c000.csv
-rw-r--r--   1 hadoop supergroup     792532 2020-03-20 12:21 hdfs:///data/st_transform/part-00001-c07a4e2c-965f-406a-adb7-c31a42cb92b6-c000.csv
```

----

## 参考文献
- https://hadoop.apache.org/docs/r2.7.4/hadoop-project-dist/hadoop-common/SingleCluster.html
- https://www.alexjf.net/blog/distributed-systems/hadoop-yarn-installation-definitive-guide/
- https://zhuanlan.zhihu.com/p/97693616?utm_source=wechat_session&utm_medium=social&utm_oi=747365067686682624