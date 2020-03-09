# 环境准备

# 测试目录的结构

# 执行arctern的测试
我们是通过spark-submit来提交测试，如果已经在上一步的环境准备中安装了spark，你应该清楚spark-submit的位置

`/usr/local/bin/spark/bin/spark-submit ./spark_test.py`

通过以上命令来运行所有的测试，如果想要定制运行的测试，请修改./spark_test.py中的main函数部分

# 比较arctern和postgis的结果
在上一步中，运行spark-submit命令得到的结果只是arctern测试的初步结果，如果需要和postgis的结果进行比对，还需要进一步的处理

1，需要将初始结果进行改动  
2，需要规范初始结果中不合规范的数据格式  
3，执行compare.py中的compare_all()方法  

## 如果两边结果不相等

# 如何增加测试用例
增加用例的步骤比较繁琐，请一定按步骤操作  
注意：如果只是为已经编写好的测试函数增加测试数据，可以从步骤3开始

1，在./spark_test.py中编写测试函数（命名请按照run_test_xxxx的格式），并在main中增加对函数的调用  
2，在config文件中增加一行，格式为x1=x2=x3=x4，其中  
   x1：测试函数的名称，严格一致，全局唯一  
   x2：测试函数中table的名称，严格一致，全局唯一  
   x3：转换postgis sql的sql文件名称  
   x4：转换postgis sql的执行结果存放的文件名称  

   比如：run_test_st_area=test_area=st_area=st_area，其中run_test_st_area是arctern测试中测试函数的名称；test_area则是测试过程中临时创建的table的名称；test_area既是sql文件的名称，也是sql执行结果的文件名，分别存放在./expected/sqls和./expected/results中  
3，在./data中为arctern测试增加数据，目前支持的数据格式为json，注意编写测试函数时你的sql会与这里json的keys相关  
4，将你的测试函数和数据转换成postgis可以识别的sql语句，并存放在./expected/sqls中  
5，在postgis中执行上述sql，并将结果记录在./expected/results中
