# API文档生成

如果你的doc目录下没有api-doc文件夹开始生成API文档，请按照下面的操作一步一步执行，如果你的doc目录下存在api-doc文件夹且里面有rst文件，请直接跳到生成API文档操作。wq

## 环境准备

操作系统  | 版本
:-----------:|:----------:
Ubuntu LTS  | 18.04 或以上

|软件名称  |
|:-----------:|
|miniconda（推荐） 或者 anaconda  |

## 安装所需包
```
conda install sphinx
conda install sphinx_automodapi
conda install sphinx_rtd_theme
```

## 生成conf.py文件
```
cd arctern/doc
mkdir api-doc
cd api-doc
sphinx-quickstart /*输入项目信息*/
```

## 配置conf.py及生成rst文件
```
在source/conf.py中加入如下代码：
    import os
    import sys
    sys.path.insert(0, os.path.abspath('../../../python/arctern'))
	......
	extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.viewcode',
   'sphinx_automodapi.automodapi',
   'sphinx.ext.inheritance_diagram'
   ]
   ......
   html_last_updated_fmt = '%b %d, %Y'
   html_domain_indices = True
   html_theme = 'sphinx_rtd_theme'
   html_logo = './_static/arctern-color.png' #图片请下载
   
生成rst文件：
   cd ../../..
   sphinx-apidoc -o doc/api-doc/source python/arctern
```

## 替换automodules为automodapi
```
执行replace.py(复制代码后在api-doc目录下执行）:
    import os
    source_api_path = '/source'
    automo_method = 'automodapi' # automodapi | automodsumm | automodule
    for rst_file in os.listdir(source_api_path):
      if rst_file.endswith('.rst'):
        with open(source_api_path + os.sep + rst_file, 'r') as f:
            contents = f.read()
        contents = contents.replace('automodule', automo_method)
        with open(source_api_path + os.sep + rst_file, 'w') as f:
            f.write(contents)
```

## 修改rst文件识别\_开头的文件以及html格式
```
识别\_开头文件的格式如下：
     .. automodapi:: arctern._wrapper_func
       :members:
       :undoc-members:
       :show-inheritance:
   
删除Submodules以及下面的模块名标题，例如：
     Submodules
     arctern.util.vega.choroplethmap.vega_choroplethmap module
   
如果需要打印类继承图，执行如下命令：
     conda install -c conda-forge graphviz 
     conda install -c conda-forge python-graphviz
	 
如果不打印类继承图，修改如下：
     .. automodapi:: arctern.util.vega.heatmap.vega_heatmap
        :members:
        :undoc-members:
        :no-inheritance-diagram:
```

## 生成API文档
```
cd arctern/doc/api-doc
mkdir build
make clean
make html
```
