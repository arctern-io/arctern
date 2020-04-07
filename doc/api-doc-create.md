# API文档生成

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

## 生成API文档
```
cd arctern/doc/api-doc
make clean
make html
```
