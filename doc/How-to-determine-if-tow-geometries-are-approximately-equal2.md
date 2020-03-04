
## Use shapely to determine if two geometries are approximately equal.
 
### Example :
#### `arctern`demo
```python
import shapely
from shapely import wkt

geo1=wkt.loads('GEOMETRYCOLLECTION(POLYGON((0 0,0 100000,100000 100000,100000 0,0 0)),LINESTRING(160 90, 10 150, 90 190.00001 ),POINT(90 190) ) ')
geo2=wkt.loads('GEOMETRYCOLLECTION(POLYGON((0 0,0 100000,100000.001 100000,100000 0,0 0)),LINESTRING(160 90, 10 150, 90 190.00001 ),POINT(90 190) ) ')
geo3=wkt.loads('GEOMETRYCOLLECTION(POLYGON((0 0,0 100000,100000.00100001 100000,100000 0,0 0)),LINESTRING(160 90, 10 150, 90 190.00001 ),POINT(90 190) ) ')

geo1.equals(geo2)
geo1.almost_equals(geo2)
geo1.equals_exact(geo2,0)
geo1.equals_exact(geo2,1e-2)  
geo1.equals_exact(geo2,1e-46)

geo3.equals(geo2)
geo3.almost_equals(geo2)
geo3.equals_exact(geo2,0)
geo3.equals_exact(geo2,1e-2)  
geo3.equals_exact(geo2,1e-46)

```
### conda安装 shapely
```bash
conda install shapely
```

### 参考链接
- [shapely github](https://github.com/Toblerity/Shapely)
- [shapely manual](https://shapely.readthedocs.io/en/latest)
