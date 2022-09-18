
# PCB Art Generator

## Usage

前两个必选参数为输入图片和颜色配置文件

可选参数：

-o 指定输出的位置，留空则以输入文件名创建新文件夹存放

-w 指定图片宽度对应的物理尺寸，带单位，如100mm、0.1m、1000mil

-r 指定生成png图片的物理分辨率，带单位，同上

```
python pcb_art_gen.py cirno_bb.png colors/photo_jlc_blue.yaml -w 50mm -r 1mil
```


### 输出

copper_mask.png / copper_mask.svg

![copper_mask.svg](cirno_bb_output/copper_mask.svg)

overlay_mask.png / overlay_mask.svg

![overlay_mask.svg](cirno_bb_output/overlay_mask.svg)

solder_mask.png / solder_mask.svg

![solder_mask.svg](cirno_bb_output/solder_mask.svg)

三个层的mask，其中黑色代表实体，白色为背景。svg格式可以通过 https://github.com/xsrf/easyeda-svg-import.git 无损导入立创eda


### 其他可调节参数（均带物理单位）

--overlay-clearance 丝印和阻焊边界之间预留的最小距离

--min-line-width 所有层的最小线宽

--min-gap-width 所有层的最小间隙

--min-copper-size 铜的最小尺寸

--svg-approx-eps 矢量图转换精度



