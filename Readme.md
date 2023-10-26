一份简单的环视系统制作实现，包含完整的标定、投影、拼接和实时运行流程  
...
author: jinghui.tang  
date: 2023/10/15  
参考： 'https://github.com/hynpu/surround-view-system-introduction  
[ADAS-开源环视360全景拼接代码原理分析与实现](https://mp.weixin.qq.com/s?__biz=MzkzNjQ0NDMyMg==&mid=2247483912&idx=1&sn=cc456edd073e8e8e791b361b843ce099&chksm=c29feac5f5e863d355745c433eeb28f10fb77acc96801ba377ba9e02207cfda0dc61b00dd18c&token=201619039&lang=zh_CN#rd)
...

# doc
张正友标定论文以及推导流程参考 ./doc/张正友标定.pdf  
棋盘格标定原理参考  ./doc/棋盘格标定.pdf  
环视标定参考 ./doc/avm.pdf  

# run
运行前需更改图像读取保存路径  
mkdir build && cd build  
cmake ..  
make  

计算投影矩阵  
./calib_main  
手动选点 顺序以及位置参考 ./images/choose_back.png  

环视bev图像拼接  
./main  
--------------------------------

