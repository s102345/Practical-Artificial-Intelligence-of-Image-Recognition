#作業1: python, np, plt 繪圖應用
#載入相關套件模組
import numpy as np
import matplotlib.pyplot as plt

fileName = input("請輸入圖片名稱：") #輸入影像檔名字串 filename (用 input)

## Part 1: 繪製 Colorbars ################
a = np.arange(256) #生成0~255的一維整數陣列a (用 numpy)
a.astype(dtype="uint8") #格式改為uint8 (用 astype)
b = np.repeat(a, 3, axis=0) #a重複3次，長度成為256*3的b陣列 (用 repeat)
c = np.zeros((64*8, 256*3, 3), dtype="uint8") #建立(64*8,256*3,3)大小的零矩陣c，格式是uint8

#用兩層for迴圈產生影像資訊：第一層跟高有關，第二層跟RGB通道有關
for h in range(8):
    if h < 2: # black & white
        for channel in range(0, 3):
            c[h*64 : (h + 1) * 64, : , channel] = b #將b複製到c矩陣特定掃描線與色通道
    else: # R(2, 3) G(4, 5) B(6, 7)
        c[h*64 : (h + 1) * 64, : , int(h / 2) - 1] = b #將b複製到c矩陣特定掃描線與色通道

    b = np.invert(b) #可用np.invert將陣列b的順序倒置

plt.figure(1) #建立1號視窗(用 figure)
plt.imshow(c) #顯示影像c(用 imshow)
plt.title("Fig.1 Colorbars") #加標題
plt.axis('off') #不顯示 x,y 軸刻度值
plt.pause(2) #暫停2秒(用 pause)
plt.close() #關閉1號視窗(用 close)

## Part 2: 繪製黑白圈圈圖 #################
## 自訂一個函數，名稱是 radius，輸入u,v影像座標，算出(return)遠離中心的半徑
def radius(u, v):
    return np.sqrt(u**2 + v**2) #用平方和開根號計算遠離影像中心的半徑r
d = np.arange(6, 1, -0.2) #產生一個6至1之間，以-0.2為間隔的數列d
e = np.cumsum(np.square(d)) #將d開平方，再用cumsum產生半徑的門檻值數列e
f = np.zeros((512, 512), dtype='float32') #以float32格式，建立512x512大小的零矩陣f


#利用雙層for迴圈，窮舉x,y影像座標
for x in range(512):
    for y in range(512):
        r = radius(x - 255, y - 255) #用radius函式計算遠離影像中心的半徑r，x,y要先減去255
        idx = 1
        #用for迴圈，依序查詢數列e裡的數值
        for z in range(e.size - 1, -1, -1):
            #如果r大於數列e裡的第z筆數值，就用變數idx記錄該索引值z，並用(break)離開for迴圈
            if r >= e[z]:
                idx = z
                break
        f[x, y] = (idx + 1) % 2  #將idx值取除以2的餘數，將該數值存入影像f的x,y位置(為了得到相符的答案所以需要idx+1)
    
fig = plt.figure(1) #建立2號視窗(用 figure)
plt.imshow(f, cmap='gray') #顯示影像f，色彩對用表用'gray'(用 imshow, 以及cmap參數)
plt.title("Fig. 2 Rings") #加標題
plt.xticks(np.arange(0, 512, 64)) #水平軸刻度用0~512, 以64為間隔(用 xtick)
plt.yticks(np.arange(0, 512, 64)) #垂直軸刻度用0~512, 以64為間隔(用 ytick)
plt.pause(2) #暫停2秒(用 pause)
plt.close() #關閉2號視窗(用 close)

## Part 3: 6種影像處理之水平拼接圖像，隨機排序+輪播 #########
## 自訂一個函數，名稱是 process
## 輸入正方形彩色影像(im_in)以及影像處理選項p
## 根據選項 p, 將處理後的影像(im_out)輸出(return)
## 可以考慮的處理有 rot90, fliplr, flipud, bitwise_not, clip, where 等等
def process(im_in, p):
    if p==0:
       im_out = np.rot90(im_in) 
    elif p==1:
       im_out = np.fliplr(im_in)
    elif p==2:
       im_out = np.flipud(im_in)
    elif p==3:
       im_out = 255 - im_in
    elif p==4:
       im_out = np.clip(im_in, 0, 128)
    elif p==5:
       im_out = np.where(im_in > 127, 255, 2 * im_in)  
    return im_out

#主程式進入點
im1 = plt.imread(fileName) #讀取 filename 指定的彩色影像im1 (用 plt.imread)
h, w, ch = im1.shape #讀取該影像的高(h)/寬(w)/通道數(ch) (用.shape)
im2 = im1[:, :h, :ch] #取影像im1的局部，使im2的高寬都是h
im3 = np.zeros((6, h, h, ch), dtype='uint8') #建立uint8格式的四維零陣列，尺寸是(6,h,h,ch)
#建立一個 for 迴圈，分別把 process 產生的6種影像，存到im3裡
for p in range(6):
    im3[p] = process(im2, p) #im3的第一維就是process的選項
#用 for 迴圈跑五次隨機順序排列的影像im4
for i in range(5):
    g = np.random.permutation(6) #用np.random.permutation(6)產生0~5的隨機序列
    im4 = np.hstack(im3[g, :, : ,:]) #用 hstack, 以g的順序，水平合併6個子圖

    fig = plt.figure(3) #建立3號視窗(用 figure)
    plt.imshow(im4 ) #顯示影像im4(用 imshow)
    plt.title(f"Random order: {i}") #加標題(含隨機次數)
    plt.axis('off') #不顯示 x,y 軸刻度值
    plt.pause(1) #暫停1秒
plt.imsave("HW1.jpg", im4) #將im4存入'HW1.jpg'檔
plt.close() #關閉3號視窗(用 close)

## Part 4: 將最後的6個子圖，以2x3圖形陣列(用add_subplot)，全螢幕顯示 ####
fig = plt.figure(4) #建立4號視窗(用 figure)
#建立 for 迴圈，依序處理圖形陣列中的6個圖
for i in range(6):
    fig.add_subplot(2,3,i+1) #添加2x3圖形陣列的第i+1個子圖
    im5 = im3[g[i], :, :, :] #讀取im3中g[i]序號所對映的子影像，並用squeeze降成三維，存入im5
    plt.imshow(im5) #顯示影像im5
    plt.title(f"Proc. {g[i]}") #顯示子圖的標題
    plt.axis('off') #不顯示 x,y 軸刻度值

#全螢幕顯示
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show() #顯示




