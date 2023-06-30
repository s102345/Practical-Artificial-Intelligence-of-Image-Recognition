'''作業三：OpenCV瞌睡偵測 Level 3 提示'''
import cv2 # 匯入 openCV 套件
import numpy as np # 匯入 Numpy 套件

# 載入哈爾小波級聯正臉偵測訓練集(用CascadeClassifier())
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
# 建立視訊物件並讀取影片檔
cap = cv2.VideoCapture("data/sleepy.mp4")

## 讀取視訊參數 ##
#視訊畫面高(用.get)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) #視訊畫面高
#視訊畫面寬(用.get)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) #視訊畫面寬
#視訊幀率(用.get)
fps = cap.get(cv2.CAP_PROP_FPS) #視訊幀率
#視訊總幀數(用.get)
total_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) #視訊總幀數

#畫面編號(預設0)
frame_count = 0
#人臉特效選項(預設1，不做特效)
face_effect = 1
#閃爍計數器(預設10)
face_counter = 10

#特效名稱list
effect_name = ['None', 'GrayScale', 'Bilevel', 'Mosaic', 'Negative', 'Edges', 'Mask']

#畫面編號(預設0)
frame_count = 0
# 用無窮迴圈讀取影片中每個畫格(幀)   
while True:
  # 畫面數量加 1
  frame_count += 1
  # 讀取影片中的畫格(用.read())
  ret, frame = cap.read()
  # 若有讀取到影片中的畫格  
  if ret:
    ###### 偵測膚色 #######
    # ROI是高寬25%:75%範圍
    ROI = frame[int(height * 0.25):int(height * 0.75), int(width * 0.25):int(width * 0.75)]
    # 將ROI從 BGR 轉換至 HSV 色空間(用 cvtColor())
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
    # 定義 HSV (hue, saturation, value) 空間的膚色上下界範圍(下界約 (0,50,50), 上界約 (80,180,220))
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([80, 180, 220])
    # 註：HSV的範圍上限 [180, 256, 256]
    # 取HSV色空間下，ROI範圍內的膚色遮罩(用inRange())
    mask = cv2.inRange(ROI, lower_bound, upper_bound)
    mask_w, mask_h = mask.shape

    # 算出膚色面積率，也就膚色遮罩非零數值佔遮罩面積的比率(用np.count_nonzero()，可用mask[:]忽略維度，用round(,精度)四捨五入)
    skin_mask = np.count_nonzero(mask[:])
    mask_area = mask_w * mask_h
    skin_ratio = round(skin_mask / mask_area, 2)

    # 如果膚色面積率高於0.07，代表「有膚色」，否則代表「無膚色」
    if skin_ratio > 0.07:
      isSkin = True
    else:
      isSkin = False

    # frame的中心點
    midX, midY= int(height / 2), int(width / 2)
    # 將膚色遮罩轉換為彩色格式(用cvtColor)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 在遮罩上加面積率數值(用putText())    
    cv2.putText(mask, "area: " + str(skin_ratio), (100, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 128, 255), 2)  
    # 建立跟輸入影像一樣大，一樣格式的背景影像(用np.zeros())
    background = np.zeros(frame.shape, dtype=np.uint8)
    # 將背景影像設成灰色
    background[:] = (192, 192, 192)
    # 把膚色遮罩貼入背景影像
    background[midX - mask_w // 2: midX + mask_w // 2, midY - mask_h // 2: midY + mask_h // 2] = mask

    ###### 偵測人臉 ######
    # 影像轉成灰階格式(用cvtColor())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 偵測正臉(用.detectMultiScale())
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #偵測人臉
    # 如果正臉數量(可用len(faces))為零，代表「無正臉」
    if len(faces) == 0:
      isFrontFace = False
      if face_counter == 10:
        face_counter -= 1
      elif face_counter < 10 and face_counter > 0:
        face_counter -= 1
    # 否則「有正臉」
    else:
      #簡易紀錄是否有正臉
      isFrontFace = True
      #閃爍計數器
      face_counter = 10
      for(x, y, w, h) in faces:
        #增加矩形框的高度
        h += 10

        face_ROI = frame[y:y+h, x:x+w] #取得人臉ROI

        # 如果人臉特效選項為1
        if face_effect == 1:
          #不做任何事
          pass 
        # 如果人臉特效選項為2
        elif face_effect == 2:
          # 執行特效2處理 ...
          # 灰階
          face_ROI = cv2.cvtColor(face_ROI, cv2.COLOR_BGR2GRAY)
          # 灰階轉3通道 
          face_ROI = cv2.cvtColor(face_ROI, cv2.COLOR_GRAY2BGR)    
        # 如果人臉特效選項為3
        elif face_effect == 3:
          # 執行特效3處理 ... 
          # 灰階
          face_ROI = cv2.cvtColor(face_ROI, cv2.COLOR_BGR2GRAY) 
          # 二值化
          _, face_ROI = cv2.threshold(face_ROI, 127, 255, cv2.THRESH_BINARY)
          # 灰階轉3通道 
          face_ROI = cv2.cvtColor(face_ROI, cv2.COLOR_GRAY2BGR)        
        # 如果人臉特效選項為4 
        elif face_effect == 4:
        # 執行特效4處理 ....
          #以破壞性插值法達成馬賽克
          #以線性插值縮小
          face_ROI = cv2.resize(face_ROI, (w//10, h//10), interpolation=cv2.INTER_LINEAR) 
          #以鄰近像素插值放大
          face_ROI = cv2.resize(face_ROI, (w, h), interpolation=cv2.INTER_NEAREST) 
        # 如果人臉特效選項為5
        elif face_effect == 5:
        # 執行特效5處理 ....
          #負片
          face_ROI = -face_ROI
        # 如果人臉特效選項為6
        elif face_effect == 6:
        # 執行特效6處理 ....
          #灰階
          face_ROI = cv2.cvtColor(face_ROI, cv2.COLOR_BGR2GRAY) 
          #Canny邊緣偵測
          face_ROI = cv2.Canny(face_ROI, 30, 150)
          #灰階轉3通道 
          face_ROI = cv2.cvtColor(face_ROI, cv2.COLOR_GRAY2BGR) 
        # 如果人臉特效選項為7
        elif face_effect == 7:
        # 執行特效7處理 ....
          #載入口罩圖片
          mask = cv2.imread('data/mask.png')
          #縮放到適當的大小
          mask = cv2.resize(mask, (w, h - h//2))
          #放口罩到臉上
          face_ROI[h//2:h, :] = mask

        frame[y:y+h, x:x+w] = face_ROI #人臉ROI放回原圖
        if face_effect != 1:
          #在人臉矩形框下方放特效名稱(用putText())
          cv2.putText(frame, effect_name[face_effect-1], (x, y+h+10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        #繪製人臉矩形框(用rectangle())
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #繪製人臉矩形框
        #繪製人臉橢圓框(用ellipse())     
        cv2.ellipse(frame,(x+w//2, y+h//2),(w//2, h//2), 0, 0, 360, (0, 255, 255), 2) #繪製人臉橢圓框
        #在人臉矩形框上方放學號文字(用putText())
        cv2.putText(frame, 'B10915030', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

    #### 判斷是否打瞌睡 #####
    # 如果「有膚色」但「無正臉」，在視窗中顯示"Wake Up!"字樣(最好能閃爍)
    if isSkin and not isFrontFace:
      # 閃爍效果 兩次黃色兩次藍色
      if face_counter % 4  < 2:
        cv2.putText(frame, 'Wake Up!', (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 255), 4)
      else:
        cv2.putText(frame, 'Wake Up!', (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

    # 如果「無膚色」且「無正臉」，在視窗中顯示"Nobody"字樣
    if not isSkin and not isFrontFace:
      cv2.putText(frame, 'Nobody', (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

    #水平合併膚色遮罩與人臉偵測影像(用np.hstack)，並顯示該畫格(用imshow())
    cv2.imshow('frame', np.hstack((background, frame)))
    # 停每800/fps毫秒讀取鍵盤的按鍵(用waitKey())
    key = cv2.waitKey(round(800/fps)) & 0xFF
    #當按鍵為ESC(ASCII碼為27)時跳出迴圈 
    if key == 27:
      break
    # 如果電腦鍵盤按下1(可用ord('1')轉成相應的ASCII碼)，將人臉區域處理選項設為 1(不做特效)
    elif key == ord('1'): 
      face_effect = 1
    # 如果電腦鍵盤按下2，將人臉區域處理選項設為2 (執行特效2)
    elif key == ord('2'): 
      face_effect = 2
    # 如果電腦鍵盤按下3，將人臉區域處理選項設為3 (執行特效3)
    elif key == ord('3'): 
      face_effect = 3
    # 如果電腦鍵盤按下4，將人臉區域處理選項設為4 (執行特效4)
    elif key == ord('4'): 
      face_effect = 4
    # 如果電腦鍵盤按下5，將人臉區域處理選項設為5 (執行特效5)
    elif key == ord('5'): 
      face_effect = 5
    # 如果電腦鍵盤按下6，將人臉區域處理選項設為6 (執行特效6)
    elif key == ord('6'): 
      face_effect = 6
    #如果電腦鍵盤按下7，將人臉區域處理選項設為7 (執行特效7)
    elif key == ord('7'): 
      face_effect = 7
  # 當沒讀到影片中的畫格時，跳出迴圈
  else:
    break

# 關閉視訊物件
cap.release()
# 關閉視窗
cv2.destroyAllWindows()