'''作業二：OpenCV基本應用 提示'''
#載入相關套件模組
import cv2
import numpy as np

buttonDown= False #滑鼠左鍵是否按下(全域面數)

## 自定義滑鼠回應函式
def onMouse(event, x, y, flags, param):
	#如果滑鼠左鍵按下
	if event == cv2.EVENT_LBUTTONDOWN:
		global buttonDown
		#buttonDown設為 True
		buttonDown = True 
	#如果滑鼠移動
	if event == cv2.EVENT_MOUSEMOVE:
		#如果按鈕按下
		if buttonDown:
			#在(x,y)位置繪製半徑6px的圓形點
			cv2.circle(im1, (x,y), 6, (0,255,255), -1)
            #顯示影像
			cv2.imshow('draw', im1)
	#如果滑鼠左鍵彈起
	if event == cv2.EVENT_LBUTTONUP:
		#buttonDown設為 False
		buttonDown = False
	#如果按下滑鼠右鍵
	if event == cv2.EVENT_RBUTTONDOWN:
		#將視窗刪除
		cv2.destroyAllWindows()

  
#    event: EVENT_LBUTTONDOWN,   EVENT_RBUTTONDOWN,   EVENT_MBUTTONDOWN,
#         EVENT_LBUTTONUP,     EVENT_RBUTTONUP,     EVENT_MBUTTONUP,
#         EVENT_LBUTTONDBLCLK, EVENT_RBUTTONDBLCLK, EVENT_MBUTTONDBLCLK,
#         EVENT_MOUSEMOVE: 

#    flags: EVENT_FLAG_CTRLKEY, EVENT_FLAG_SHIFTKEY, EVENT_FLAG_ALTKEY,
#         EVENT_FLAG_LBUTTON, EVENT_FLAG_RBUTTON,  EVENT_FLAG_MBUTTON

## 自定義滑桿回應函式
def onTrackbar(pos):
	#讀取sliders的資料
	weight = cv2.getTrackbarPos('weight', 'fusion')
    #注意slider2不得等於0
	size = cv2.getTrackbarPos('size', 'fusion')
	if size == 0:
		size = 1
		cv2.setTrackbarPos('size', 'fusion', size)
	#讀取sliders的資料
	negative = cv2.getTrackbarPos('negative', 'fusion')

	#算出im2的寬高
	h, w, _ = im2.shape
	#讓im1縮小成im2的寬高，縮小後稱為im3
	im3 = cv2.resize(im1, (w, h))
	#建立跟im2一樣大的黑影像im4
	im4 = np.ones((h, w, 3), np.uint8)
	#根據slider2的數值,用cv2.resize把im3等比例縮小
	new_h, new_w = h * size // 100, w * size // 100 #縮小後的height, width
	im3 = cv2.resize(im3, (new_w, new_h)) #縮小im3
	#將縮小的im3貼入im4的中央
	cx, cy = (w - new_w) // 2, (h - new_h) // 2 #計算貼入的起始點 -> 根據im3的height, width
	im4[cy:cy + new_h, cx:cx + new_w] = im3	#將im3貼入im4

    #根據slider1的數值,用cv2.addWeighted對im2與im4加權混合
	alpha = weight / 100 #手繪weight
	beta = 1 - alpha #背景weight
	im5 = cv2.addWeighted(im2, alpha, im4, beta, 0)

	#複製im5到im6
	im6 = im5
	#用cv2.bitwise_not對im6取反
	im6 = cv2.bitwise_not(im6)
	#根據slider3的數值,決定反相部分的寬度
	negative_width = w * negative // 100
	#把反相後的im6貼入im5
	im5[:, 0:negative_width] = im6[:, 0:negative_width]
    #顯示影像
	cv2.imshow('fusion', im5)
        
## 主程式起始處
if __name__ == '__main__':
	#建立400x400的黑影像im1
	im1 = np.ones((400,400,3), np.uint8)
	#顯示黑影像
	cv2.imshow('draw', im1)
	#用cv2.setMouseCallback建立滑鼠回應函式
	cv2.setMouseCallback('draw', onMouse)
	#等待
	cv2.waitKey(0)
	# 讀取背景影像im2
	im2 = cv2.imread(r'./ntust.jpg')
	# 顯示背景影像
	cv2.imshow('fusion', im2)
	# 用cv2.createTrackbar建立weight滑桿
	cv2.createTrackbar('weight', 'fusion', 50, 100, onTrackbar)
	# 用cv2.createTrackbar建立size滑桿
	cv2.createTrackbar('size', 'fusion', 50, 100, onTrackbar)
	# 用cv2.createTrackbar建立negative滑桿
	cv2.createTrackbar('negative', 'fusion', 50, 100, onTrackbar)
	# onTrackbar回應函式初始化
	onTrackbar(0)
	# 等待
	while True:
		k = cv2.waitKey(0)
		if k == 27:
			#按下ESC離開
			break
		elif k == ord('r'):
			# 重設所有的trackbar到預設值
			cv2.setTrackbarPos('weight', 'fusion', 50)
			cv2.setTrackbarPos('size', 'fusion', 50)
			cv2.setTrackbarPos('negative', 'fusion', 50)
		else:
			#無視其他按鍵
			continue