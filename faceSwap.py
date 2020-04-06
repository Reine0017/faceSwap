import cv2
import dlib
import numpy as np

def extract_index_nparray(nparray):
	index = None
	for num in nparray[0]:
		index = num
		break
	return index

img = cv2.imread("bradley_cooper.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#create a black mask the size of img_gray
mask = np.zeros_like(img_gray)

img2 = cv2.imread("jim_carrey.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faces = detector(img_gray)
for face in faces:
	landmarks = predictor(img_gray, face)
	landmarks_points = []

	for n in range(0,68):
		x = landmarks.part(n).x
		y = landmarks.part(n).y
		landmarks_points.append((x,y))

		#cv2.circle(img,(x,y), 3, (0,0,255), -1)

	points = np.array(landmarks_points, np.int32)
	print(points)
	#use convexhull instead of using fixed points because in case a person turns, hard coded points wldnt be accurate.
	convexhull = cv2.convexHull(points)
	print(convexhull)

	#to see the convexhull drawn on face
	#cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
	cv2.fillConvexPoly(mask, convexhull, 255)

	face_image_1 = cv2.bitwise_and(img, img, mask=mask)

	#delaunay triangulation
	rect = cv2.boundingRect(convexhull)
	subdiv = cv2.Subdiv2D(rect)
	subdiv.insert(landmarks_points)
	triangles = subdiv.getTriangleList()
	triangles = np.array(triangles, dtype = np.int32)

	indexes_triangles = []

	for t in triangles:
		print("Triangles", t)
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])

		index_pt1 = np.where((points == pt1).all(axis=1))
		print(pt1)
		index_pt1 = extract_index_nparray(index_pt1)
		print(index_pt1)

		index_pt2 = np.where((points == pt2).all(axis=1))
		index_pt2 = extract_index_nparray(index_pt2)

		index_pt3 = np.where((points == pt3).all(axis=1))
		index_pt3 = extract_index_nparray(index_pt3)

		if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
			triangle = [index_pt1, index_pt2, index_pt3]
			indexes_triangles.append(triangle)

		cv2.line(img, pt1, pt2, (0,0,255), 2)
		cv2.line(img, pt2, pt3, (0,0,255), 2)
		cv2.line(img, pt1, pt3, (0,0,255), 2)
	print(indexes_triangles)

	#(x,y,w,h) = rect
	#cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0))

#for second face
faces2 = detector(img2_gray)
for face in faces2:
	landmarks2 = predictor(img2_gray, face)
	landmarks_points2 = []

	for n in range(0,68):
		x = landmarks2.part(n).x
		y = landmarks2.part(n).y
		landmarks_points2.append((x,y))

		cv2.circle(img2, (x,y), 3, (0, 255, 0), -1)

#triangulation of second face, from first face's delaunay triangulation
for triangle_index in indexes_triangles:
	pt1 = landmarks_points2[triangle_index[0]]
	pt2 = landmarks_points2[triangle_index[1]]
	pt3 = landmarks_points2[triangle_index[2]]

	cv2.line(img2, pt1, pt2, (0,0,255), 2)
	cv2.line(img2, pt3, pt2, (0,0,255), 2)
	cv2.line(img2, pt1, pt3, (0,0,255), 2)

cv2.imshow("Image1", img)
cv2.imshow("Image2", img2)
cv2.imshow("mask", mask)
cv2.imshow("face_image_1", face_image_1)
cv2.waitKey(0)
cv2.destroyAllWindows()