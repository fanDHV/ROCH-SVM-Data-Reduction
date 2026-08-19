import numpy as np
import time
from shapely.geometry import Polygon
# Find special_points

class O_quickhull:
    def __init__(self, points):
        self.points = points
        self.arranged_points = self.calculate_o_quickhull()
        self.dientich = self.S()
    def find_special_points(self):
        points = self.points
        maxY = np.nanmax(points[:, 1])
        minY = np.nanmin(points[:, 1])
        maxX = np.nanmax(points[:, 0])
        minX = np.nanmin(points[:, 0])

        rightPoints = points[np.where(np.equal(points[:, 0], maxX))]
        leftPoints = points[np.where(np.equal(points[:, 0], minX))]
        topPoints = points[np.where(np.equal(points[:, 1], maxY))]
        bottomPoints = points[np.where(np.equal(points[:, 1], minY))]

        if len(topPoints) == 1:
            top = (topPoints[0],)
        else:
            topPoints = sorted(topPoints, key = lambda x : x[0])
            top = (topPoints[0], topPoints[len(topPoints)-1])

            # bottom
        if len(bottomPoints) == 1:
            bottom = (bottomPoints[0],)
        else:
            bottomPoints = sorted(bottomPoints, key = lambda x : -x[0])
            bottom = (bottomPoints[0], bottomPoints[len(bottomPoints)-1])

            # right
        if len(rightPoints) == 1:
            right = (rightPoints[0],)
        else:
            rightPoints = sorted(rightPoints, key = lambda x : -x[1])
            right = (rightPoints[0], rightPoints[len(rightPoints)-1])

            # left
        if len(leftPoints) == 1:
            left = (leftPoints[0],)
        else:
            leftPoints = sorted(leftPoints, key = lambda x : x[1])
            left = (leftPoints[0], leftPoints[len(leftPoints)-1])

        if len(top) == 1:
            q1 = qq4 = top[0]
        else:
            q1 = top[0]
            qq4 = top[1]
        q4 = right[0]
        if len(right) == 1:
            qq3 = right[0]
        else:
            qq3 = right[1]
        q3 = bottom[0]     
        if len(bottom) == 1:
            qq2 = bottom[0]
        else:
            qq2 = bottom[1]
        q2 = left[0]
        if len(left) == 1:
            qq1 = left[0]
        else:
            qq1 = left[1]

        
        #  Tra ve cac diem dac biet
        return top, bottom, right, left, q1, qq1, q2, qq2, q3, qq3, q4, qq4

    # Các hàm tìm các điểm O-Quickhull cho mỗi vùng
    def find_o_hull1(self, set1, q1, qq1):
        if len(set1) == 0:
            return []
        
        key1 = (set1[:,0] - q1[0])*(set1[:,0] - q1[0]) + (set1[:,1] - qq1[1])*(set1[:,1] - qq1[1])
        maxset1 = np.nanmax(key1)
        new_point1 = set1[np.where(key1 == maxset1)][0]
        new_set11 = set1[np.where(set1[:,1] > new_point1[1])]
        new_set12 = set1[np.where(set1[:,0] < new_point1[0])]
        
        return self.find_o_hull1(new_set11, q1, new_point1) + [new_point1] + self.find_o_hull1(new_set12, new_point1, qq1)

    def find_o_hull2(self, set2, q2, qq2):
        if len(set2) == 0:
            #po = ortho(pf, pt, xInc, yInc) # SUPPORT POINTS
            return []

        
        key2 = (set2[:,0] - qq2[0])*(set2[:,0] - qq2[0]) + (set2[:,1] - q2[1])*(set2[:,1] - q2[1])
        maxset2 = np.nanmax(key2)
        new_point2 = set2[np.where(key2 == maxset2)][0]
        
        new_set21 = set2[np.where(set2[:,0] < new_point2[0])]
        new_set22 = set2[np.where(set2[:,1] < new_point2[1])]
        
        
        return self.find_o_hull2(new_set21, q2, new_point2) + [new_point2] + self.find_o_hull2(new_set22, new_point2, qq2)

    def find_o_hull3(self, set3, q3, qq3):
        if len(set3) == 0:
            #po = ortho(pf, pt, xInc, yInc) # SUPPORT POINTS
            return []
    
        key3 = (set3[:,0] - q3[0])*(set3[:,0] - q3[0]) + (set3[:,1] - qq3[1])*(set3[:,1] - qq3[1])
        maxset3 = np.nanmax(key3)
        new_point3 = set3[np.where(key3 == maxset3)][0]
        
        new_set31 = set3[np.where(set3[:,1] < new_point3[1])]
        new_set32 = set3[np.where(set3[:,0] > new_point3[0])]
        
        return self.find_o_hull3(new_set31, q3, new_point3) + [new_point3] + self.find_o_hull3(new_set32, new_point3, qq3)

    def find_o_hull4(self, set4, q4, qq4):
        if len(set4) == 0:
            return []

        key4 = (set4[:,0] - qq4[0])*(set4[:,0] - qq4[0]) + (set4[:,1] - q4[1])*(set4[:,1] - q4[1])
        maxset4 = np.nanmax(key4)
        new_point4 = set4[np.where(key4 == maxset4)][0]
        
        new_set41 = set4[np.where(set4[:,0] > new_point4[0])]
        new_set42 = set4[np.where(set4[:,1] > new_point4[1])]
        
        return self.find_o_hull4(new_set41, q4, new_point4) + [new_point4] + self.find_o_hull4(new_set42, new_point4, qq4)


    def calculate_o_quickhull(self):
        points = self.points
        start_time = time.time()
        top, bottom, right, left, q1, qq1, q2, qq2, q3, qq3, q4, qq4 = self.find_special_points()
        set1 = points[np.where((points[:, 0] <= q1[0]) & (points[:, 1] >= qq1[1]))]
        set2 = points[np.where((points[:, 0] <= qq2[0]) & (points[:, 1] <= q2[1]))]
        set3 = points[np.where((points[:, 0] >= q3[0]) & (points[:, 1] <= qq3[1]))]
        set4 = points[np.where((points[:, 0] >= qq4[0]) & (points[:, 1] >= q4[1]))]

  
        arranged_points = []
        arranged_points += [q1] + self.find_o_hull1(set1, q1, qq1) + [qq1]
        arranged_points += [q2] + self.find_o_hull2(set2, q2, qq2) + [qq2]
        arranged_points += [q3] + self.find_o_hull3(set3, q3, qq3) + [qq3]
        arranged_points += [q4] + self.find_o_hull4(set4, q4, qq4) + [qq4]

        arranged_points.append(arranged_points[0])
        S = []
        n = len(arranged_points)
        for i in range(0, n-1):
        
            if arranged_points[i+1][0] > arranged_points[i][0] and arranged_points[i+1][1] > arranged_points[i][1]:
            
                p3 = [arranged_points[i][0], arranged_points[i+1][1]]
            
                S.append([i+1, p3])

            elif arranged_points[i+1][0] > arranged_points[i][0] and arranged_points[i+1][1] < arranged_points[i][1]:
            
                p3 = [arranged_points[i+1][0], arranged_points[i][1]]
            
                S.append([i+1, p3])

            elif arranged_points[i+1][0] < arranged_points[i][0] and arranged_points[i+1][1] < arranged_points[i][1]:
        
                p3 = [arranged_points[i][0], arranged_points[i+1][1]]
        
                S.append([i+1, p3])

            elif arranged_points[i+1][0] < arranged_points[i][0] and arranged_points[i+1][1] > arranged_points[i][1]:
            
                p3 = [arranged_points[i+1][0], arranged_points[i][1]]
            
                S.append([i+1, p3])



        for i in range(len(S)):
            arranged_points.insert(S[i][0]+i, S[i][1])
        
        arranged_points.append(arranged_points[0])

        arranged_points = np.array(arranged_points)
        return arranged_points

    def S(self):
        return Polygon(self.arranged_points).area



