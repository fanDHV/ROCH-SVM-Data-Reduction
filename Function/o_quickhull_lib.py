import numpy as np

class O_quickhull:
    def __init__(self, points):
        self.points = np.asarray(points)
        self.arranged_points = self.calculate_o_quickhull()
    
    def find_special_points(self):
        points = self.points

        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
        top_mask = points[:, 1] == max_y
        bottom_mask = points[:, 1] == min_y
        right_mask = points[:, 0] == max_x
        left_mask = points[:, 0] == min_x
        
        top_points = points[top_mask]
        bottom_points = points[bottom_mask]
        right_points = points[right_mask]
        left_points = points[left_mask]
        def get_corner_points(edge_points, x_priority='min'):
            if len(edge_points) == 1:
                return edge_points[0], edge_points[0]
            
            if x_priority == 'min':
                primary = edge_points[np.argmin(edge_points[:, 0])]
                secondary = edge_points[np.argmax(edge_points[:, 0])]
            else:
                primary = edge_points[np.argmax(edge_points[:, 0])]
                secondary = edge_points[np.argmin(edge_points[:, 0])]
                
            return primary, secondary
        
        q1, qq4 = get_corner_points(top_points, 'min')
        q2, qq1 = get_corner_points(left_points, 'min')
        q3, qq2 = get_corner_points(bottom_points, 'max')
        q4, qq3 = get_corner_points(right_points, 'max')
        
        return q1, qq1, q2, qq2, q3, qq3, q4, qq4
    
    def find_o_hull(self, points_set, q, qq, quadrant):
        if len(points_set) == 0:
            return []
        
        result = []
        stack = [('task', points_set, q, qq)]
        
        while stack:
            entry = stack.pop()
            entry_type = entry[0]
            
            if entry_type == 'point':
                point_val, q_val, qq_val = entry[1], entry[2], entry[3]
                if not np.array_equal(point_val, q_val) and not np.array_equal(point_val, qq_val):
                    result.append(point_val)
                continue
                
            _, current_set, current_q, current_qq = entry
            if len(current_set) == 0:
                continue
            
            if quadrant in (1, 3):
                distances = ((current_set[:, 0] - current_q[0])**2 + 
                           (current_set[:, 1] - current_qq[1])**2)
            else:
                distances = ((current_set[:, 0] - current_qq[0])**2 + 
                           (current_set[:, 1] - current_q[1])**2)
            
            max_idx = np.argmax(distances)
            new_point = current_set[max_idx]
            
            if quadrant == 1:
                set1 = current_set[current_set[:, 1] > new_point[1]]
                set2 = current_set[current_set[:, 0] < new_point[0]]
            elif quadrant == 2:
                set1 = current_set[current_set[:, 0] < new_point[0]]
                set2 = current_set[current_set[:, 1] < new_point[1]]
            elif quadrant == 3:
                set1 = current_set[current_set[:, 1] < new_point[1]]
                set2 = current_set[current_set[:, 0] > new_point[0]]
            else:  # quadrant == 4
                set1 = current_set[current_set[:, 0] > new_point[0]]
                set2 = current_set[current_set[:, 1] > new_point[1]]

            stack.append(('task', set2, new_point, current_qq))
            stack.append(('point', new_point, current_q, current_qq))
            stack.append(('task', set1, current_q, new_point))
        
        return result
    
    def calculate_o_quickhull(self):
        
        points = self.points
        
        if len(points) < 3:
            return points
            
        q1, qq1, q2, qq2, q3, qq3, q4, qq4 = self.find_special_points()
        
        mask1 = (points[:, 0] <= q1[0]) & (points[:, 1] >= qq1[1])
        mask2 = (points[:, 0] <= qq2[0]) & (points[:, 1] <= q2[1])
        mask3 = (points[:, 0] >= q3[0]) & (points[:, 1] <= qq3[1])
        mask4 = (points[:, 0] >= qq4[0]) & (points[:, 1] >= q4[1])
        
        set1 = points[mask1]
        set2 = points[mask2]
        set3 = points[mask3]
        set4 = points[mask4]
        
        
        hull1 = [q1] + self.find_o_hull(set1, q1, qq1, 1) + [qq1]
        hull2 = [q2] + self.find_o_hull(set2, q2, qq2, 2) + [qq2]
        hull3 = [q3] + self.find_o_hull(set3, q3, qq3, 3) + [qq3]
        hull4 = [q4] + self.find_o_hull(set4, q4, qq4, 4) + [qq4]
        
        arranged_points = hull1 + hull2 + hull3 + hull4
        
        
        new_points = []
        for i in range(len(arranged_points) - 1):
            new_points.append(arranged_points[i])
            current = arranged_points[i]
            next_pt = arranged_points[i + 1]
            
            dx = next_pt[0] - current[0]
            dy = next_pt[1] - current[1]
            
            if dx > 0 and dy > 0:
                new_points.append([current[0], next_pt[1]])
            elif dx > 0 and dy < 0:
                new_points.append([next_pt[0], current[1]])
            elif dx < 0 and dy < 0:
                new_points.append([current[0], next_pt[1]])
            elif dx < 0 and dy > 0:
                new_points.append([next_pt[0], current[1]])
        
        new_points.append(arranged_points[-1])
        
        return np.array(new_points)
    
