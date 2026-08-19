from matplotlib import pyplot as plt

def draw_points(points, color = 'b.' ,label = None):
    plt.plot([x[0] for x in points], [x[1] for x in points], color , label = label)

def draw_axis(mean = None, axis = None):
    if axis is None:
        return
    for i in range(2):
        component = axis[i]
        plt.quiver(mean[0], mean[1], component[0], component[1], 
                        angles='xy', scale_units='xy', scale = 0.01, color=['black', 'black'][i], 
                        width=0.005)

def draw_hull(hull, color = "red", label = None, linestyle = None):
    plt.plot(hull[:, 0], hull[:, 1], linewidth = 2 ,color = color, label = label, linestyle=linestyle)

def plt_samples(points, hull,  mean = None, axis = None, title = None):
    draw_points(points)
    draw_axis(mean, axis)
    draw_hull(hull)
    plt.title(title)
    
