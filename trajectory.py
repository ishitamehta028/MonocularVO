import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
from monocularVO import visual_odometry 


file_path = "C:\\Users\\Ishita\\Desktop\\CS\\openCV\\MVO\\dataset\\sequences\\00\\image_0\\"
pose_path = "C:\\Users\\Ishita\\Desktop\\CS\\openCV\\MVO\\dataset\\dataset\\poses\\00.txt"



traj = np.zeros(shape=(600, 800, 3))



def trajectory(mono_co, true_co, traj) :
    

    cv.putText(traj, 'Estimated Odometry Position:', (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv.putText(traj, 'Green', (270, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)
    cv.putText(traj, 'Ground Truth:', (140, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv.putText(traj, 'Red', (270, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)

    draw_x, draw_y, draw_z = [int(round(x)) for x in mono_co]
    true_x, true_y, true_z = [int(round(x)) for x in true_co]

    traj = cv.circle(traj, (draw_x + 400, draw_z + 100), 5,((0, 255, 0)), -1)
    traj = cv.circle(traj, (true_x + 400, true_z + 100), 5, ((0, 0, 255)), -1)

    cv.imwrite("./trajectory.png", traj)



xline = list()
yline = list()
zline = list()

x_truth = list()
y_truth = list()
z_truth = list()




def plot():

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Estimated Position vs Ground Truth')

    ax.scatter3D(xline, yline, zline, color = 'green')
    ax.scatter3D(x_truth, y_truth, z_truth, color = 'red')
    plt.show()



def main():

    id = 1

    length = len(os.listdir(file_path))
    
    while( id < length) :
        if id < 2 :
            old_frame = cv.imread(file_path + str(0).zfill(6) + ".png")
            current_frame = cv.imread(file_path + str(id).zfill(6) + ".png")
        
         
            mono , true_co = visual_odometry(old_frame, current_frame, id)
            print(id, " Translation Error : ", true_co - mono)

            trajectory(mono, true_co, traj)
        

            xline.append(mono[0])
            yline.append(mono[1])
            zline.append(mono[2])
            x_truth .append(true_co[0])
            y_truth .append(true_co[1])
            z_truth .append(true_co[2]) 

            id = 2


        else : 
            old_frame = current_frame
            current_frame = cv.imread(file_path + str(id).zfill(6) + ".png")
            mono , true_co = visual_odometry(old_frame, current_frame,id)
            print(" Translation Error : ", true_co - mono)

            trajectory(mono, true_co, traj)
    

            xline.append(mono[0])
            yline.append(mono[1])
            zline.append(mono[2])
            x_truth .append(true_co[0])
            y_truth .append(true_co[1])
            z_truth .append(true_co[2]) 
    
            id += 1

    plot()
   



if __name__ == "__main__" :
    main()