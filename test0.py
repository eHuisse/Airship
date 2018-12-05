from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import time


class BalloonUpside:
    '''
    class plotting a balloon from above on a canva and moving it (rotation and translation)
    '''
    def __init__(self, belongingCanva):
        '''
        Balloon class, for plotting and moving the hull on the monitor from above
        :param belongingCanva: (Tk().canva) Canva on which plot the hull
        '''
        self.canva = belongingCanva
        self.xMax = int(self.canva.cget('width'))
        self.yMax = int(self.canva.cget('height'))
        self.x = int(self.xMax/2)
        self.y = int(self.yMax/2)
        self.angle = 0

        self.hullImg = Image.open("pix/imgup.png")
        self.hullImg = self.hullImg.resize((int(self.xMax/20), int(self.xMax/20)), Image.ANTIALIAS)
        self.hullImgPI = ImageTk.PhotoImage(self.hullImg)

        self.printedHull = self.canva.create_image(self.x, self.y, image=self.hullImgPI, anchor="c")

    def moveHull(self, x=0, y=0):
        '''
        fonction moving the hull on the canva
        :param x: (float) Xpos on the canva
        :param y: (float) Ypos on the canva
        '''
        self.x = x
        self.y = y
        self.canva.coords(self.printedHull, self.x, self.y)

    def rotateHull(self, angle_deg=0):
        '''
        fonction rotating the hull on the canva
        :param angle_deg: (float) angle in rad
        '''
        self.angle = angle_deg
        self.canva.delete(self.printedHull)
        self.hullImgPI = ImageTk.PhotoImage(self.hullImg.rotate(int(self.angle), expand=1))
        self.printedHull = self.canva.create_image(self.x, self.y, image=self.hullImgPI, anchor="c")



class BalloonSide:
    '''
    class plotting a balloon from side on a canva and moving it (rotation and translation)
    '''
    def __init__(self, belongingCanva):
        '''
        Balloon class, for plotting and moving the hull on the monitor from above
        :param belongingCanva: (Tk().canva) Canva on which plot the hull
        '''
        self.canva = belongingCanva
        self.xMax = int(self.canva.cget('width'))
        self.yMax = int(self.canva.cget('height'))
        self.y = int(self.yMax/2)

        self.hullImg = Image.open("pix/imgside.png")
        self.hullImg = self.hullImg.resize((int(self.yMax/8), int(self.yMax/8)), Image.ANTIALIAS)
        self.hullImgPI = ImageTk.PhotoImage(self.hullImg)

        self.printedHull = self.canva.create_image(int(self.xMax/2), self.y, image=self.hullImgPI, anchor="c")

    def moveHull(self, y=0):
        '''
        fonction moving the hull on the canva
        :param y: (float) Ypos on the canva
        '''
        self.y = y
        self.canva.coords(self.printedHull, int(self.xMax/2), self.y)


    def rotateHull(self, angle_deg=0):
        '''
        fonction rotating the hull on the canva
        :param angle_deg: (float) angle in rad
        '''
        self.canva.delete(self.printedHull)
        self.hullImgPI = ImageTk.PhotoImage(self.hullImg.rotate(int(angle_deg), expand=1))
        self.printedHull = self.canva.create_image(int(self.xMax/2), self.y, image=self.hullImgPI, anchor="c")


class monitorSimple:
    '''
    class plotting the monitor
    '''
    def __init__(self, CHeight, CWidth):
        '''
        :param CHeight: (int) monitor height
        :param CWidth:  (int) monitor width
        '''

        #Monitor creation
        self.frame = Tk()
        self.CWidth = CWidth
        self.CHeight = CHeight
        self.zoomCoefup = 2
        self.zoomCoefside = 1
        self.isDefiningTraj = False
        self.isDefiningTrajUp = True

        #test data (to be removed after testing)
        self.testR = 1
        self.testtheta = 0

        #coordinate of balloon (above view)
        self.xUp = 0
        self.yUp = 0

        #coordinate of balloon (side view)
        self.ySide = 0

        #contain graduation mark
        self.graduationsUP = []
        self.graduationsSide = []

        #contain trajectory and trajectory mark
        self.trajectory = [[0,0]]
        self.trajectoryLines = []

        #trajectory
        self.xytrajectory = []
        self.ztrajectory = []

        #creation of Canves above view
        self.upMonitorFrame = Canvas(self.frame, width=self.CHeight, height=self.CHeight, bg='white')
        self.upMonitorFrame.grid(row=0, column=1)

        #creation of Canva side view
        self.sideMonitorFrame = Canvas(self.frame, width=self.CWidth, height=self.CHeight, bg='white')
        self.sideMonitorFrame.grid(row=0, column=2)

        #Balloon creation
        self.upHull = BalloonUpside(self.upMonitorFrame)
        self.sideHull = BalloonSide(self.sideMonitorFrame)

        #plotting graduation
        self.upMonitorFrame.create_line(0, int(self.CHeight / 2), int(self.CHeight), int(self.CHeight / 2), arrow=LAST)
        self.upMonitorFrame.create_line( int(self.CHeight / 2), int(self.CHeight), int(self.CHeight / 2), 0, arrow=LAST)

        self.sideMonitorFrame.create_line(int(self.CWidth/30), int(self.CHeight), int(self.CWidth/30), 0, arrow=LAST)

        #plotting marks on graduation
        self.actualize_graduationUP()
        self.actualize_graduationSide()

        #creation of button area
        frame = Frame(self.frame)
        frame.grid(row=0, column=0, sticky="n")

        #Label(frame, text="dist").grid(row=0, column=0, sticky=W)
        #s1 = Scale(frame, from_=0, to=500, orient=VERTICAL, command=self.upHull.moveHull)
        #s1.grid(row=0, column=1, columnspan=2)
        #s1.set(self.CHeight)

        Label(frame, text="Angle").grid(row=1, column=0, sticky=W)
        s1 = Scale(frame, from_=0, to=359, orient=VERTICAL, command=self.upHull.rotateHull)
        s1.grid(row=1, column=1, columnspan=2)
        s1.set(0)

        Button(frame, text='Feu !', width=20, command=self.testMove).grid(row=5, column=0, columnspan=2)
        Button(frame, text='Define trajectory', width=20, command=self.define_trajectory).grid(row=6, column=0, columnspan=2)
        Button(frame, text='Stop defining', width=20, command=self.stop_defining).grid(row=7, column=0, columnspan=2)

        self.frame.bind_class("Canvas", "<Button-1>", self.on_click)

        self.frame.mainloop()


    def on_click(self, evt):
        if self.isDefiningTraj:
            if self.isDefiningTrajUp:
                print("Position de la souris:", evt.x, evt.y)
                self.xytrajectory.append([evt.x, evt.y])
                self.upMonitorFrame.configure(bg="white")
                self.sideMonitorFrame.configure(bg='#ffcccc')
                self.isDefiningTrajUp = False
                pass
            else:
                print("Position de la souris:", evt.x, evt.y)
                self.ztrajectory.append
                self.sideMonitorFrame.configure(bg="white")
                self.upMonitorFrame.configure(bg='#ffcccc')
                self.isDefiningTrajUp = True


    def actualize_defined_trajectory(self):
        '''
        Plot trajectory in fonction of zoom coef
        :return:
        '''
        #delete old trajectory
        for i in range(len(self.trajectoryLines)):
            self.upMonitorFrame.delete(self.trajectoryLines[i])

        #plot newone
        for i in range(len(self.trajectory)-1):
            #before all, convert trajectory coordinate in meter, in pixels for plotting
            x0Conv, y0Conv = self.convert_coord_in_pix(self.zoomCoefup, self.trajectory[i][0], self.trajectory[i][1])
            x0Conv = x0Conv + int(self.upMonitorFrame.cget('height'))/2
            y0Conv = y0Conv + int(self.upMonitorFrame.cget('width'))/2

            x1Conv, y1Conv = self.convert_coord_in_pix(self.zoomCoefup, self.trajectory[i+1][0], self.trajectory[i+1][1])
            x1Conv = x1Conv + int(self.upMonitorFrame.cget('height'))/2
            y1Conv = y1Conv + int(self.upMonitorFrame.cget('width'))/2

            #plot line between converted points
            self.trajectoryLines.append(self.upMonitorFrame.create_line(x0Conv, int(self.upMonitorFrame.cget('height'))-y0Conv,
                                                                        x1Conv, int(self.upMonitorFrame.cget('height'))-y1Conv))

    def stop_defining(self):
        self.isDefiningTraj = False
        self.upMonitorFrame.configure(bg="white")

    def define_trajectory(self):
        self.isDefiningTraj = True
        if self.isDefiningTrajUp:
            self.upMonitorFrame.configure(bg='#ffcccc')


    def actualize_graduationUP(self):
        '''
        plotting graduation mark function of zoom coef for above view
        '''
        #Delete old mark
        for i in range(len(self.graduationsUP)):
            self.upMonitorFrame.delete(self.graduationsUP[i])

        #plotting new mark
        x = self.CHeight / (10 * self.zoomCoefup)
        for i in range(1, 10 * self.zoomCoefup):
            #print('i : '+ str(i) +' x0 : ' + str((self.CHeight/2-10))+' y0 : ' + str(int(i * x))+' x1 : ' + str((self.CHeight/2+10))+' y1 : ' + str(int(i * x)))
            #print('testgrad')
            self.graduationsUP.append(self.upMonitorFrame.create_line((self.CHeight/2-10), int(i * x), (self.CHeight/2+10),
                                                              int(i * x)))
            self.graduationsUP.append(self.upMonitorFrame.create_line(int(i * x), (self.CHeight/2-10), int(i*x),
                                                                (self.CHeight / 2 + 10)))
            
    def actualize_graduationSide(self):
        '''
        plotting graduation mark function of zoom coef for above view
        '''
        #Delete old mark
        for i in range(len(self.graduationsSide)):
            self.sideMonitorFrame.delete(self.graduationsSide[i])

        #plotting new mark
        x = self.CHeight / (10 * self.zoomCoefside)
        for i in range(1, 10*self.zoomCoefside):
            #print('i : '+ str(i) +' x0 : ' + str((self.CHeight/2-10))+' y0 : ' + str(int(i * x))+' x1 : ' + str((self.CHeight/2+10))+' y1 : ' + str(int(i * x)))
            #print('testgrad')
            self.graduationsSide.append(self.sideMonitorFrame.create_line(int(self.CWidth/30-10), int(i * x), int(self.CWidth/30+10),
                                                              int(i * x)))


    def actualize_trajectory(self):
        '''
        Plot trajectory in fonction of zoom coef
        :return:
        '''
        #delete old trajectory
        for i in range(len(self.trajectoryLines)):
            self.upMonitorFrame.delete(self.trajectoryLines[i])

        #plot newone
        for i in range(len(self.trajectory)-1):
            #before all, convert trajectory coordinate in meter, in pixels for plotting
            x0Conv, y0Conv = self.convert_coord_in_pix(self.zoomCoefup, self.trajectory[i][0], self.trajectory[i][1])
            x0Conv = x0Conv + int(self.upMonitorFrame.cget('height'))/2
            y0Conv = y0Conv + int(self.upMonitorFrame.cget('width'))/2

            x1Conv, y1Conv = self.convert_coord_in_pix(self.zoomCoefup, self.trajectory[i+1][0], self.trajectory[i+1][1])
            x1Conv = x1Conv + int(self.upMonitorFrame.cget('height'))/2
            y1Conv = y1Conv + int(self.upMonitorFrame.cget('width'))/2

            #plot line between converted points
            self.trajectoryLines.append(self.upMonitorFrame.create_line(x0Conv, int(self.upMonitorFrame.cget('height'))-y0Conv,
                                                                        x1Conv, int(self.upMonitorFrame.cget('height'))-y1Conv))


    def testMove(self):
        '''
        Test function to be remove
        :return:
        '''
        self.moveHull(self.xUp, self.yUp, True)
        self.moveHull(self.ySide, 0, upBalloon=False)
        self.rotateUpHull(self.testtheta*360/(2*np.pi)+90)
        self.rotateSideHull(self.testtheta*360/(2*np.pi))
        self.testtheta = self.testtheta + 0.02
        self.testR = self.testR + 0.1
        self.xUp = self.testR * np.cos(self.testtheta)
        self.yUp = self.testR * np.sin(self.testtheta)
        self.ySide = self.ySide + 0.01

        self.frame.after(30, self.testMove)


    def convert_coord_in_pix(self, zoomCoef, x, y=0):
        '''
        Given x and y coord in meter will return coord in pixel fonction of the zoom of window
        :param zoomCoef: zoom coefficient
        :param x: x coord in meter
        :param y: y coord in meter
        :return: x and y in pixel
        '''
        zoom = int(self.upMonitorFrame.cget('height'))/(zoomCoef*10)
        return x*zoom, y*zoom


    def convert_pix_in_coord(self, zoomCoef, x, y=0):
        '''
        Given x and y coord in meter will return coord in pixel fonction of the zoom of window
        :param zoomCoef: zoom coefficient
        :param x: x coord in pix
        :param y: y coord in pix
        :return: x and y in meter
        '''
        zoom = int(self.upMonitorFrame.cget('height'))/(zoomCoef*10)
        return x/zoom, y/zoom


    def rotateUpHull(self, angle):
        self.upHull.rotateHull(angle)


    def rotateSideHull(self, angle):
        self.sideHull.rotateHull(angle)


    def moveHull(self, xmeter, ymeter = 0, upBalloon = True):
        '''
        function moving the hull
        :param xmeter: (float) X coordinate to move on in meter
        :param ymeter: (float) Y coordinate to move on in meter
        :param upBalloon: (bool) if true -> above view, else side view
        :return:
        '''

        if upBalloon:
            self.xUp = xmeter
            self.yUp = ymeter
            xConv, yConv = self.convert_coord_in_pix(self.zoomCoefup, xmeter, ymeter)
            xConv = xConv + int(self.upMonitorFrame.cget('height'))/2
            yConv = yConv + int(self.upMonitorFrame.cget('width'))/2
            #print('upBalloon : ' + str(xmeter) + ' ; ' +str(ymeter))

            if not (not (xConv > int(self.upMonitorFrame.cget('height'))) and not (
                    yConv > int(self.upMonitorFrame.cget('width'))) and not (xConv < 0)) or yConv < 0:

                self.zoomCoefup = self.zoomCoefup + 1
                self.moveHull(xmeter, ymeter, upBalloon = True)
                self.actualize_graduationUP()
                self.actualize_trajectory()

            else:
                #Making trajectory
                self.trajectory.append([xmeter, ymeter])
                print('len : ' + str(len(self.trajectory)))

                x0Conv, y0Conv = self.convert_coord_in_pix(self.zoomCoefup, self.trajectory[-2][0],
                                                           self.trajectory[-2][1])
                x0Conv = x0Conv + int(self.upMonitorFrame.cget('height')) / 2
                y0Conv = y0Conv + int(self.upMonitorFrame.cget('width')) / 2

                x1Conv, y1Conv = self.convert_coord_in_pix(self.zoomCoefup, self.trajectory[-1][0],
                                                           self.trajectory[-1][1])
                x1Conv = x1Conv + int(self.upMonitorFrame.cget('height')) / 2
                y1Conv = y1Conv + int(self.upMonitorFrame.cget('width')) / 2

                self.trajectoryLines.append(self.upMonitorFrame.create_line(x0Conv, int(self.upMonitorFrame.cget('height')) - y0Conv,
                                                                            x1Conv, int(self.upMonitorFrame.cget('height')) - y1Conv))

                self.upHull.moveHull(xConv, int(self.upMonitorFrame.cget('height')) - yConv)

        if not upBalloon:
            self.ySide = xmeter
            yConv, _ = self.convert_coord_in_pix(self.zoomCoefside, xmeter)
            yConv = yConv + int(self.upMonitorFrame.cget('width')) / 10
            print('sideBalloon : ' + str(xmeter))

            if yConv < 0 or yConv > int(self.upMonitorFrame.cget('width')):
                self.zoomCoefside = self.zoomCoefside + 1
                self.moveHull(xmeter, upBalloon=False)
                self.actualize_graduationSide()

            else:
                self.sideHull.moveHull(int(self.upMonitorFrame.cget('height'))-yConv)



if __name__ == '__main__':
    monitorSimple(1000, 300)