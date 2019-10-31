import random

class ObjectTracking(object) :

    def __init__(self, classID):
        self.classID = classID
        self.probability = 1
        self.bbox = []
        self.id = 0
        self.counter = 1


    def createNewID(self, bbox, allObjects):
        id = random.randint(1,9999)
        unicID = False

        while unicID != True and len(allObjects) != 0:
            for object in allObjects:
                if object.id == id:
                    id = random.randint(1,9999)
                    unicID = False
                else:
                    unicID = True

        self.id = id
        self.bbox = bbox

    def tracking(self, newbbox, k, ex):
        if ex == False:
            if k >= 0.5:
                self.counter += 1
                if self.probability <= 10:
                    self.probability += 1
                    self.bbox = newbbox
                    return 0
        else:
            #self.counter -= 1
            self.probability -= 1
            if self.probability <= 0:
                return self.id

    def getcount(self):
        return self.counter

    def getbbox(self):
        return self.bbox

    def getClassName(self):
        return self.classID

    def getIntersection(self, newbbox):
        oldx = range(self.bbox[0], self.bbox[2])
        oldy = range(self.bbox[1], self.bbox[3])
        newx = range(newbbox[0], newbbox[2])
        newy = range(newbbox[1], newbbox[3])
        xintersection = []
        yintersection = []

        for pixel in newx:
            if pixel in oldx:
                xintersection.append(pixel)

        for pixel in newy:
            if pixel in oldy:
                yintersection.append(pixel)
        #print(xintersection)
        #print(yintersection)
        if len(xintersection) != 0 and len(yintersection) != 0:
            sqold = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
            xintersectionMin = min(xintersection)
            yintersectionMin = min(yintersection)
            xintersectionMax = max(xintersection)
            yintersectionMax = max(yintersection)
            sqintersection = (xintersectionMax - xintersectionMin) * (yintersectionMax - yintersectionMin)

            k = sqintersection / sqold
            #print(k)
            return k

        else:
            #print('0')
            return 0
