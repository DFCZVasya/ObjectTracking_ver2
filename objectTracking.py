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
        boxA = self.bbox
        boxB = newbbox
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return intersection
