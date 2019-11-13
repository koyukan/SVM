import pickle

class Point:
    def __init__(self):
        self.x = 3
        self.y = 3

    def __str__(self):
        return "Point: " + str(self.x) + "-" + str(self.y)


'''
p = Point()
print(p)
pickle.dump(p, open("obj.p", "wb"))'''

p = pickle.load(open("obj.p", "rb"))

print(p)