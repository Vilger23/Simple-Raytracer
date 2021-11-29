import tkinter as tk
from functools import reduce
import numpy as np 
from PIL import Image, ImageTk
import numbers

class Light:
    P = None

class v:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
    def __add__(self, other):
        return v(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return v(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, other):
        return v(self.x*other, self.y*other, self.z*other)
    def __truediv__(self, other):
        return v(self.x/other, self.y/other, self.z/other)

    def abs(self):
        return self.dot(self)
    def magnitude(self):
        return np.sqrt(self.abs())
    def normalized(self):
        mag = np.sqrt(self.abs())
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def dot(self, other):
        return self.x*other.x + self.y * other.y + self.z * other.z
    def extract(self, mask):
        return v(extract(mask, self.x), extract(mask, self.y), extract(mask, self.z))
    def set_rgb(self, mask):
        color = v(np.zeros(mask.shape), np.zeros(mask.shape), np.zeros(mask.shape))
        np.place(color.x, mask, self.x)
        np.place(color.y, mask, self.y)
        np.place(color.z, mask, self.z)
        return color
    def xyz(self):
        return self.x, self.y, self.z

def extract(Bool, obj):
    if isinstance(obj, numbers.Number):
        return obj
    else:
        return np.extract(Bool, obj)

class Sphere:
    def __init__(self, r, PosV, ColV, shininess, Reflected = 0.5, spec = v(1, 1, 1)):
        self.center = PosV
        self.r = r 
        self.Bcolor = ColV
        self.reflected = Reflected
        self.spec = spec
        self.shininess = shininess
        self.ambient = ColV * 0.1

    def get_color(self, p_rf):
        return self.Bcolor

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.center)
        c = self.center.abs() + O.abs() - 2 * self.center.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, to_far)

    def rgb(self, scene, origin, Vect, t, reflections, current_ir = 1):
        P_rf = origin + Vect*t
        N = (P_rf-self.center).normalized()
        L = (Light.P - P_rf).normalized()  
        I = (E - P_rf).normalized()
        color = self.ambient
        epsilon = 0.00001
        nudged = P_rf + L*epsilon

        intensity = np.maximum(N.dot(L), 0)
        shadow_rays = [s.intersect(nudged, L) for s in scene if not isinstance(s, Plane)]
        nearest_intersection = reduce(np.minimum, shadow_rays)
        SeeLight = nearest_intersection > (Light.P - P_rf).magnitude()
        color += self.get_color(P_rf) * intensity* SeeLight 

        if reflections < 1 and self.reflected != 0:
            Rv = (Vect - N * 2 * Vect.dot(N)).normalized()
            r_term = self.reflected
            color += trace(scene, nudged, Rv, reflections + 1)*r_term #(self.reflected + (1 - self.reflected)*((1-dot_prod)**5))

        h = (L + I).normalized()
        color += self.spec * np.power(np.clip(N.dot(h), 0, 1), self.shininess/4) #* SeeLight
        return color

class Plane(Sphere):
    def __init__(self, pos, norm, col, reflected):
        center = pos + norm.normalized()*99999
        Sphere.__init__(self, 99999, center, col, 0, reflected, v(0, 0, 0))   

class CheckeredPlane(Plane):
    def get_color(self, p_rf):
        white = (p_rf.x+100).astype(int) % 2 == (p_rf.z-100).astype(int) % 2
        return self.Bcolor*white

def trace(scene, origin, Vect, reflections = 0):
    t_vals = [s.intersect(origin, Vect) for s in scene]
    closests = reduce(np.minimum, t_vals)
    color = v(0, 0, 0)
    for (s, t) in zip(scene, t_vals):
        seen = (t == closests) & (closests != to_far)
        if np.any(seen):
            pt = extract(seen, t)
            pOr = origin.extract(seen)
            pVe = Vect.extract(seen)
            pC = s.rgb(scene, pOr, pVe, pt, reflections)
            color += pC.set_rgb(seen)
    return color

Light.P = v(5, 5, 0)
to_far = 10.0**39
E = v(0, 0, -2) 

width = 400
aspect_ratio = 4, 4
r = aspect_ratio[0]/aspect_ratio[1]
height = int(width/r)

scene = [
    Sphere(2, v(0, 0, 10), v(1, 0.8, 0), 100, 0.5),
    CheckeredPlane( v(0, -2, 0), v(0, -1, 0), v(0.7, 0.7, 0.7), 0),
    #Plane( v(0, 0, 20), v(0, 0, 1), v(0.4, 0.4, 0.8), 0),
    #Plane( v(0, 0, -5), v(0, 0, -1), v(0.8, 0.4, 0.4), 0),
    #Plane( v(5, 0, 0), v(1, 0, 0), v(0.4, 0.8, 0.4), 0),
    #Plane( v(-5, 0, 0), v(-1, 0, 0), v(0.4, 0.4, 0.8), 0),
    #Plane( v(0, 7, 0), v(0, 1, 0), v(0.8, 0.8, 0.4), 0)
]

def mouseclick(event, scene):
    x, y = 2*event.x/width - 1, (height - event.y)/height*(2/r) - 1/r

    t = [s.intersect(E, (v(x, y, 0)-E)) for s in scene]
    t_close = np.amin(t)
    for (s, t) in zip(scene, t):
        if (t == t_close) and (t != to_far):
            obj = s
    P = E + (v(x,y,0)-E)*t_close
    vec = (P - obj.center).normalized()
    Light.P = P + vec*4
    render()

def render():
    x = np.tile(np.linspace(-1, 1, width), height)
    y = np.repeat(np.linspace(1/r, -1/r, height), width)
    Vect = (v(x, y, 0) - E).normalized()
    color = trace(scene, E, Vect)
    rgb = [Image.fromarray((255*np.clip(c, 0, 1).reshape((height, width))).astype(np.uint8), "L") for c in color.xyz()]
    img = Image.merge("RGB", rgb)#.resize((int(width/2), int(height/2)))
    IMAGE = ImageTk.PhotoImage(image=img)

    label.config(image=IMAGE)
    label.pack()
    root.update()

root = tk.Tk()
label = tk.Label(root)
render()
frames = 50
i=0
while True:
    i += 1
    theta = np.pi*2/frames
    Light.P = v(np.cos(theta*i)*6, 2, np.sin(theta*i)*6+10)
    render()
    
root.bind('<Button-1>', lambda x : mouseclick(x, scene))
root.mainloop()