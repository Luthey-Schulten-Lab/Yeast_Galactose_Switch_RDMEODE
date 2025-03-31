from functools import reduce
import json, itertools, io, base64, colorsys
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

class VolData:
    def __init__(self,basename, transpose=[0,1,2]):
        self.basename = basename
        self.name = basename.split('/')[-1]
        self.scalar = np.transpose(np.load(basename +".npy")).transpose(transpose)
        self.attr = json.load(open(basename +".json"))
        self.dx = np.array([self.attr["apix_x"],
                            self.attr["apix_y"],
                            self.attr["apix_z"]])
        self.dx = self.dx[transpose]
        self.shape = np.array(self.scalar.shape)
        self.maxVal = np.max(self.scalar)
        self.minVal = np.min(self.scalar)
        if self.maxVal-self.minVal > 0:
            self._thumb()

    def _thumb(self):
        dims = np.array(self.shape)
        axis0 = np.argmax(dims)
        dims[axis0] = -1
        axis1 = np.argmax(dims)
        dims = np.array(self.shape)
        if dims[axis0] == dims[axis1]:
            axis0,axis1 = min(axis0,axis1), max(axis0,axis1)
        else:
            axis0 = axis0 if dims[axis0]>dims[axis1] else axis1
            axis1 = axis1 if dims[axis0]>dims[axis1] else axis0
        axis2 = (set((0,1,2)) - set((axis0,axis1))).pop()

        # find most interesting slice from image entropy

        maxEntropy, planeIndex, sl = float("-inf"), 0, None

        for i in range(dims[axis2]):
            s = [slice(None), slice(None), slice(None)]
            s[axis2] = i
            im = self.scalar[s]
            h = 1.0*np.histogram(im,bins=10)[0]
            h /= np.sum(h)
            e = sum(-x*np.log(x) for x in h if x > 0)

            if e > maxEntropy:
                maxEntropy, imBest, planeIndex, bestSlice = e, im, i, s


        self.maxEntropySlice = bestSlice
        text = [None,None,None]
        text[axis0] = "y"
        text[axis1] = "x"
        text[axis2] = str(planeIndex)
        caption = "slice: [{}, {}, {}]".format(*text)

        f = io.BytesIO()
        im = imBest - imBest.min()
        im *= 255/im.max()
        im = im.astype(np.uint8)
        im = Image.fromarray(im)
        im.thumbnail((256,256), Image.ANTIALIAS)
        rgb = Image.new("RGB", im.size)
        rgb.paste(im)

        draw = ImageDraw.Draw(rgb)
        # font = ImageFont.truetype('/usr/share/fonts/TTF/FreeSans.ttf', 16)
        draw.text( (5,5), caption, fill='rgb(200,0,0)')
        rgb.save(f,'PNG' )
        self._thumbB64 = base64.b64encode(f.getvalue()).decode()

    def showAsLabeled(self, axis, plane):
        if not hasattr(self, "labels"):
            self.labels = self.scalar.astype(int)

        labelSet = np.bincount(self.labels.ravel())
        if labelSet.size > 255:
            raise RuntimeError("Probably not a labeled map")



        colorMap = {i: np.array([int(255*c) for c in colorsys.hsv_to_rgb((i-1)/(labelSet.size-1), 0.9, 1)], dtype=np.uint8) for i in range(1,len(labelSet))}
        colorMap[0] = np.zeros(3, dtype=np.uint8)

        sl = [slice(None)]*3
        sl[axis] = plane
        array = self.labels[sl]
        u = axis
        v = (u+1)%3
        w = (u+2)%3
        rgb = np.zeros((self.shape[v], self.shape[w], 3), dtype=np.uint8)
        for i,j in itertools.product(range(rgb.shape[0]), range(rgb.shape[1])):
            rgb[i,j,:] = colorMap[array[i,j]]

        return Image.fromarray(rgb)




    
    def _repr_html_(self):
        s = ("<img src=\"data:image/png;base64,{}\" />".format(self._thumbB64)
             + "<table>"
             + " <caption>{}</caption>"
             + " <tr><th>Attribute</th><th>Value</th></tr>" ).format(self.name)

        for k,v in sorted(self.attr.items()):
            s += " <tr><td>{}</td><td>{}</td></tr>".format(k,v)
        s += "</table>"
        return s

    def show(self, axis, plane):
        sl = [slice(None)]*3
        sl[axis] = plane
        array = self.scalar[sl]

        f = io.BytesIO()
        data = array - np.min(array)
        data *= 255/data.max()
        data = data.astype(np.uint8)
        return Image.fromarray(data)


    def __call__(self, xin):
        evals = reduce(lambda x,y: x*y, xin.shape[1:])

        dx = self.dx

        x = xin.reshape((xin.shape[0], evals)).T
        n0s = np.array(x/dx, dtype=int)
        n1s = n0s+1

        ns = np.array([n0s,n1s])
        ws = np.array([dx*n1s-x, x-dx*n0s])
        
        f= self.scalar
        oob = np.any((n0s < 0) | (n1s >= self.shape), axis=1)
        ns[:,oob,:] = 0

        acc = None
        for i,j,k in itertools.product([0,1],repeat=3):
            pacc = np.where(oob, 0, f[ns[i,:,0],ns[j,:,1],ns[k,:,2]])
            for d,s in enumerate([i,j,k]):
                pacc *= ws[s,:,d]
            if acc is None:
                acc = pacc
            else:
                acc += pacc

        interp = acc/(self.dx[0]*self.dx[1]*self.dx[2])

        return interp.reshape(xin.shape[1:])





        

