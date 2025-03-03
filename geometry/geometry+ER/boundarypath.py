
import itertools, io, base64, random,string, os

import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import IPython.display as ipd


def boundaryPath(binarySlice, faceColor='none', edgeColor='r'):

    # collect edges of each lattice site
    # only keep boundary edges, if more than one edge traverses two points, it is internal

    faces = dict()
    def ap(*v):
        k = tuple(int(x) for x in v)
        if k in faces:
            faces[k] = None
        else:
            faces[k] = 1

    for x,y in itertools.product(*map(range,binarySlice.shape)):
        if binarySlice[x,y]:
            ap( x,y, x+1,y )
            ap( x,y, x,y+1 )
            ap( x,y+1, x+1,y+1 )
            ap( x+1,y, x+1,y+1 )


    faces2 = set((f[:2],f[2:]) for f,v in faces.items() if v is not None )

    # build map for looking up an edge based on its verticies
    vertexTable = { x:[] for x in  set(x for x,_ in faces2) | set( x for _,x in faces2) }

    for l,r in faces2:
        vertexTable[l].append((l,r))
        vertexTable[r].append((l,r))

    # merge edges into single closed path
    face = faces2.pop()
    paths = [ [ face[0], face[1] ] ]

    def faceEq(f0,f1):
        return (f0[0] == f1[0] and f0[1] == f1[1]) or (f0[1] == f1[0] and f0[0] == f1[1]) 

    while len(faces2) > 0:
        firstFace = paths[0][-2:]
        lastFace = paths[-1][-2:]
        pl, pr = lastFace
        for face in vertexTable[pr]:
            # append next vertex, making sure it's not already visited
            if not faceEq(face, lastFace) and not faceEq(face, firstFace):
                try:
                    vertexTable[pr].remove(face)
                    faces2.remove(face)
                except KeyError:
                    break

                if pr == face[0]:
                    paths[-1].append(face[1])
                else:
                    paths[-1].append(face[0])

                break
        else: 
            # start a new path if we ran out of edges to connect
            if len(faces2) > 0:
                face = faces2.pop()
                paths.append([ face[0], face[1] ])


    newPaths = []
    for path in paths:
        # set consistent winding order
        p0,p1 = path[0:2]
        if p0[0] == p1[0]:
            if p0[1] > p1[1]:
                path = path[::-1]
        else:
            if p0[0] > p1[0]:
                path = path[::-1]

        newPaths.append(path)

    # save path in format mpl can deal with

    verts, codes = [], []

    # reverse sort by arc length
    def pathKey(x):
        return -sum((x0-x1)**2+(y0-y1)**2 for (x0,x1),(y0,y1) in zip(x,[x[-1]] + x[:-1]) )

    for pidx,path in enumerate(sorted(newPaths,key=pathKey)):
        vs, cs = [], []
        for i,vertex in enumerate(path):
            vs.append( (vertex[1], vertex[0]) )
            if i == 0:
                cs.append( Path.MOVETO )
            else:
                cs.append( Path.LINETO )
        
        # assume longest path is external edge and reverse so that following paths subtract fill correctly
        # if there are "islands" inside cavities, they will have the wrong winding order.
        if pidx == 0:
            vs = vs[::-1]

        vs.append( (0,0) )
        cs.append( Path.CLOSEPOLY )
        verts += vs
        codes += cs

    return patches.PathPatch(Path(verts,codes), fc=faceColor, ec=edgeColor)
