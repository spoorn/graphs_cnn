#!/usr/bin/python

class EPSObject:
  def __init__(self):
    return

class EPSDict(EPSObject):
  def __init__(self):
    # number of defined terms
    self.n = 0
    # keys
    self.keys = []
    # values
    self.values = []


class EPSVectorPath(EPSObject):
  def __init__(self):
    EPSObject.__init__(self)
    self.n = 0
    self.x = []
    self.y = []
    self.t = []
    self.color = 0.0
    self.filled = False
    self.completed = False


class EPSImage(EPSObject):
  def __init__(self):
    EPSObject.__init__(self)
    self.setup = EPSDict()
    self.setup.n = 12
    self.setup.keys = [ "ImageType","Width","Height","Interpolate",\
                        "BitsPerComponent","Decode","DataSource",\
                        "ASCII85Decode","DCT/Flate","FlateDCTDecode",\
                        "ImageMatrix","setcolorspace","image/mask"]
    self.setup.values = [ 1, 0, 0, None, 1, [1,0], "currentfile",\
                          "/DCTDecode","currentfile","currentfile",\
                          [1,0,0,-1,0,0],None,"image"]
    self.encoded = None


