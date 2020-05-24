import fcl

class Simplex(fcl.Simplex):
    def __init__(self, shape_id=None):
        super().__init__()
        # transforms the shape id into the object id ...