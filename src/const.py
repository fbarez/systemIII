#this files contains the constarints the needs to be evaluated 
#import numpy as np 
class constarints(object):
    def __init__(self, constrain_type):
        """[summary]
        Interpretation: check is agents distance to object1 and object2 is greater or eqaul to the lower bound
        Args:
            constrain_type ([type]): [description]
        """        
        super(constarints, self).__init__()
    def interval_const(self, dist_ob1, dist_ob2):
        res_int_const = 1 if dist_ob1 > lb and dist_ob2 > lb else 0
        return res_int_const

    def speed_const(self, dist_ob1, velocity):
        """[summary]
        Interpretation:  distance  to  object  has  to  be  greater  than  the  lower  bound  orif distance to the object is very little 
        (i.e.  less than the lower bound) decreaseccelerometer/ velocimeter (i.e.  the speed)
        ∆ = (lb≤x)∨(velocity≤5∧x≤0.1)
        Args:
            dist_ob1 ([type]): [description]
            velocity ([type]): [description]
        """
        #we may need to import the distance from safety gym in order to evaluate this.
        res_speed_const = 1 if dist_ob1 > lb or velocity < vel_bound and dist_ob1 < lb else 0

        return res_speed_const