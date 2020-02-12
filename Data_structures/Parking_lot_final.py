# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:20:48 2019

@author: harsh
"""


"""
Parking Lot
Parking Space->Regular,Compact,Handicapped
Payment





""" 

class Billing:
    cost=10
    def __init__(self,start_time,end_time):
        self.start_time=start_time
        self.end_time=end_time
    
    def get_payment(self):
        return (self.end_time-self.start_time)*(self.cost)
        
    
    def payment(self):
        return self.get_payment()
    
class ParkingSpace:
    parking_space={"1001":True,"1002":True}
    def __init__(self,plate_number,parking_spot,start_time,end_time):
        self.parking_spot=parking_spot
        self.plate_number=plate_number
        self.start_time=start_time
        self.end_time=end_time
    
    def parking_status(self):
        self.parking_space.update({self.parking_spot:False})
        return self.parking_space
    
    def billing(self):
        billing=Billing(self.start_time,self.end_time)
        self.parking_space.update({self.parking_spot:True})
        print(self.parking_space)
        return billing.get_payment()



    

p=ParkingSpace("LA101","1001",8,10)
print(p.billing())
    
    

    
    
