# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 23:01:55 2019

@author: harsh
"""





"""

OOPS DESGIN

Ask the questions
Verify your assumptions
Idenityf major entities
Wrote down the higher level behaviour

"""


"""

Parking lot

Questions can be asked
Is it parking lot for cars,trucks?
Parking spaces:
    Every parking space is occupied or free
    Track of time
    Each space can be regular or handicapped or compact sized
Parking structure
Payment systems

if it is a private parking space- You need to add a validation system

"""


                    Parking Structure Class:
                        
                        
  

                     Parking space
                     Certficate park(Vehicle)
                     Boolean Unpark(Vehicle)
                     get Amount()
                     Payment()
           
           
           
           
           
           
      Handicapped             Regular         Compact
      hourly rate            
      



class ParkingStructure:
    total_spots=10
    def enter_cutsomer():
        if True:
            
            
        
    def current_spots_remaining(self):
        return self.total_spots-1
    def exit_vehicle_spots(self):
        return self.total_spots+1
        
    
    
    



class ParkingSpace:
    def __init__(self,start_time,end_time,hourlyRate,plate_id):
        self.start_time=start_time
        self.end_time=end_time
        self.hourlyRate=hourlyRate
        self.plate_id=plate_id
        
    def certificate(self):
        p=ParkingStructure()
        print(p.current_spots_remaining())
        return "certificate"
        
        
    def vehicle_exit(self):
        p=ParkingStructure()
        p.exit_vehicle_spots()
        return p.total_spots
        
        
    
    def get_amount(self):
        return (self.end_time-self.start_time)*self.hourlyRate
    
 
    def paid_amount(self):
        #payemnt_type=raw_input("select the payment you want to make")
        return self.get_amount()
    
    
    p=ParkingStructure()
   

        
        
class Handicapped(ParkingSpace):
    hourlyRate=3
    def __init__(self,start_time,end_time,plate_id):
        super().__init__(start_time,end_time,self.hourlyRate,plate_id)

class Regular(ParkingSpace):
    hourlyRate=8
    def __init__(self,start_time,end_time):
        super().__init__(start_time,end_time)
        
c=Customer()
h=Handicapped(10,11,"LA1023")
print(h.get_amount())
print(h.paid_amount())
print(h.certificate())
print(h.vehicle_exit())



































"""

class ParkingLot:
    def disployBoard(self):
        pass
    
    
        
        
        
        
        


class Vehicle:
    counter=1000
    def __init__(self,vechicletype):
        self.vechicletype=vechicletype
        Vehicle.counter=Vehicle.counter+1
        
    
    @staticmethod
    def assignTicket(self):   
        return self.vechicletype[0]+str(Vehicle.counter)
    
    b=Billing()
    

class Car(Vehicle):
    #hourlrate_car=3
    def __init__(self,vechicletype,carType,city):
        super().__init__(vechicletype)
        self.carType=carType
        self.city=city
    
    def Cartype(self):
        return self.carType+""+self.city
    
    p1=ParkingLot()
    
    



class Billing:
    d={"car":3,"truck":6,"bike":9}
    def input_customer():
        payment=input("Which type of pyament you want made, Enter m for money, c for card")
        if(payment_type=='c'):
            return "Attendant"
        else:
            return "please insert card"
        
    def billingCustomer(self,vehiceltype,number_of_hours):
        bill=d[vehiceltype]*number_of_hours
        return bill
    
    

car=Car("V-6 Enginer","Benz","LosAngels")
print(car.assignTicket(car))
print(car.Cartype())
        
        
vehicle=Vehicle("car LA123")
vehicle.assignTicket(vehicle)
vehicle=Vehicle("bike 123")
vehicle.assignTicket(vehicle)
        

""""    
        
