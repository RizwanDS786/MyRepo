# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:17:35 2019

@author: harsh
"""

#"HotelID","HotelName","CityID","AreaID","StarClass","NbRooms","Amenities","Daily","Reviews","PageUrl"

import requests
import random
import time
import sys
from bs4 import BeautifulSoup
url='https://www.tripadvisor.com'
base_url='/'+'Restaurants-g60745-Boston_Massachusetts.html'
timeDelay=random.randrange(0,20)
import re
def all_pages(url,base_url):
    Trip_Advisor_Restaurants_Pages=[url+base_url]
    initial_page_restaurants=requests.get(url+base_url)
    #time.sleep(timeDelay)
    while(initial_page_restaurants.status_code!=200):
        initial_page_restaurants=requests.get(url+base_url)
    soup=BeautifulSoup(initial_page_restaurants.text,'html.parser')
    next_available_page=soup.find(class_="nav next rndBtn ui_button primary taLnk")
    while(next_available_page!=None):
        updated_url=url+next_available_page['href']
        print(updated_url)
        next_pages_requests=requests.get(updated_url)
        soup2=BeautifulSoup(next_pages_requests.text,'html.parser')
        next_available_page=soup2.find(class_="nav next rndBtn ui_button primary taLnk")
        Trip_Advisor_Restaurants_Pages.append(updated_url)
      
    return Trip_Advisor_Restaurants_Pages

Pages_Date=all_pages(url,base_url)
        
def locations(restaurent_url):
    hotel_page=requests.get(restaurent_url)
    soup=BeautifulSoup(hotel_page.text,'html.parser')
    locations=soup.find(class_="detail ")
    Restaurant_Location=locations.text
    no_of_reviews=soup.find(class_="reviewCount")
    if(no_of_reviews!=None):
        Reviews_Count=no_of_reviews.text.split('reviews')[0]
    else:
        Reviews_Count='NA'
    restaurant_rating=soup.find(class_="prw_rup prw_common_bubble_rating rating")
    if(restaurant_rating!=None):
        Restaurant_Rating=int(restaurant_rating.find('span')['class'][1].split('_')[1])/10
    else:
        Restaurant_Rating='NA'
    all_options=soup.find_all(class_="restaurants-details-card-TagCategories__categoryTitle--28rB6")
    all_options_features=soup.find_all(class_="restaurants-details-card-TagCategories__tagText--Yt3iG")
    Restaurant_Features='NA'
    Restaurant_Special_Diets='NA'
    for i in range(0,len(all_options)):
        if(all_options[i].text=="FEATURES"):
            Restaurant_Features=all_options_features[i].text
        elif(all_options[i].text=="Special Diets"):
            Restaurant_Special_Diets=all_options_features[i].text
    other_ratings=soup.find_all(class_="restaurants-detail-overview-cards-RatingsOverviewCard__ratingQuestionRow--5nPGK")
    bubble_ratings=soup.find_all(class_="restaurants-detail-overview-cards-RatingsOverviewCard__ratingBubbles--1kQYC")
    Food_Rating='NA'
    Service_Rating='NA'
    Value_Rating='NA'
    Ambiance_Rating='NA'
    for i in range(0,len(other_ratings)):
        if(other_ratings[i].text=="Food"):
            Food_Rating=int(bubble_ratings[i].find('span')['class'][1].split('_')[1])/10
        elif(other_ratings[i].text=="Service"):
            Service_Rating=int(bubble_ratings[i].find('span')['class'][1].split('_')[1])/10
        elif(other_ratings[i].text=="Value"):
            Value_Rating=int(bubble_ratings[i].find('span')['class'][1].split('_')[1])/10
        elif(other_ratings[i].text=="Atmosphere"):
            Ambiance_Rating=int(bubble_ratings[i].find('span')['class'][1].split('_')[1])/10
            
    return{
            "Restaurant_Location":Restaurant_Location,"Reviews_Count":Reviews_Count,"Restaurant_Rating":Restaurant_Rating,
            "Restaurant_Features":Restaurant_Features,"Restaurant_Special_Diets":Restaurant_Special_Diets,"Food_Rating":Food_Rating,"Service_Rating":Service_Rating,"Value_Rating":Value_Rating,"Ambiance_Rating":Ambiance_Rating,
            
             
            }
    
 
def  Restaurant_Details(url,Pages_Date):
    RestaurantDetails={}
    for  i in range(0,len(Pages_Date)):
        print(Pages_Date[i])
        pages_requests=requests.get(Pages_Date[i])
        time.sleep(timeDelay)
        while(pages_requests.status_code!=200):
            print("request failed.. Retrying URL")
            time.sleep(timeDelay)
            pages_requests=requests.get(Pages_Date[i])
        soup3=BeautifulSoup(pages_requests.text,'html.parser')
        Cusines=[]
        restaurant_containers=soup3.find_all(class_="ui_column is-9 shortSellDetails")
        for i in range(0,len(restaurant_containers)):
            Restaurant_Name=re.sub(r'\s+','',restaurant_containers[i].find(class_="title").text.rstrip())            
            if((restaurant_containers[i].find(class_="ui_merchandising_pill sponsored_v2 small"))!=None):
                restaurant_url=restaurant_containers[i].find('a')['data-url']
                Restaurant_Id=restaurant_url.split('-')[2][1:]
                Area_Id=restaurant_url.split('-')[1][1:]
            else:
                restaurant_href=restaurant_containers[i].find('a',class_="property_title")['href']
                restaurant_url=url+restaurant_href
                Restaurant_Id=restaurant_containers[i].find(class_="property_title")['href'].split('-')[2][1:]
                Area_Id=restaurant_containers[i].find(class_="property_title")['href'].split('-')[1][1:]
            
            other_details=locations(restaurant_url)
            if(restaurant_containers[i].find(class_="item price")!=None):
                Restaurant_Cost=(restaurant_containers[i].find(class_="item price")).text
            else:
                Restaurant_Cost='NA'
            cusines_class=restaurant_containers[i].find_all(class_="item cuisine")
            Cusines.append([])
            for j in range(0,len(cusines_class)):
                Cusines_Type=cusines_class[j].text
                Cusines[i].append(Cusines_Type)
            #print(other_details['Location'])
            Results={
                    Restaurant_Id:{
                            "Restaurant_Name":Restaurant_Name,"Area_ID":Area_Id,
                            "Cost":Restaurant_Cost,"Cusines":",".join(Cusines[i]),
                            "Reviews_Count":other_details['Reviews_Count'],
                            "Restaurant_Rating":other_details['Restaurant_Rating'],
                            "Restaurant_Location":other_details['Restaurant_Location'],
                            "Restaurant_Features":other_details['Restaurant_Features'],
                            "Restaurant_Special_Diets":other_details['Restaurant_Special_Diets'],
                            "Food_Rating":other_details['Food_Rating'],
                            "Service_Rating":other_details['Service_Rating'],
                            'Value_Rating':other_details['Service_Rating']
                            }
                    }
            RestaurantDetails.update(Results)


           # print(RestaurentDetails)
    return RestaurantDetails    
rd=Restaurant_Details(url,Pages_Date)


import pickle
def Pickle(rd):
    pickle.dump(rd,open("C:\\Users\\harsh\\OneDrive\\Desktop\\TripAdvisor\\Restuarents_Data\\Resturanet_Info.p","wb"))
     
Pickle(rd)


import pandas as pd
df=pd.DataFrame.from_dict(rd,orient='index')
df.to_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\TripAdvisor\\Restuarents_Data\\Resturanet_Info.csv",sep=",",index_label='Restuarent_ID',header=True)
      