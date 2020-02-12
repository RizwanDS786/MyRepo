# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 14:57:16 2019

@author: harsh
"""

"""
Let’s say you have two SQL tables: authors and books.
The authors dataset has 1M+ rows; here’s the first six rows:
    
    AUTHORS TABLE

author_name	book_name
author_1	book_1
author_1	book_2
author_2	book_3
author_2	book_4
author_2	book_5
author_3	book_6
…	…
The books dataset also has 1M+ rows and here’s the first six:
    
    BOOKS TABLE

book_name  sold_copies
book_1	   1000
book_2	   1500
book_3	   34000
book_4	   29000
book_5	   40000
book_6	   4400
…	…
Create an SQL query that shows the TOP 3 authors who sold the most books in total!"""




"""
SOLUTION

select
author_name,sum(solf_copies) from
authors inner join
books
on authors.book_name==books.book_name
group by author_name
LIMIT 3



"""

____________________________________________________


"""


You work for a startup that makes an online presentation software. You have an event log 
that records every time a user inserted an image into a presentation. 
(One user can insert multiple images.) The event_log SQL table looks like this:
    
    PRESENTATION TABLE

user_id	event_date_time
7494212	1535308430
7494212	1535308433
1475185	1535308444
6946725	1535308475
6946725	1535308476
6946725	1535308477
…	…
…and it has over one billion rows.
Note: If the event_date_time column’s format doesn’t look familiar, google “epoch timestamp”!

Write an SQL query to find out how many users inserted more than 1000 but less than 2000 images
 in their presentations!
 
"""


"""
SOLUTION


(select
user_id,count(*) from prsent_table
gorup by user_id
having count>1000 and count,2000)







"""

__________________________________________________________________________________


"""
You have two SQL tables! The first one is called employees and it contains the employee names, 
the unique employee ids and the department names of a company. Sample:

             Employees
department_name     	employee_id	           employee_name
Sales	                  123	                John Doe
Sales	                  211	                Jane Smith
HR	                      556	                Billy Bob
Sales	                  711	                Robert Hayek
Marketing	              235	                Edward Jorgson
Marketing	              236	                Christine Packard
…	…	…
The second one is named salaries. It holds the same employee names and the same employee ids – and the salaries for each employee. Sample:
          Salaries
salary	              employee_id	                      employee_name
500	                     123	                           John Doe
600	                     211	                           Jane Smith
1000	                 556	                           Billy Bob
400	                     711	                           Robert Hayek
1200	                 235	                           Edward Jorgson
200	                     236	                           Christine Packard
…	…	…
The company has 546 employees, so both tables have 546 rows.

Print every department where the average salary per employee is lower than $500!





select department_name,avg(salary) from
employees
inner join
salaries on
employees.employyeid=salaries.employee_id
group by  
department_name
hvaing avg(Salalry)<500



"""

"""

SOLUTION

select 
department_name,avg(salary)
from employees
inner join
salaries
on employees.employee_id=salaries.employee_id
group by department_name 
having avg(salary)<500




"""

______________________________________


"""
#Foundation medicine interview questions

string="I am a very good boy"

Count the number of times each and every character appers and sort them by order


SQL Query:

Student:
Student Id
Student Name
Professor Id

1001-1
1002-2
1001-2
1003-2

Professor:
Professor Id   
Professor Name

1-Harsha-1001
2-rizwan-1002
2-rizwan -1001
2-Rizwan-1003


SQL Query to find out professor name who mave more than 10 students order by the count
of number of students

select
ProfessorName,count(*) as c from Professor
inner join
student 
on Professor.ProfessorId=Student.ProfessorId
group by Professor_id


Professor C
Harsha   1
Rizwan   3





"""



______________________________________________________________


"""

Desgin a schema for sports league


Sports league database design. 
players are employees, belong to multiple teams, team belongs to league, 
games are played between teams.


Query for all teams and leagues that a player given first and last name belongs to.
Query all players on al teams that play at a particular position
query all players that are not on a team.

"""


"""
Solution


     Players

Player_id
Player_first_anme
Player_flast_name
Team_id
Position




Team
Team_Id
Team_name
League_id


League
League_id
League_Name


Matches
Match_Id
Team_id_1
Team_Id_2
League_Id



select
player_fn,player_sn
from players
inner join team on players.team_id=team.team_id
inner join lagaue on team.teague-di=leage.legae-Id'
eher forst_nae= and secondname

"""


_____________________________________



"""
Data Base Schema Design - Movies  Theatre Booking system

We’re going to design a movies database.
 
Each movie has a title and year, one (and only one) director, 
and some number of actors. Actors can star in multiple movies

Movies             

MovieID        Movie Name        Movie_Year       Director_name

  

Actor:


Movie Id  Actor_id   Actor_name





Movies Booking system- With standrad prices

Customer go to many theater
Each theater can have money movies
duplicate movie name may exist


Customer
Customer Id -  MovieID- FK   Theatre_ID- FK Ticket_ID
1001                  1        10               1
1002                  1        9
1003                  2        5

Movies
MOvie Id    MOvie Name   
1             Varsham
2             Avengers


Tickets
Ticket_Id   Ticket_Type Ticket_Price
1            Class-1     50
2            Class-2     70
3            Class-3     100

Theatre
TheatreId- PK  MovieID Location Langitude -    Location Latitude -


"""
____________________________

"""
Data Base desin netflix

Movies ---. Directors


Movie 

1001 Baahubali   Rajamouli,VV.Vinayak    1,2
1002 Tagore      VV.Vinayak,Puri         2,3
1003 Srinu       Srinu Vaitla,Rajamouli  4,1



Movie

Movie Id
Movie Name


Movie Id     Movie Name     
1001        Baahubali       
1002        Tagore
1003        Srinu


Director_movie
Movie ID   Director_id       
1001       1       
1001       2
1002       2
1002       3
1003       4
1003       1


DirectorId    Director      Wife
 1              R          Drama
 2      



"""

_____________________

"""
Sports league Data base desgin- Wayfair


Each player is allowed to play for mutiple teams
Each team has mutiple players
Some players can be coaches

                                              


                                             CoachTable:
                                              CoachId
                                              CoachName


Player Table:
    
Player Id
player f_name
player l_name
                        Player-Team Table:
                            PlayerId
                            TeamId


Team Table:                                  
    Team Id           
    Team Strength
    Team Owner
    
    



Statistic Table:
    MatchId   TeamId   Score   
    1001       1       10  
    1001
    
    
    Schedule Table:
    MacthId - Date of match
    
    
Queires I remebers
--> How many of players who registered were not in any team
--> How many of players worked as coaches
--> Which team has players less than 10
--> Find the winners of all the matches using Staistics Table

     
    
    






















