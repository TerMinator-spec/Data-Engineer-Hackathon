
#Import the necessary modules
import pandas as pd
import numpy as np


class Extract_features:
   
    # init method or constructor 
    def __init__(self, vist_data, userdata):
        self.vist_data = vist_data
        self.userdata = userdata
   
    def preprocess_data(self):
        #Load the visitors data
        vis_data=pd.read_csv(self.vist_data)
        #Load the user data
        user_data=pd.read_csv(self.userdata)
        #join the two tables on user_id
        fin_data=pd.merge(vis_data, user_data,on='UserID')

        # fill the missing visitdatetime, productid, activity using ffill
        fin_data['VisitDateTime'] = fin_data['VisitDateTime'].ffill(axis=0)
        fin_data['ProductID'] = fin_data['ProductID'].ffill(axis=0)
        fin_data['Activity'] = fin_data['Activity'].ffill(axis=0)
        # Convert all the rows of visitdatetime into suitable date time format
        new_l=[] #list containing all the timestamps
        for i in range(fin_data.shape[0]):
          if(fin_data.iloc[i,1][4]=='-'): # checking if string is not unix
              new_l.append(pd.to_datetime(fin_data.iloc[i,1],unit='ns'))
          else: # if unix then convert str to int then to datetime
              new_l.append(pd.to_datetime(int(fin_data.iloc[i,1]),unit='ns'))
              
        # Add new_l to our dataframe also set date time as index
        fin_data['vis_time']=new_l
        fin_data=fin_data.set_index(fin_data['vis_time'])
        
        
        df_fin=fin_data.copy()
        
        # There are some changes needs to be done in df_fin
        #product id starts with "Pr" or 'pr'
        df_fin['ProductID']=df_fin["ProductID"].str.lower()
        # Change "CLICK" to 'click' and 'PAGELOAD' to 'pageload'
        df_fin['Activity']=df_fin["Activity"].str.lower()
        # all the OS in lowercase
        df_fin['OS']=df_fin["OS"].str.lower()
        # all the browsers in lowercase
        df_fin['Browser']=df_fin["Browser"].str.lower()
        
        # drop the country and city columns as we won't be using them for further analysis
        df_fin=df_fin.drop(['Country','VisitDateTime', 'City'], axis=1)
        return df_fin
        
    def solv_prob(self):
        X= self.preprocess_data()
        #Since the last day is 27 may 2018
        last_date=pd.Timestamp(2018, 5, 28, 0,0,0)
        
        a=pd.Timedelta(days=7) # getting time delta of past 7 days
        wkday=last_date-a # getting date of 7 days before today
        
        X['day']=X['vis_time'].dt.day # getting the day from datetime


        # Select values only last 7 days older 
        # count how many times a user visited the same day
        X_7=X[(X['vis_time']>wkday) & (X['vis_time']<last_date) & (X['Activity']!='no activity')].groupby(['UserID','day'])['ProductID'].count().reset_index()
        # count the number of unique days user visited
        X_q1=X_7.groupby('UserID')['day'].count().reset_index()
        
        X_q1=X_q1.rename(columns={'day':'No_of_days_Visited_7_Days'}) # required dataframe for no. of visits in last 7 days
        
        #Number of Products viewed by the user in the last 15 days
        b=pd.Timedelta(days=15) # getting time delta of past 15 days
        day15=last_date-b # getting date of 15 days before today
        
        X['vis_time']=pd.to_datetime(X['vis_time'])
        
        # Select values only last 15 days older 
        # Count the number of times user viewed a particular product
        X_15=X[(X['vis_time']>day15) & (X['vis_time']<last_date)].groupby(['UserID', 'ProductID'])['Activity'].count().reset_index()
        # COunt the unique number of products viewed by the user
        X_q2=X_15.groupby('UserID')['Activity'].count().reset_index()
        
        X_q2=X_q2.rename(columns={'Activity':'No_Of_Products_Viewed_15_Days'})
        
        # Vintage user
        user_data=pd.read_csv(self.userdata)
        X_vin=user_data.drop('User Segment', axis=1) # drop user segment from user_data
        
        X_vin['Signup Date']=pd.to_datetime(X_vin['Signup Date']) # Change to datetime format
        X_vin['Signup Date']=X_vin['Signup Date'].dt.tz_localize(None) #remove the timezone
        X_vin['user_vintage']=last_date-X_vin['Signup Date'] #find age of each user
        X_vin['user_vintage']=X_vin['user_vintage'].dt.days #find age in days
        
        X_q3=X_vin.copy()
        
        X_q3.drop('Signup Date',axis=1, inplace=True)
        X_q3=X_q3.rename(columns={'user_vintage':'User_Vintage'})
        
        
        # Most frequently viewed (page loads) product by the user in the last 15 days
        # Consider values only 15 days older and activity should be pageload
        # Find the number of times a product was viewed and most recent time it was viewed
        X_15=X[(X['vis_time']>day15) & (X['vis_time']<last_date)  & (X['Activity']=='pageload')].groupby(['UserID', 'ProductID'])['day','vis_time'].agg({'day':'count','vis_time':'max'}).reset_index()
        
        
        # Groupby by number of times a product was viewed and if products were viewed same number of time then select the most recent one
        xa=X_15.groupby(['UserID','day']).apply(lambda x: x[x['vis_time']==x['vis_time'].max()]['ProductID']).reset_index()
        
        xa.drop('level_2', axis=1, inplace=True)# drop the additional column
        
        # Select the product IDs which were viewed the most
        X_q4=xa.groupby('UserID').apply(lambda x: x[x['day']==x['day'].max()]['ProductID']).reset_index().drop('level_1',axis=1)
        
        X_q4=X_q4.rename(columns={'ProductID':'Most_Viewed_product_15_Days'})
        
        X_q4=X_q4.drop_duplicates(subset=['UserID'], keep='first').reset_index(drop=True) # drop the duplicates if any
        
        #Most Frequently used OS by user. 
        a=X.groupby(['UserID', 'OS'])['Activity'].count().reset_index() # Count the number of times each OS is used by the user
        
        # FInd the OS which is uded mostly by each user
        X_q5=a.groupby('UserID').apply(lambda x: x[x['Activity']==x['Activity'].max()]['OS']).reset_index().drop('level_1', axis=1)
        
        X_q5=X_q5.drop_duplicates(subset=['UserID'], keep='first').reset_index(drop=True) # drop duplicates if any
        
        X_q5=X_q5.rename(columns={'OS':'Most_Active_OS'})
        
        # Filter with activity= pageloads and then select product IDs which were viewed most recently
        X_q6=X[(X['Activity']=='pageload')].groupby(['UserID']).apply(lambda x: x[x['vis_time']==x['vis_time'].max()]['ProductID']).reset_index()
        
        X_q6.drop('vis_time',axis=1, inplace=True)
        X_q6=X_q6.rename(columns={'ProductID':'Recently_Viewed_Product'})
        
        X_q6=X_q6.drop_duplicates(subset=['UserID'], keep='first').reset_index(drop= True) # drop duplicates if any
        
        #Count of Page loads in the last 7 days by the user
        # Select last 7 days only and activity=pageload then find count for each userID
        X_q7=X[(X['vis_time']>wkday) & (X['vis_time']<last_date) & (X['Activity']=='pageload')].groupby(['UserID'])['Activity'].count().reset_index()
        
        X_q7=X_q7.rename(columns={'Activity':'Pageloads_last_7_days'})
        
        #Count of Clicks in the last 7 days  by the user
        # Select last 7 days only and activity=click then find count for each userID
        X_q8=X[(X['vis_time']>wkday) & (X['vis_time']<last_date) & (X['Activity']=='click')].groupby(['UserID'])['Activity'].count().reset_index()
        X_q8=X_q8.rename(columns={'Activity':'Clicks_last_7_days'})
        
        df=user_data['UserID'].sort_values().reset_index() # make a dataframe consisting of all userIDs

        df.drop('index', axis=1, inplace=True)
        
        
        dfs=[X_q1,X_q2, X_q3, X_q4, X_q5, X_q6, X_q7, X_q8] #list comprising of all dataframes
        for i in range(len(dfs)):
          df=pd.merge(df,dfs[i],how='left', on='UserID') # Merge all dataframes on userID
        
        
        # If a user has not viewed any product in the last 15 days then put it as Product101
        df['Most_Viewed_product_15_Days']=df['Most_Viewed_product_15_Days'].fillna('Product101')
        
        # If a user has not viewed any product then put it as Product101
        df['Recently_Viewed_Product']=df['Recently_Viewed_Product'].fillna('Product101')
        
        df=df.fillna(0)
        
        return df

        
a=Extract_features('VisitorLogsData.csv','userTable.csv')
df=a.solv_prob()   

df.to_csv('input_features.csv')


