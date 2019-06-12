#!/usr/bin/env python
# coding: utf-8

# # Cleaning Car Sales Data

# ## Introduction
# 
# In this project, we're cleaning a Kaggle dataset containing listings of used cars from eBay in Germany, scraped in March 2016. It has 50,000 entries. The dataset can be found [here](https://www.kaggle.com/orgesleka/used-cars-database/data). 
# 
# The data dictionary is the following: 
# * `dateCrawled` - When this ad was first crawled. 
# 
# * `name` - Name of the car.
# 
# * `seller` - Is the seller  private or a dealer?
# 
# * `offerType` - Type of listing
# 
# * `price` - Asking price.
# 
# * `abtest` - Whether the listing is included in an A/B test.
# 
# * `vehicleType` - The vehicle Type.
# 
# * `yearOfRegistration` - The year in which the car was first registered.
# 
# * `gearbox` - The transmission type.
# 
# * `powerPS` - The power of the car in horsepower (Pferdestärke).
# 
# * `model` - The car model name.
# 
# * `kilometer` - Distance driven.
# 
# * `monthOfRegistration` - The month in which the car was first registered.
# 
# * `fuelType` - What type of fuel the car uses.
# 
# * `brand` - The brand of the car.
# 
# * `notRepairedDamage` - Has the car been damaged and is it unrepaired?
# 
# * `dateCreated` - Date of listing creation.
# 
# * `nrOfPictures` - The number of pictures in the ad.
# 
# * `postalCode` - The postal code for the location of the vehicle.
# 
# * `lastSeenOnline` - When the crawler saw this ad last online.
#     
#     
# What we'll be trying to do here is to tidy up the data, and then analyse the listings using pandas.
# 

# ## Preliminary data exploration

# In[2]:


import pandas as pd
import numpy as np

autos = pd.read_csv("autos.csv", encoding = "Latin-1")


# In[3]:


autos.head()


# Some of the columns have null objects. 
# 
# fuelType has about 4500 null objects. 
# notRepairedDamage has about 10000. 
# gearbox and model have about 3000 each.
# And vehicleType has about 5000. 
# 
# Unsurprisingly, the info about the cars is in German. Most of the columns are strings. 

# In[49]:


autos.columns


# In[3]:


autos.columns =['date_crawled', 'name', 'seller', 'offer_type', 'price', 'abtest',
      'vehicle_type', 'registration_year', 'gearbox', 'power_PS', 'model',
      'odometer', 'registration_month', 'fuel_type', 'brand',
      'unrepaired_damage', 'ad_created', 'nr_of_pictures', 'postal_code',
      'last_seen']
cols = ['date_crawled', 'name', 'brand', 'model', 'vehicle_type',  'abtest',
       'registration_year', 'registration_month', 'gearbox', 'power_PS', 
      'odometer',  'fuel_type', 
      'unrepaired_damage','seller', 'offer_type', 'price', 'ad_created', 'nr_of_pictures', 'postal_code',
      'last_seen']
autos = autos[cols]
autos.head()


# In[ ]:





# Here, we changed the column names from camelcase to snakecase, and then rearranged the columns so that similar groups would be together - date/year of registration, the seller's info, the brand/model/type of car, etc. 

# ## Tidying data

# In[51]:


autos.describe()


# From these descriptive stats, there are a couple curious things.
# 
# * None of the 50,0000 cars have any pictures. Therefore, the pictures column could be dropped.
# * Postal codes are numeric in Germany, they would need to be turned into strings rather than left as integers.
# * A 2015 Ferrari has a horsepower of 660. it's probably an error that the maximum horsepower of 17700 is an error. 
# * The max/min year (9999 and 1000) are unlikely. 

# In[4]:


# autos["price"] = autos["price"].str.replace("$", "").str.replace(",","").astype(int)

# autos["odometer"] = autos["odometer"].str.replace(",","").str.replace("km", "").astype(int)
# autos["postal_code"] = autos["postal_code"].astype(str)
# autos.rename(index=str, columns={"odometer":"odometer_km"})


# Above, we converted the distance on the `odomter`, as well as the price, from strings into numbers, and changed `postal_code` into a string. 

# In[47]:


autos["price"].value_counts().sort_index(ascending=False)


# There are only 13 different values for the `odometer`, and most of the cars (more than half!) have about 150,000 km on the clock. This makes some sense for used cars. 
# 
# On the other hand, there are quite a few cars for urealistic `price`s. Here, we decided that the cutoff point for realism would be $300,000
# 

# In[48]:


autos = autos[autos["price"].between(1,300000)]
autos["price"].describe()


# 

# In[7]:


# autos[autos[['date_crawled','ad_created','last_seen']].str[:10].value_counts(normalize = True, dropna=False).sort_index()]
autos["date_crawled"].str[:10].value_counts(normalize = True, dropna = False).sort_index()




# The crawler took data every day over the month of March in 2016 (and some change). 

# In[49]:


autos["ad_created"].str[:10].value_counts(normalize = True, dropna = False).sort_index()


# Some of the cars put up for sale were there for months (and a minority were there for more than a year!), while others were there for a few days.

# In[50]:


autos["last_seen"].str[:10].value_counts(normalize = True, dropna = False).sort_index()


# In[51]:


autos["registration_year"].describe()


# Some of the car data is suspect as cars didn't exist in 1000 and we don't know if they will in 9999.

# In[52]:


autos["registration_year"].value_counts().sort_index(ascending = False)


# Since the first cars were made in the late 19th century, I'd delete any option before 1880. I'd also exclude options after 2016. For further accuracy, we'll also ignore cars made before 1980, since their number isn't significant enough.
# 

# In[7]:


autos = autos[autos["registration_year"].between(1980, 2016)]
autos["registration_year"].value_counts(normalize = True).sort_index(ascending = False)


# There's a nice little bell curve here, with most of the cars being made between the early 90s and 2016, specifically between 2000 and 2005. Not many people are trying to sell new cars. 
# 

# In[8]:


get_ipython().magic('matplotlib inline')
from scipy.stats import norm 
from matplotlib import pyplot as plt

autos.registration_year.plot(kind = 'hist')

range = np.arange(1980, 2016, 0.001)
plt.title("Distribution of car year registration, 1980-2016")
print(plt.plot(range, norm.pdf(range,0,1)))


# ## Analysis of car brands

# In[55]:


autos["brand"].unique().shape


# There are 40 different brands of car for sale. We're interested in the brands of cars who make up 1% of cars sold or greater. 

# In[56]:


autos["brand"].value_counts(normalize = True)


# 

# In[57]:


brands = autos["brand"].value_counts(normalize=True)
chosen_brands = brands[brands> 0.01].index
print(chosen_brands)

mean_prices_brands  = {}
mean_mileage_brands = {}

for brand in chosen_brands:
    the_brand = autos[autos["brand"] == brand]
    the_mean = the_brand["price"].mean()
    the_mileage = the_brand["odometer"].mean()
    mean_prices_brands[brand] = the_mean.astype(int)
    mean_mileage_brands[brand] = the_mileage.astype(int)

mean_prices_brands


# Renaults were on average the least expeinsive at 2425, while Audis were the most, at 9333.  
# Audis, Mercs, BMWs all led the pack in terms of price, in rthe rage of 8000+.
# VWs, Hyundais and Toyota were in between in the low 5000s. 
# Opel, Renault and Fiat were all sub 3000. 

# In[58]:


mpb_series = pd.Series(mean_prices_brands)
mmb_series = pd.Series(mean_mileage_brands)

mean_brands = pd.DataFrame(mpb_series, columns = ["mean_price"] )

mean_brands["mean_mileage_km"] = mmb_series
print(mean_brands)


# 

# In[59]:


auto_names = autos[["brand", "model"]].dropna()
autos["full_name"]= auto_names["brand"] + "-" +  auto_names["model"]
autos["full_name"].value_counts(ascending = False)


# There are 290 different types of car in total here. Far and away he most popular is the VW Golf. 
# 3,705 Golfs were available to purchase during the month of crawling. 

# In[60]:


full_name = autos["full_name"].value_counts(normalize = True)
commonest_cars  = full_name[full_name > 0.01].index
print(commonest_cars)

avg_price_common = {}

for car in commonest_cars:
    the_name = autos[autos["full_name"] == car]
    the_mean = the_name["price"].mean()
    avg_price_common[car] = the_mean.astype(int)

    
avg_price_common


# Here, we can see that the most common cars varied in average price.

# 
# Here's the key analytical nugget: for cars with the same brand and make, what's the percentage difference for damanged and undamaged cars?

# In[62]:


avg_price_broken = {}
avg_price_not_broken = {}
broken_cars = autos[autos["unrepaired_damage"] == "ja"].dropna()
broken_car_names = broken_cars["full_name"]


car_broken =  (autos["unrepaired_damage"] == "ja")
filtered_cars = autos[car_broken]
#filtered_cars
for car in broken_car_names:
    the_name = filtered_cars[filtered_cars["full_name"] == car]
    the_mean = the_name["price"].mean()
    avg_price_broken[car] = the_mean.astype(int)
    
#avg_price_broken

for car in broken_car_names:
    the_name = autos[autos["full_name"] == car]
    the_mean = the_name["price"].mean()
    avg_price_not_broken[car] = the_mean.astype(int)
    
    
apb_series = pd.Series(avg_price_broken)
apn_series = pd.Series(avg_price_not_broken)


price_brokenness = pd.DataFrame(apb_series, columns = ["average_price_broken"] )

price_brokenness["average_price_not_broken"] = apn_series
price_brokenness.dtypes

price_brokenness["delta"] = ((price_brokenness["average_price_broken"] / price_brokenness["average_price_not_broken"]) * 100) - 100


print(price_brokenness)

outliers = (price_brokenness["delta"] > 0)
outliers_true  = price_brokenness[outliers]
outliers_true


# The overwhelming majority of cars are, as expected, cheaper when they're damaged, but a few were more expensive. I had an idea why, and wanted to figure it out.
# 
# What I did was, nevertheless, pretty cool: I found the average price of both damaged `average_price_broken` and undamaged `average_price_not_broken` cars, divided the former by the latter, multiplied by 100 to convert it to a percent value, and then subtracted 100 to give a `delta`. 

# The cars that were, on average, more expensive damaged than undamaged were the Alfa Romeo 159, the Kia Carnival, the Lancia Andere, the Lancia Lybra, the Mercedes Benz CL and Viano, the Skoda Superb and the VW Touareg. 
# 
# My hypothesis is that there were a significantly smaller number of damaged cars than undamaged cars, and among them, the range of prices were narrower, therefore providing what appeared to be a greater average price.

# In[79]:


# autos[autos["full_name"] == "alfa_romeo-159"]
# autos[autos["full_name"] == "kia-carnival"]
# autos[autos["full_name"] == "lancia-andere"]
# autos[autos["full_name"] == "lancia-lybra"]
# autos[autos["full_name"] == "mercedes_benz-cl"]
# autos[autos["full_name"] == "mercedes_benz-viano"]
# autos[autos["full_name"] == "skoda-superb"]
broken_touareg = (autos["full_name"] == "volkswagen-touareg") & (autos["unrepaired_damage"] == "ja")
autos[broken_touareg]


# In the case of Alfa Romeo 159, only two of the thirty-two cars had been damaged. 3 had "NaN" in the damage column, a cause for concern. The remainder had not been damaged. The price of the remaining cars varied between 2,500 and 12,899 euros, with an average of 6,626. 
# 
# Something is interesting among the Kia Carnivals: one of the cars was listed as "NaN" in the damage column, while its name read "KIA_Carnival_Automatik_Motor_defekt". Including this in the calculation for the average price of the damaged cars brought it down to 2,156 euros — still 400 euros more expensive than their unbroken compatriots. Of the six cars out of thirty that were damaged, one of them was priced at 4,500 euros, one of the highest prices among all the Kia Carnivals. This possibly explains why.
# 
# There were only 10 Lancia Anderes. The explanation for why they had the highest price delta (broken Anderes were  a whopping 95% more expensive than unbroken ones) was because there was only one broken Andere listed, and it was priced at 8,750
# 
# Equally, there were 10 Lancia Libras, and it's the same story as the Anderes, at a fraction of the cost. The single broken Libra was listed at 799 euros.
# 
# There were 38 Mercedes-Benz CLs, and only two of them were listed as damaged. While the unbroken CLs varied wildly from over 50,000 euros to below 3,000, the damaged ones both were close to one another in price. 
# 
# The story's not quite the same for The Merc Vianos: actually, it's the other way round. The undamaged cars have an expected range of prices. However, the damaged ones of which there are two, have something surprising. One's available for a sweet 44,000 euros, and the other, labelled "Nicht_fahrbereit" or "Not ready to drive", is available at 250 euros. 
# 
# A possible reason for the damaged cars costing more than the undamaged ones is that a specific car, Chromleiste_Skoda_Superb_3T_Heckklappe_Neu, is listed as costing 12 euros. This is very likely a typo. The deletion of this entry will probably result in the problem being removed. 
# 
# 
# Finally, there are 94 VW Touaregs, three of which were damaged, and one of those cost 44,200 euros, explaining the price difference.
# 
# In short, my hypothesis was largely correct. In all but two of the cases, the discrepency was due to there being fewer broken cars than unbroken for a given brand and make, and those cars having a greater average cost.  
# 
# 

# ## Conclusion & Next Steps
# 
# In this project, we explored and cleaned data about used cars on eBay from  March 2016. We converted strings to numbers and vice versa, deleted unneeded columns, and dealt with unexpected values. 
# 
# The centrepiece of the project was exploring and explaining the discrepency between prices of damaged and undamaged vehicles. 
# 
# The purpose of this project was chiefly to explore and clean the dataset, but there is quite a bit more that can be done. 
# 
# * Using basemap, divising a visualization of the geographic distribution of cars sold in Germany
# * Exploring the `name` field for keywords and potentially starting new relevant columns.
# * Exploring the `vehicle_type` column, as well as the `gearbox` and `fuel` columns for potential insight.  
# * Examining `horsepower` for potential mistakes. 
# 
# 
