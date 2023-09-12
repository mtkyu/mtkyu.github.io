# Profitable App Profiles for the App Store and Google Play Markets

What the project is about: Our aim in this project is to find mobile app profiles that are profitable for the App Store and Google Play markets. We're working as data analysts for a company that builds Android and iOS mobile apps, and our job is to enable our team of developers to make data-driven decisions with respect to the kind of apps they build.

At our company, we only build apps that are free to download and install, and our main source of revenue consists of in-app ads. This means that our revenue for any given app is mostly influenced by the number of users that use our app. Our goal for this project is to analyze data to help our developers understand what kinds of apps are likely to attract more users.

# Opening and Exploring the Data


our aim is to help our developers understand what type of apps are likely to attract more users on Google Play and the App Store. To do this, we'll need to collect and analyze data about mobile apps available on Google Play and the App Store.

As of September 2018, there were approximately 2 million iOS apps available on the App Store, and 2.1 million Android apps on Google Play.

Collecting data for over 4 million apps requires a significant amount of time and money, so we'll try to analyze a sample of the data instead. To avoid spending resources on collecting new data ourselves, we should first try to see if we can find any relevant existing data at no cost. Luckily, here are two data sets that seem suitable for our goals:



A dataset containing data about approximately 10,000 Android apps from Google Play; the data was collected in August 2018. You can download the data set directly from this link.
A dataset containing data about approximately 7,000 iOS apps from the App Store; the data was collected in July 2017. You can download the data set directly from this link.
We'll start by opening and exploring these two data sets. To make them easier to explore, we created a function named explore_data() that you can repeatedly use to print rows in a readable way.


```python
def explore_data(dataset, start, end, rows_and_columns=False):
    dataset_slice = dataset[start:end]    
    for row in dataset_slice:
        print(row)
        print('\n') # adds a new (empty) line between rows
        
    if rows_and_columns:
        print('Number of rows:', len(dataset))
        print('Number of columns:', len(dataset[0]))

```


```python
from csv import reader

```


```python
opened_apple_file = open('AppleStore.csv')
opened_google_file = open('googleplaystore.csv')
```


```python
read_apple_file = reader(opened_apple_file)
read_google_file = reader(opened_google_file)
```


```python
android = list(read_google_file)
android_header = android[0]
android = android[1:]

apple = list(read_apple_file)
apple_header = apple[0]
apple = apple[1:]
```


```python
import pandas as pd

df_android = pd.DataFrame(android, columns=android_header)

android_checker = pd.DataFrame(android, columns=android_header)

df_apple = pd.DataFrame(apple, columns = apple_header)

apple_checker = pd.DataFrame(apple, columns=apple_header)

```


```python
#Make a function that explores data: 
def explore_data(dataset, x, y, rows_and_columns = False):
    dataslice = dataset[x:y]
    for row in dataslice:
        print(row)
        #start a new line
        print('\n')
    
    if rows_and_columns:
        print('Number of rows:', len(dataset))
        print('Number of columns:', len(dataset[0]))
```


```python
explore_data(android, 1000, 1003, True)
```

    ['Imgur: Find funny GIFs, memes & watch viral videos', 'ENTERTAINMENT', '4.3', '160164', '12M', '10,000,000+', 'Free', '0', 'Teen', 'Entertainment', 'August 1, 2018', '4.2.0.8447', '5.0 and up']
    
    
    ['Meme Generator', 'ENTERTAINMENT', '4.6', '3771', '53M', '100,000+', 'Paid', '$2.99', 'Mature 17+', 'Entertainment', 'August 3, 2018', '4.426', '4.1 and up']
    
    
    ['SketchBook - draw and paint', 'ENTERTAINMENT', '4.3', '256664', '77M', '10,000,000+', 'Free', '0', 'Everyone', 'Entertainment', 'May 4, 2018', '4.1.7', '4.0.3 and up']
    
    
    Number of rows: 10841
    Number of columns: 13
    


```python
explore_data(apple,0,3, True)
```

    ['284882215', 'Facebook', '389879808', 'USD', '0.0', '2974676', '212', '3.5', '3.5', '95.0', '4+', 'Social Networking', '37', '1', '29', '1']
    
    
    ['389801252', 'Instagram', '113954816', 'USD', '0.0', '2161558', '1289', '4.5', '4.0', '10.23', '12+', 'Photo & Video', '37', '0', '29', '1']
    
    
    ['529479190', 'Clash of Clans', '116476928', 'USD', '0.0', '2130805', '579', '4.5', '4.5', '9.24.12', '9+', 'Games', '38', '5', '18', '1']
    
    
    Number of rows: 7197
    Number of columns: 16
    

- GOOGLE: There are __10841 rows__ and __13 columns__  
- APPLE: There are __7197 rows__ and __16 columns__ 


```python
print('The column names for Apple are:', df_apple.columns)
print('The column names for Google are:', df_android.columns)
```

    The column names for Apple are: Index(['id', 'track_name', 'size_bytes', 'currency', 'price',
           'rating_count_tot', 'rating_count_ver', 'user_rating',
           'user_rating_ver', 'ver', 'cont_rating', 'prime_genre',
           'sup_devices.num', 'ipadSc_urls.num', 'lang.num', 'vpp_lic'],
          dtype='object')
    The column names for Google are: Index(['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type',
           'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver',
           'Android Ver'],
          dtype='object')
    

# Data Cleaning

We're going to remove any data that is not in English. We only wish to analyze apps that are free so for price = 0.0

# Deleting Wrong Data and Non-English Data

As seen from discussion page. There's a post that says Wrong Entry for 10472 row
Wrong Entry for 10472 row if Header is not included.

Header: - ['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver', 'Android Ver']
Row: - ['Life Made WI-Fi Touchscreen Photo Frame', '1.9', '19', '3.0M', '1,000+', 'Free', '0', 'Everyone', '', 'February 11, 2018', '1.0.19', '4.0 and up']
Category is not present and hence column shift has happened in this row.


```python
for row in android:
    name = row[0]
    if name == 'Life Made WI-Fi Touchscreen Photo Frame':
        del row
#Check to see if it has been deleted
for row in android:
    name = row[0]
    if name == 'Life Made WI-Fi Touchscreen Photo Frame':
        print('Row has not been deleted')
```

So this method does not delete the row. ChatGPT says to use the drop method where we assign the android list again: 


```python
name_to_delete = 'Life Made WI-Fi Touchscreen Photo Frame'
android = android[android[0] != name_to_delete]

# Print the number of rows in the modified DataFrame
print("Number of rows after deletion:", len(android))
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-385-0440a39382ed> in <module>
          1 name_to_delete = 'Life Made WI-Fi Touchscreen Photo Frame'
    ----> 2 android = android[android[0] != name_to_delete]
          3 
          4 # Print the number of rows in the modified DataFrame
          5 print("Number of rows after deletion:", len(android))
    

    IndexError: string index out of range



```python
#Check again
check_app = []
for row in android:
    name = row[0]
    if name == 'Life Made WI-Fi Touchscreen Photo Frame':
        check_app.append(row)
print(check_app)

```

    []
    

# Removing Duplicate Entries

The criteria I have for removing duplicate entries is if the entry has the same app name, I'll remove them since everything else will be the same. 

How do I detect duplicate entries? We are going to:

- Create two lists: one for storing the name of duplicate apps, and one for storing the name of unique apps.
- Loop through the android data set (the Google Play data set), and for each iteration, we did the following:
- Save the app name to a variable named name. If name was already in the unique_apps list, we appended name to the duplicate_apps list. 
    Else (if name wasn't already in the unique_apps list), we appended name to the unique_apps list.

(As a side note, you may notice we used the in operator above to check for membership in a list. We only learned to use in to check for membership in dictionaries, but in also works with lists):


```python
duplicate_apps_android = []
unique_apps_android = []

for app in android:
    name = app[0]
    if name in unique_apps:
        duplicate_apps_android.append(name)
    else:
        unique_apps_android.append(name)

print('For Android', '\n', 'Number of duplicate apps:', len(duplicate_apps))
print('\n')
print(' Examples of duplicate apps:', duplicate_apps[:15])

    


```

    For Android 
     Number of duplicate apps: 1181
    
    
     Examples of duplicate apps: ['Quick PDF Scanner + OCR FREE', 'Box', 'Google My Business', 'ZOOM Cloud Meetings', 'join.me - Simple Meetings', 'Box', 'Zenefits', 'Google Ads', 'Google My Business', 'Slack', 'FreshBooks Classic', 'Insightly CRM', 'QuickBooks Accounting: Invoicing & Expenses', 'HipChat - Chat Built for Teams', 'Xero Accounting Software']
    

In some rows with the same app names, there are different numbers of reviews. Meaning the app was updated many times and we want to take the most recent row. Therefore select the row with max() value of the review column. 


```python
duplicate_apple = []
unique_apple = []

for app in apple:
    name = app[1]
    if name in unique_apple:
        duplicate_apple.append(name)
    else:
        unique_apple.append(name)

print('For Apple', '\n', 'Number of duplicate apps:', len(duplicate_apple))
print('\n')
print('Examples of duplicate apps:', duplicate_apple[:15])

```

    For Apple 
     Number of duplicate apps: 2
    
    
    Examples of duplicate apps: ['Mannequin Challenge', 'VR Roller Coaster']
    

# Removing Duplicates


Now that we see the duplicates, it's time to remove them


```python
print('Expected Length:', len(df_android)-1181)
```

    Expected Length: 9660
    


```python
df_android.columns
```




    Index(['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type',
           'Price', 'Content Rating', 'Genres', 'Last Updated', 'Current Ver',
           'Android Ver'],
          dtype='object')




```python
#Use not in operator. And create a dictionary named:
reviews_max = {}

#loop through google play dataset not including header row:
for row in android:
    name = row[0]
    n_reviews = float(row[3])
    if (name not in reviews_max) or (n_reviews > reviews_max[name]):
        reviews_max[name] = n_reviews
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-375-dae8f133866a> in <module>
          5 for row in android:
          6     name = row[0]
    ----> 7     n_reviews = float(row[3])
          8     if (name not in reviews_max) or (n_reviews > reviews_max[name]):
          9         reviews_max[name] = n_reviews
    

    IndexError: string index out of range



```python
def convert_reviews_to_float(reviews):
    if 'M' in reviews:
        return float(reviews[:-1]) * 1_000_000
    else:
        return float(reviews)
```


```python
#Use not in operator. And create a dictionary named:
reviews_max = {}

#loop through google play dataset not including header row:
for row in android:
    name = row[0]
    n_reviews = row[3]
    n_reviews_float = convert_reviews_to_float(n_reviews)

    if (name not in reviews_max) or (n_reviews_float > reviews_max[name]):
        reviews_max[name] = n_reviews_float
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-339-6c5a2aacc001> in <module>
          5 for row in android:
          6     name = row[0]
    ----> 7     n_reviews = row[3]
          8     n_reviews_float = convert_reviews_to_float(n_reviews)
          9 
    

    IndexError: string index out of range


we compare n_reviews_float (the float value of the current app's reviews) with reviews_max[name], which fetches the maximum number of reviews for the current app from the reviews_max dictionary.

With this change, the code should now filter and append the rows to android_clean as intended, removing duplicate rows based on the maximum number of reviews for each app:


```python
android_clean = []
already_added = []

for row in df_android:
    name = row[0]
    n_reviews = row[3]
    n_reviews_float = convert_reviews_to_float(n_reviews)

    if (n_reviews_float == reviews_max[name] and name not in already_added):
        android_clean.append(row)
        already_added.append(name)

```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-377-e03700500614> in <module>
          4 for row in df_android:
          5     name = row[0]
    ----> 6     n_reviews = row[3]
          7     n_reviews_float = convert_reviews_to_float(n_reviews)
          8 
    

    IndexError: string index out of range



```python

```


      File "<ipython-input-378-a7ee24357134>", line 1
        android_clean.
                      ^
    SyntaxError: invalid syntax
    


### now doing the same for Apple: 


```python
df_apple.columns
```




    Index([], dtype='object')




```python
apple_reviews = {}

#loop through google play dataset not including header row:
for row in apple:
    name = row[1]
    n_reviews = row[5]
    n_reviews_float = convert_reviews_to_float(n_reviews)

    if (name not in apple_reviews) or (n_reviews_float > apple_reviews[name]):
        apple_reviews[name] = n_reviews_float
```


```python
apple_clean = []
apple_added = []

for row in apple:
    name = row[1]
    n_reviews = row[5]
    n_reviews_float = convert_reviews_to_float(n_reviews)

    if (n_reviews_float == apple_reviews[name] and name not in apple_added):
        apple_clean.append(row)
        apple_added.append(name)
apple_clean[:5]
```




    [['284882215',
      'Facebook',
      '389879808',
      'USD',
      '0.0',
      '2974676',
      '212',
      '3.5',
      '3.5',
      '95.0',
      '4+',
      'Social Networking',
      '37',
      '1',
      '29',
      '1'],
     ['389801252',
      'Instagram',
      '113954816',
      'USD',
      '0.0',
      '2161558',
      '1289',
      '4.5',
      '4.0',
      '10.23',
      '12+',
      'Photo & Video',
      '37',
      '0',
      '29',
      '1'],
     ['529479190',
      'Clash of Clans',
      '116476928',
      'USD',
      '0.0',
      '2130805',
      '579',
      '4.5',
      '4.5',
      '9.24.12',
      '9+',
      'Games',
      '38',
      '5',
      '18',
      '1'],
     ['420009108',
      'Temple Run',
      '65921024',
      'USD',
      '0.0',
      '1724546',
      '3842',
      '4.5',
      '4.0',
      '1.6.2',
      '9+',
      'Games',
      '40',
      '5',
      '1',
      '1'],
     ['284035177',
      'Pandora - Music & Radio',
      '130242560',
      'USD',
      '0.0',
      '1126879',
      '3594',
      '4.0',
      '4.5',
      '8.4.1',
      '12+',
      'Music',
      '37',
      '4',
      '1',
      '1']]



# Isolating Free Apps 

we only build apps that are free to download and install, and our main source of revenue consists of in-app ads. Our datasets contain both free and non-free apps; we'll need to isolate only the free apps for our analysis.

Isolating the free apps will be our last step in the data cleaning process. On the next screen, we're going to start analyzing the data.

1. Loop through each dataset to isolate the free apps in separate lists.

2.  identify the columns describing the app price correctly. And prices come up as strings ('0', '$ 0.99', $2.99, etc.), so make sure you're not checking an integer or a float in your conditional statements.
3. After isolating the free apps, check the length of each dataset to see how many apps you have remaining.


```python
df_android.columns
```




    Index([], dtype='object')




```python
free_apps = []

for row in android:
    name = row[0]
    price = row[7].replace('$', '')
    price_float = float(price)
    if price == 0:
        free_apps.append(row)
        
print(free_apps)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-259-2ca0f31e6225> in <module>
          4     name = row[0]
          5     price = row[7].replace('$', '')
    ----> 6     price_float = float(price)
          7     if price == 0:
          8         free_apps.append(row)
    

    ValueError: could not convert string to float: ''


During this code, I encountered some Value Errors, It looks like there were some input  that says 'Everyone'. I want to draw it out to see why that is: 


```python
everyone_price = []
for row in android:
    name = row[0]
    price = row[7]
    if price == 'Everyone':
        everyone_price.append(row)
```


```python
print(everyone_price)
```

    []
    

(This was all from before) I have since deleted the row with wrong entry: Looks like price is indeed 0. however for  this app, looks like they switched price and rating. I'm going to switch the values to its correct index:



```python
free_apps = []

for row in android:
    name = row[0]
    price = row[7].replace('$', '')

    # Skip rows where the price is not a numeric value
    if not price.isdigit():
        continue

    price_float = float(price)

    if price_float == 0:
        free_apps.append(name)

print("the number of free apps for Android are:", len(free_apps))

```

    the number of free apps for Android are: 10040
    


```python
non_numeric_prices = []

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

for row in df_android:
    name = row[0]
    price = row[7].replace('$', '')

    # Check if the price is not a numeric value
    if not isfloat(price):
        non_numeric_prices.append(row)

print("Number of rows where the price is not a numeric value:", len(non_numeric_prices))
```

    Number of rows where the price is not a numeric value: 0
    

## Check if cleaned data is equal to expected length


```python
print('Expected length:', len(android) - 1181)
```

    Expected length: 9660
    


```python
len(android_clean)
```




    0



Cool! Looks like my cleaned data is equal to the expected length


```python

```
