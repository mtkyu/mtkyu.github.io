```python
import csv
```


```python
import csv

file_path = r"C:\Users\mtmic\datasets\hacker_news.csv"
with open(file_path, 'r', encoding='utf-8') as opened_file:
    read_file = csv.reader(opened_file)
    hn = list(read_file)

```


```python
headers = hn[:1]
print(headers)
```

    [['id', 'title', 'url', 'num_points', 'num_comments', 'author', 'created_at']]
    


```python
hn = hn[1:]
print(hn[:6])
print(len(hn))
```

    [['12579005', 'SQLAR  the SQLite Archiver', 'https://www.sqlite.org/sqlar/doc/trunk/README.md', '1', '0', 'blacksqr', '9/26/2016 3:24'], ['12578997', 'What if we just printed a flatscreen television on the side of our boxes?', 'https://medium.com/vanmoof/our-secrets-out-f21c1f03fdc8#.ietxmez43', '1', '0', 'pavel_lishin', '9/26/2016 3:19'], ['12578989', 'algorithmic music', 'http://cacm.acm.org/magazines/2011/7/109891-algorithmic-composition/fulltext', '1', '0', 'poindontcare', '9/26/2016 3:16'], ['12578979', 'How the Data Vault Enables the Next-Gen Data Warehouse and Data Lake', 'https://www.talend.com/blog/2016/05/12/talend-and-Â\x93the-data-vaultÂ\x94', '1', '0', 'markgainor1', '9/26/2016 3:14'], ['12578975', 'Saving the Hassle of Shopping', 'https://blog.menswr.com/2016/09/07/whats-new-with-your-style-feed/', '1', '1', 'bdoux', '9/26/2016 3:13'], ['12578954', "Macalifa  A new open-source music app for UWP that won't suck", 'http://forums.windowscentral.com/windows-phone-apps/440523-macalifa-new-open-source-music-app-uwp-wont-suck.html', '1', '0', 'thecodrr', '9/26/2016 3:06']]
    293118
    

## Extracting Ask HN and Show HN Posts

To find the posts that begin with either Ask HN or Show HN, we'll use the string method startswith. Given a string object, say, string1, we can check if starts with, say, dq by inspecting the output of the object string1.startswith('dq'). If string1 starts with dq, it will return True; otherwise, it will return False.


```python
ask_posts = []
show_posts = []
other_posts = []

for row in hn:
    title = row[1]
    if title.lower().startswith('ask hn'):
        ask_posts.append(row)
    elif title.lower().startswith('show hn'):
        show_posts.append(row)
    else:
        other_posts.append(row)

```


```python
print(len(ask_posts))
print(len(show_posts))
print(len(other_posts))
```

    9139
    10158
    273822
    


```python
print(ask_posts[:6])
```

    [['12578908', 'Ask HN: What TLD do you use for local development?', '', '4', '7', 'Sevrene', '9/26/2016 2:53'], ['12578522', 'Ask HN: How do you pass on your work when you die?', '', '6', '3', 'PascLeRasc', '9/26/2016 1:17'], ['12577908', 'Ask HN: How a DNS problem can be limited to a geographic region?', '', '1', '0', 'kuon', '9/25/2016 22:57'], ['12577870', 'Ask HN: Why join a fund when you can be an angel?', '', '1', '3', 'anthony_james', '9/25/2016 22:48'], ['12577647', 'Ask HN: Someone uses stock trading as passive income?', '', '5', '2', '00taffe', '9/25/2016 21:50'], ['12576946', 'Ask HN: How hard would it be to make a cheap, hackable phone?', '', '2', '1', 'hkt', '9/25/2016 19:30']]
    


```python
total_ask_comments = 0
for row in ask_posts:
    num_comments = int(row[4])
    total_ask_comments += num_comments
    
avg_ask_comments = total_ask_comments/len(ask_posts)
print(avg_ask_comments)
```

    10.393478498741656
    


```python
total_show_comments = 0
for row in show_posts:
    num_comments = int(row[4])
    total_show_comments += num_comments
avg_show_comments = total_show_comments/len(show_posts)
print(avg_show_comments)
```

    4.886099625910612
    

## Which type of posts receive more comments?
It seems like ask posts receive a majority of comments compared to show posts.

# Number of Comments by hour created

In this section, I am going to:
1. Calculate the number of ask posts created in each hour of the day, along with the number of comments received.
2. Calculate the average number of comments ask posts receive by hour created.


```python
import datetime as dt
```


```python
result_list = []
for row in ask_posts:
    created_at = row[6]
    num_comments = int(row[4])
    result_list.append([created_at, num_comments])
```


```python
print(result_list[:5])
```

    [['9/26/2016 2:53', 7], ['9/26/2016 1:17', 3], ['9/25/2016 22:57', 0], ['9/25/2016 22:48', 3], ['9/25/2016 21:50', 2]]
    


```python
counts_by_hour = {}
comments_by_hour = {}
```


```python
for element in result_list:
    date = element[0]
    comment = element[1]
    formatted = "%m/%d/%Y %H:%M"
    parsed = dt.datetime.strptime(date,formatted)
    by_hour = parsed.strftime("%H")
    if by_hour not in counts_by_hour:
        counts_by_hour[by_hour] = 1
        comments_by_hour[by_hour] = comment
    else:
        counts_by_hour[by_hour] += 1
        comments_by_hour[by_hour] += comment
    
```


```python
comments_by_hour
```




    {'02': 2996,
     '01': 2089,
     '22': 3372,
     '21': 4500,
     '19': 3954,
     '17': 5547,
     '15': 18525,
     '14': 4972,
     '13': 7245,
     '11': 2797,
     '10': 3013,
     '09': 1477,
     '07': 1585,
     '03': 2154,
     '23': 2297,
     '20': 4462,
     '16': 4466,
     '08': 2362,
     '00': 2277,
     '18': 4877,
     '12': 4234,
     '04': 2360,
     '06': 1587,
     '05': 1838}




```python
avg_by_hour = []
for element in comments_by_hour:
    avg_by_hour.append([element, comments_by_hour[element]/counts_by_hour[element]])
```


```python
avg_by_hour
```




    [['02', 11.137546468401487],
     ['01', 7.407801418439717],
     ['22', 8.804177545691905],
     ['21', 8.687258687258687],
     ['19', 7.163043478260869],
     ['17', 9.449744463373083],
     ['15', 28.676470588235293],
     ['14', 9.692007797270955],
     ['13', 16.31756756756757],
     ['11', 8.96474358974359],
     ['10', 10.684397163120567],
     ['09', 6.653153153153153],
     ['07', 7.013274336283186],
     ['03', 7.948339483394834],
     ['23', 6.696793002915452],
     ['20', 8.749019607843136],
     ['16', 7.713298791018998],
     ['08', 9.190661478599221],
     ['00', 7.5647840531561465],
     ['18', 7.94299674267101],
     ['12', 12.380116959064328],
     ['04', 9.7119341563786],
     ['06', 6.782051282051282],
     ['05', 8.794258373205741]]




```python
swap_avg_by_hour = []

for row in avg_by_hour:
    swap_avg_by_hour.append([row[1], row[0]])
    
print(swap_avg_by_hour)

sorted_swap = sorted(swap_avg_by_hour, reverse=True)

sorted_swap
```

    [[11.137546468401487, '02'], [7.407801418439717, '01'], [8.804177545691905, '22'], [8.687258687258687, '21'], [7.163043478260869, '19'], [9.449744463373083, '17'], [28.676470588235293, '15'], [9.692007797270955, '14'], [16.31756756756757, '13'], [8.96474358974359, '11'], [10.684397163120567, '10'], [6.653153153153153, '09'], [7.013274336283186, '07'], [7.948339483394834, '03'], [6.696793002915452, '23'], [8.749019607843136, '20'], [7.713298791018998, '16'], [9.190661478599221, '08'], [7.5647840531561465, '00'], [7.94299674267101, '18'], [12.380116959064328, '12'], [9.7119341563786, '04'], [6.782051282051282, '06'], [8.794258373205741, '05']]
    




    [[28.676470588235293, '15'],
     [16.31756756756757, '13'],
     [12.380116959064328, '12'],
     [11.137546468401487, '02'],
     [10.684397163120567, '10'],
     [9.7119341563786, '04'],
     [9.692007797270955, '14'],
     [9.449744463373083, '17'],
     [9.190661478599221, '08'],
     [8.96474358974359, '11'],
     [8.804177545691905, '22'],
     [8.794258373205741, '05'],
     [8.749019607843136, '20'],
     [8.687258687258687, '21'],
     [7.948339483394834, '03'],
     [7.94299674267101, '18'],
     [7.713298791018998, '16'],
     [7.5647840531561465, '00'],
     [7.407801418439717, '01'],
     [7.163043478260869, '19'],
     [7.013274336283186, '07'],
     [6.782051282051282, '06'],
     [6.696793002915452, '23'],
     [6.653153153153153, '09']]




```python
# Sort the values and print the the 5 hours with the highest average comments.

print("Top 5 Hours for 'Ask HN' Comments")
for avg, hr in sorted_swap[:5]:
    print(
        "{}: {:.2f} average comments per post".format(
            dt.datetime.strptime(hr, "%H").strftime("%H:%M"),avg
        )
    )
```

    Top 5 Hours for 'Ask HN' Comments
    15:00: 28.68 average comments per post
    13:00: 16.32 average comments per post
    12:00: 12.38 average comments per post
    02:00: 11.14 average comments per post
    10:00: 10.68 average comments per post
    

# Summary

I set a goal, collected and sorted the data. Later on, I cleaned and reformatted the data to prepare it for analysis.
I can take this project further and consider my next steps. 

I could:
- Determine if show or ask posts receive more points on average.
- Determine if posts created at a certain time are more likely to receive more points.
- Compare results to the average number of comments and points other posts receive.

# What's Next?


Now that I've finished thsi project, I need to consider other projects that are more interesting to me and are similar to skills used in this one.
I think my next project (self-guided) would be seasonal sales analysis to determine which time of the year makes the most sales in a certain business or certain type of business.



```python

```
