# Opening Thoughts
Raj recently talked about opening a new location, but doesn't know where. It gave me a new project idea: use data to determine the 'best' location to open a restaurant. To do so, I'd need to collect data from scratch. And I need to webscrape Yelp and Google. But I still need more experience with webscraping, there are some Yelp and Google Maps policies that don't really allow me to scrape the data. So I'll generate a fake dataset with fake values to see how I can approach this model. 


```python
import pandas as pd
import random
import numpy as np
```


```python
# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate fake data
num_locations = 100
locations = [f"Location_{i}" for i in range(1, num_locations + 1)]
latitudes = [round(random.uniform(37.7, 37.8), 6) for _ in range(num_locations)]
longitudes = [round(random.uniform(-122.5, -122.4), 6) for _ in range(num_locations)]
average_income = [round(random.uniform(30000, 100000), 2) for _ in range(num_locations)]
population_density = [round(random.uniform(500, 20000), 2) for _ in range(num_locations)]
competition_score = [round(random.uniform(1, 10), 2) for _ in range(num_locations)]

# Generate the num_competitors column
num_competitors = [random.randint(1, 10) for _ in range(num_locations)]

# Create a DataFrame
data = {
    'Location': locations,
    'Latitude': latitudes,
    'Longitude': longitudes,
    'Average_Income': average_income,
    'Population_Density': population_density,
    'Competition_Score': competition_score,
    'Num_Competitors': num_competitors  # Adding the new column
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('restaurant_locations_with_competitors.csv', index=False)

```


```python
#df['Competition_Score'] = df['Population_Density'] * (1 + df['Num_Competitors'] / 10)
#df.to_csv('restaurant_competition.csv', index=False)
```


```python
locations = pd.read_csv('restaurant_locations_with_competitors.csv')
#competition = pd.read_csv('restaurant_competition.csv')
```


```python
locations.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Average_Income</th>
      <th>Population_Density</th>
      <th>Competition_Score</th>
      <th>Num_Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Location_1</td>
      <td>37.763943</td>
      <td>-122.498852</td>
      <td>97829.74</td>
      <td>18126.61</td>
      <td>4.98</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Location_2</td>
      <td>37.702501</td>
      <td>-122.427928</td>
      <td>94845.69</td>
      <td>11139.01</td>
      <td>2.92</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Location_3</td>
      <td>37.727503</td>
      <td>-122.431829</td>
      <td>89408.70</td>
      <td>16774.60</td>
      <td>5.26</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Location_4</td>
      <td>37.722321</td>
      <td>-122.446303</td>
      <td>41641.78</td>
      <td>11858.94</td>
      <td>9.11</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Location_5</td>
      <td>37.773647</td>
      <td>-122.473317</td>
      <td>63994.88</td>
      <td>3387.83</td>
      <td>8.16</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
location_headers = df.columns
```


```python
locations.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Average_Income</th>
      <th>Population_Density</th>
      <th>Competition_Score</th>
      <th>Num_Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>Location_96</td>
      <td>37.738162</td>
      <td>-122.413865</td>
      <td>68617.63</td>
      <td>6094.82</td>
      <td>7.11</td>
      <td>10</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Location_97</td>
      <td>37.799612</td>
      <td>-122.444967</td>
      <td>59938.08</td>
      <td>8882.82</td>
      <td>4.64</td>
      <td>5</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Location_98</td>
      <td>37.752911</td>
      <td>-122.494941</td>
      <td>30676.88</td>
      <td>11809.70</td>
      <td>2.49</td>
      <td>6</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Location_99</td>
      <td>37.797108</td>
      <td>-122.400072</td>
      <td>35267.07</td>
      <td>13266.76</td>
      <td>5.21</td>
      <td>4</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Location_100</td>
      <td>37.786078</td>
      <td>-122.416397</td>
      <td>91817.45</td>
      <td>9567.27</td>
      <td>2.15</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
sorted_locations_income = locations.sort_values(by = 'Average_Income', ascending = False)
```


```python
sorted_locations_income.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Average_Income</th>
      <th>Population_Density</th>
      <th>Competition_Score</th>
      <th>Num_Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54</th>
      <td>Location_55</td>
      <td>37.764804</td>
      <td>-122.487900</td>
      <td>99891.81</td>
      <td>19422.81</td>
      <td>3.23</td>
      <td>2</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Location_56</td>
      <td>37.760913</td>
      <td>-122.477530</td>
      <td>99726.75</td>
      <td>3982.07</td>
      <td>6.74</td>
      <td>10</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Location_16</td>
      <td>37.754494</td>
      <td>-122.412948</td>
      <td>99679.59</td>
      <td>17740.63</td>
      <td>1.22</td>
      <td>6</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Location_68</td>
      <td>37.703210</td>
      <td>-122.433102</td>
      <td>99106.31</td>
      <td>14321.31</td>
      <td>7.25</td>
      <td>10</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Location_10</td>
      <td>37.702980</td>
      <td>-122.404618</td>
      <td>98971.62</td>
      <td>17283.70</td>
      <td>4.02</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
least_competitive_location = sorted_locations_income.sort_values(by = 'Competition_Score', ascending = True)
least_competitive_location[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Average_Income</th>
      <th>Population_Density</th>
      <th>Competition_Score</th>
      <th>Num_Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>81</th>
      <td>Location_82</td>
      <td>37.726274</td>
      <td>-122.432664</td>
      <td>44251.81</td>
      <td>6356.28</td>
      <td>1.05</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Location_28</td>
      <td>37.709672</td>
      <td>-122.407090</td>
      <td>89690.39</td>
      <td>18650.91</td>
      <td>1.05</td>
      <td>9</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Location_16</td>
      <td>37.754494</td>
      <td>-122.412948</td>
      <td>99679.59</td>
      <td>17740.63</td>
      <td>1.22</td>
      <td>6</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Location_94</td>
      <td>37.742216</td>
      <td>-122.407673</td>
      <td>48336.87</td>
      <td>11184.78</td>
      <td>1.26</td>
      <td>8</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Location_36</td>
      <td>37.755204</td>
      <td>-122.451401</td>
      <td>38392.06</td>
      <td>17242.57</td>
      <td>1.45</td>
      <td>4</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Location_70</td>
      <td>37.726774</td>
      <td>-122.486769</td>
      <td>30547.62</td>
      <td>18548.72</td>
      <td>1.58</td>
      <td>10</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Location_20</td>
      <td>37.700650</td>
      <td>-122.484716</td>
      <td>50769.55</td>
      <td>18632.68</td>
      <td>1.66</td>
      <td>6</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Location_61</td>
      <td>37.798952</td>
      <td>-122.492901</td>
      <td>91660.49</td>
      <td>14706.63</td>
      <td>1.70</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Location_7</td>
      <td>37.789218</td>
      <td>-122.488845</td>
      <td>58072.82</td>
      <td>6511.04</td>
      <td>1.76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Location_31</td>
      <td>37.780713</td>
      <td>-122.469249</td>
      <td>35607.80</td>
      <td>16309.61</td>
      <td>1.96</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
highest_income_low_competition = []
for index, row in least_competitive_location.iterrows():
    income = row[3]
    competition = row[5]
    if (income > 70000 and competition < 5):
        highest_income_low_competition.append(row)
```


```python
best_locations = pd.DataFrame(highest_income_low_competition)
```


```python
populous_locs = best_locations.sort_values(by = 'Population_Density', ascending = False)
```


```python
populous_locs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Average_Income</th>
      <th>Population_Density</th>
      <th>Competition_Score</th>
      <th>Num_Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54</th>
      <td>Location_55</td>
      <td>37.764804</td>
      <td>-122.487900</td>
      <td>99891.81</td>
      <td>19422.81</td>
      <td>3.23</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Location_28</td>
      <td>37.709672</td>
      <td>-122.407090</td>
      <td>89690.39</td>
      <td>18650.91</td>
      <td>1.05</td>
      <td>9</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Location_1</td>
      <td>37.763943</td>
      <td>-122.498852</td>
      <td>97829.74</td>
      <td>18126.61</td>
      <td>4.98</td>
      <td>10</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Location_16</td>
      <td>37.754494</td>
      <td>-122.412948</td>
      <td>99679.59</td>
      <td>17740.63</td>
      <td>1.22</td>
      <td>6</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Location_21</td>
      <td>37.780582</td>
      <td>-122.423749</td>
      <td>97809.66</td>
      <td>17359.81</td>
      <td>4.73</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Location_10</td>
      <td>37.702980</td>
      <td>-122.404618</td>
      <td>98971.62</td>
      <td>17283.70</td>
      <td>4.02</td>
      <td>3</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Location_33</td>
      <td>37.753623</td>
      <td>-122.412199</td>
      <td>71652.46</td>
      <td>15853.80</td>
      <td>2.58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Location_15</td>
      <td>37.764988</td>
      <td>-122.408737</td>
      <td>97012.23</td>
      <td>15712.27</td>
      <td>2.79</td>
      <td>7</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Location_73</td>
      <td>37.787637</td>
      <td>-122.452733</td>
      <td>76437.21</td>
      <td>14968.03</td>
      <td>4.74</td>
      <td>9</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Location_26</td>
      <td>37.733659</td>
      <td>-122.467584</td>
      <td>70892.43</td>
      <td>14863.01</td>
      <td>3.20</td>
      <td>8</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Location_61</td>
      <td>37.798952</td>
      <td>-122.492901</td>
      <td>91660.49</td>
      <td>14706.63</td>
      <td>1.70</td>
      <td>7</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Location_81</td>
      <td>37.756137</td>
      <td>-122.427092</td>
      <td>80232.85</td>
      <td>14535.21</td>
      <td>4.42</td>
      <td>8</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Location_74</td>
      <td>37.731468</td>
      <td>-122.421538</td>
      <td>95725.10</td>
      <td>13577.26</td>
      <td>2.86</td>
      <td>7</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Location_92</td>
      <td>37.762745</td>
      <td>-122.472145</td>
      <td>83978.35</td>
      <td>12502.54</td>
      <td>2.97</td>
      <td>9</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Location_66</td>
      <td>37.777600</td>
      <td>-122.492914</td>
      <td>79247.79</td>
      <td>11743.47</td>
      <td>2.25</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Location_2</td>
      <td>37.702501</td>
      <td>-122.427928</td>
      <td>94845.69</td>
      <td>11139.01</td>
      <td>2.92</td>
      <td>9</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Location_71</td>
      <td>37.721098</td>
      <td>-122.406449</td>
      <td>87197.29</td>
      <td>10999.81</td>
      <td>4.67</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Location_100</td>
      <td>37.786078</td>
      <td>-122.416397</td>
      <td>91817.45</td>
      <td>9567.27</td>
      <td>2.15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Location_60</td>
      <td>37.737946</td>
      <td>-122.477978</td>
      <td>95328.16</td>
      <td>8973.99</td>
      <td>3.47</td>
      <td>7</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Location_62</td>
      <td>37.764000</td>
      <td>-122.436890</td>
      <td>91548.92</td>
      <td>6616.71</td>
      <td>3.57</td>
      <td>4</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Location_83</td>
      <td>37.758459</td>
      <td>-122.401583</td>
      <td>74396.66</td>
      <td>6531.05</td>
      <td>4.17</td>
      <td>7</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Location_93</td>
      <td>37.779208</td>
      <td>-122.475019</td>
      <td>74597.94</td>
      <td>6368.52</td>
      <td>4.92</td>
      <td>8</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Location_67</td>
      <td>37.722905</td>
      <td>-122.476200</td>
      <td>72817.44</td>
      <td>5467.09</td>
      <td>3.08</td>
      <td>5</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Location_37</td>
      <td>37.782940</td>
      <td>-122.493079</td>
      <td>92320.11</td>
      <td>4837.46</td>
      <td>3.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Location_87</td>
      <td>37.799754</td>
      <td>-122.413833</td>
      <td>89227.26</td>
      <td>2982.11</td>
      <td>4.77</td>
      <td>7</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Location_95</td>
      <td>37.706353</td>
      <td>-122.455687</td>
      <td>81886.16</td>
      <td>507.92</td>
      <td>4.03</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_locations = [] #based on highest income, lowest compeition and most density
for index, row in populous_locs.iterrows():
    density = row[4]
    if density > 10000:
        top_locations.append(row)
```


```python
top_locations = pd.DataFrame(top_locations)
```


```python
top_locations
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Average_Income</th>
      <th>Population_Density</th>
      <th>Competition_Score</th>
      <th>Num_Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54</th>
      <td>Location_55</td>
      <td>37.764804</td>
      <td>-122.487900</td>
      <td>99891.81</td>
      <td>19422.81</td>
      <td>3.23</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Location_28</td>
      <td>37.709672</td>
      <td>-122.407090</td>
      <td>89690.39</td>
      <td>18650.91</td>
      <td>1.05</td>
      <td>9</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Location_1</td>
      <td>37.763943</td>
      <td>-122.498852</td>
      <td>97829.74</td>
      <td>18126.61</td>
      <td>4.98</td>
      <td>10</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Location_16</td>
      <td>37.754494</td>
      <td>-122.412948</td>
      <td>99679.59</td>
      <td>17740.63</td>
      <td>1.22</td>
      <td>6</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Location_21</td>
      <td>37.780582</td>
      <td>-122.423749</td>
      <td>97809.66</td>
      <td>17359.81</td>
      <td>4.73</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Location_10</td>
      <td>37.702980</td>
      <td>-122.404618</td>
      <td>98971.62</td>
      <td>17283.70</td>
      <td>4.02</td>
      <td>3</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Location_33</td>
      <td>37.753623</td>
      <td>-122.412199</td>
      <td>71652.46</td>
      <td>15853.80</td>
      <td>2.58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Location_15</td>
      <td>37.764988</td>
      <td>-122.408737</td>
      <td>97012.23</td>
      <td>15712.27</td>
      <td>2.79</td>
      <td>7</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Location_73</td>
      <td>37.787637</td>
      <td>-122.452733</td>
      <td>76437.21</td>
      <td>14968.03</td>
      <td>4.74</td>
      <td>9</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Location_26</td>
      <td>37.733659</td>
      <td>-122.467584</td>
      <td>70892.43</td>
      <td>14863.01</td>
      <td>3.20</td>
      <td>8</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Location_61</td>
      <td>37.798952</td>
      <td>-122.492901</td>
      <td>91660.49</td>
      <td>14706.63</td>
      <td>1.70</td>
      <td>7</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Location_81</td>
      <td>37.756137</td>
      <td>-122.427092</td>
      <td>80232.85</td>
      <td>14535.21</td>
      <td>4.42</td>
      <td>8</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Location_74</td>
      <td>37.731468</td>
      <td>-122.421538</td>
      <td>95725.10</td>
      <td>13577.26</td>
      <td>2.86</td>
      <td>7</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Location_92</td>
      <td>37.762745</td>
      <td>-122.472145</td>
      <td>83978.35</td>
      <td>12502.54</td>
      <td>2.97</td>
      <td>9</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Location_66</td>
      <td>37.777600</td>
      <td>-122.492914</td>
      <td>79247.79</td>
      <td>11743.47</td>
      <td>2.25</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Location_2</td>
      <td>37.702501</td>
      <td>-122.427928</td>
      <td>94845.69</td>
      <td>11139.01</td>
      <td>2.92</td>
      <td>9</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Location_71</td>
      <td>37.721098</td>
      <td>-122.406449</td>
      <td>87197.29</td>
      <td>10999.81</td>
      <td>4.67</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



For this fake dataset project I learned:
1. sort_values
2. iterrows()
3. creating dataframes from lists
4. to sort values after values
5. read_csv() function instead of the whole opened_file = open() blah blah blah

So once you have the cleaned data, all you have to do is sort this sort that. exploratory data this explore data that.

## Preliminary Data Exploration

Before diving into the analysis, I would perform some initial exploratory data analysis to get a sense of the dataset's structure and contents:



```python
# Display basic information about the dataset
print(df.info())

# Display summary statistics of the numerical columns
print(df.describe())

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Data columns (total 7 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   Location            100 non-null    object 
     1   Latitude            100 non-null    float64
     2   Longitude           100 non-null    float64
     3   Average_Income      100 non-null    float64
     4   Population_Density  100 non-null    float64
     5   Competition_Score   100 non-null    float64
     6   Num_Competitors     100 non-null    int64  
    dtypes: float64(5), int64(1), object(1)
    memory usage: 5.6+ KB
    None
             Latitude   Longitude  Average_Income  Population_Density  \
    count  100.000000  100.000000      100.000000            100.0000   
    mean    37.747957 -122.451112    65773.678300          10999.8506   
    std      0.029500    0.029314    21283.191839           5745.6493   
    min     37.700650 -122.499943    30248.200000            507.9200   
    25%     37.722673 -122.477018    47103.307500           6333.5300   
    50%     37.748210 -122.453015    67274.225000          11161.8950   
    75%     37.771071 -122.423892    82967.470000          16338.3975   
    max     37.799754 -122.400072    99891.810000          19632.4200   
    
           Competition_Score  Num_Competitors  
    count         100.000000       100.000000  
    mean            5.403400         5.500000  
    std             2.434936         2.989882  
    min             1.050000         1.000000  
    25%             3.222500         3.000000  
    50%             5.365000         6.000000  
    75%             7.387500         8.000000  
    max             9.890000        10.000000  
    

## Step 3: Identify Promising Locations

Next, I would identify potential locations that meet the criteria of high income, low competition, and high population density. Here's how I would approach each criterion:


```python
#High Income Locations:
high_income_locations = df[df['Average_Income'] > 70000]

```


```python
#Low Competition Locations:
low_competition_locations = df.sort_values(by='Competition_Score').head(10)

```


```python
#High Population Density Locations:
high_density_locations = df.sort_values(by='Population_Density', ascending=False).head(10)

```

## Step 4: Combine Criteria

To find the best potential locations that fulfill all three criteria (high income, low competition, high population density), I would combine the results from the previous steps:


```python
best_locations = high_income_locations.merge(low_competition_locations, on='Location').merge(high_density_locations, on='Location')

```

This is kinda what I wanted to do. merge the tables, but I didnt know how to do so. 

## Step 5: Visualization

Visualizing the potential locations on a map can provide a clearer understanding of their distribution. You can use libraries like matplotlib or seaborn for this purpose.


```python
import matplotlib.pyplot as plt

plt.scatter(best_locations['Longitude'], best_locations['Latitude'], c='red', label='Best Locations')
plt.scatter(df['Longitude'], df['Latitude'], c='blue', alpha=0.3, label='All Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Potential Restaurant Locations')
plt.legend()
plt.show()

```


    
![png](output_30_0.png)
    


This visualization will show the best potential locations in red and all locations in blue.

Please note that this is a simplified analysis. In a real-world scenario, I would likely use more sophisticated methods, consider additional factors, and potentially use machine learning techniques to make more informed decisions about the best locations to open a restaurant.

# Me
I agree that there are additional factors because in the fake dataset, there were no sales and no type of cuisine in the dataset and what type of culture the city has. 

# Websearch: 

In a real-world analysis for determining the best locations to open a restaurant, there are many additional factors that should be considered beyond what's provided in the fake dataset. Here are some of the factors to take into account:

__Cuisine and Culture__: Understanding the local culture, preferences, and culinary trends is crucial. Certain cuisines may be more popular or better suited to a particular neighborhood or city.

__Sales and Revenue Data__: Historical sales and revenue data for existing restaurants in different areas can provide insights into potential customer demand and market saturation.

__Foot Traffic__: Analyzing foot traffic patterns, both during different times of the day and different days of the week, can help you assess the visibility and potential customer flow in a location.

__Competitor Analysis__: In addition to just the number of competitors, analyzing the types of cuisines, pricing strategies, and customer reviews of existing restaurants can help you position your restaurant effectively.

__Local Regulations__: Different areas might have varying regulations related to zoning, permits, alcohol licenses, health codes, and more, which can impact the feasibility of opening a restaurant.

__Demographics__: A deeper understanding of the demographics of different neighborhoods can help tailor restaurant concept to the target audience.

__Economic Trends__: Analyzing economic trends, such as employment rates, disposable income, and overall economic health of an area, can impact the potential customer base.

__Accessibility and Parking__: Ease of access, parking availability, and proximity to public transportation can influence customer convenience.

__Social Media and Online Presence__: Analyzing social media activity and online presence of existing restaurants can provide insights into customer engagement and preferences.

__Cultural Events and Festivals__: Understanding local cultural events and festivals can help align restaurant's offerings with the community's interests.

# Foot Traffic
I learned some things regarding foot traffic, I can use the number of reviews on platforms like Yelp and Google Reviews as a rough proxy for estimating foot traffic. While it might not provide an exact measurement of foot traffic, the number of reviews can offer some insights into the popularity and activity level of a restaurant, which could be correlated with foot traffic.

Here's how you I might approach this:

Assumption of Correlation: I can assume that a higher number of reviews indicates a higher level of customer activity, which could be associated with higher foot traffic. While this correlation might not be perfect (as some people might not leave reviews), it can still provide a rough estimate.

Caveats: factors such as the type of cuisine, restaurant size, location, and the overall customer experience can influence the number of reviews. Therefore, this method provides a very high-level estimate and might not accurately reflect foot traffic.

Normalization: Since the number of reviews can vary widely between restaurants, it's a good idea to normalize this data. I can divide the number of reviews by the restaurant's average income or population density to account for these variations.

Comparative Analysis: Use the normalized review counts to perform a comparative analysis among restaurants. Higher normalized review counts might suggest relatively higher foot traffic or customer engagement.

While using review counts as a proxy for foot traffic has limitations, it can still provide me with some insights into the relative popularity and activity levels of different restaurants. Just be aware that it's an indirect approach and should be interpreted with caution.

In a real-world scenario, I would ideally collect more specific foot traffic data using one of the methods mentioned earlier to get a more accurate understanding of customer activity.
