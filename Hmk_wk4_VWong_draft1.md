

```python
# Import Dependencies
import pandas as pd
from sklearn import preprocessing
import numpy as np
import random
```


```python
%precision %.3f
```




    '%.3f'




```python
# Define the file and folder path
heroes_json = "Resources/purchase_data.json"
```


```python
# Read the json file and display the dataframe
heroes_df = pd.read_json(heroes_json)
heroes_df.head()
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
      <th>Age</th>
      <th>Gender</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
      <th>SN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>Male</td>
      <td>165</td>
      <td>Bone Crushing Silver Skewer</td>
      <td>3.37</td>
      <td>Aelalis34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>Male</td>
      <td>119</td>
      <td>Stormbringer, Dark Blade of Ending Misery</td>
      <td>2.32</td>
      <td>Eolo46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>Male</td>
      <td>174</td>
      <td>Primitive Blade</td>
      <td>2.46</td>
      <td>Assastnya25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>Male</td>
      <td>92</td>
      <td>Final Critic</td>
      <td>1.36</td>
      <td>Pheusrical25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>Male</td>
      <td>63</td>
      <td>Stormfury Mace</td>
      <td>1.27</td>
      <td>Aela59</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Determine the total number of players through use of the counts for the columns
# This determines the # of rows in the json file
# All rows are full
heroes_df.count()

```




    Age          780
    Gender       780
    Item ID      780
    Item Name    780
    Price        780
    SN           780
    dtype: int64




```python
heroes_df.describe()
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
      <th>Age</th>
      <th>Item ID</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>780.000000</td>
      <td>780.000000</td>
      <td>780.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>22.729487</td>
      <td>91.293590</td>
      <td>2.931192</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.930604</td>
      <td>52.707537</td>
      <td>1.115780</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>1.030000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>19.000000</td>
      <td>44.000000</td>
      <td>1.960000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22.000000</td>
      <td>91.000000</td>
      <td>2.880000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>25.000000</td>
      <td>135.000000</td>
      <td>3.910000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>45.000000</td>
      <td>183.000000</td>
      <td>4.950000</td>
    </tr>
  </tbody>
</table>
</div>




```python
num_players_valuecounts = heroes_df["Item ID"].value_counts()
#num_players_valuecounts
```


```python
num_players = len(heroes_df["SN"].unique())
num_players
```




    573




```python
# Answer to problem #1
print ("Total Number of Players = ", num_players)
```

    Total Number of Players =  573
    


```python
grouped_players = heroes_df.groupby("SN")
#grouped_players.count()
```


```python
# Answer to Question #2.1
num_unique_items_id = len(heroes_df["Item ID"].unique())
#print (num_unique_items_id)
num_unique_item_name = len(heroes_df["Item Name"].unique())
print ("Number of Unique Items = ", num_unique_item_name)
```

    Number of Unique Items =  179
    


```python
# Average Purchase Price
ave_price = heroes_df["Price"].mean()
ave_price_rd = round(ave_price,3)
print ('Average Purchase Price = $', ave_price_rd)
```

    Average Purchase Price = $ 2.931
    


```python
grouped_item_id = heroes_df.groupby("Item ID")
#grouped_item_id.mean()
```


```python
grouped_item_name = heroes_df.groupby("Item Name")
#grouped_item_name.mean()
```


```python
#grouped_item_name["Item Name"].count()
```


```python
# Answer to question #2.4 Total Rev = Ave Price x # of transactions
#Total Number of Purchases are num of transactions x the average price
quantity = heroes_df["SN"].count()
tot_rev = ave_price_rd * (quantity-1)

print ("Total revenue = average price x # of transactions = $ ", tot_rev)
```

    Total revenue = average price x # of transactions = $  2283.2490000000003
    


```python
# Gender demographics
# The value_counts method counts unique values in a column
gendercount = heroes_df["Gender"].value_counts()
gendercount

```




    Male                     633
    Female                   136
    Other / Non-Disclosed     11
    Name: Gender, dtype: int64




```python
# Answer to Question 3
# % and count of male players
grouped_item_gender = heroes_df.groupby("Gender")
grouped_item_gender.count()

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
      <th>Age</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
      <th>SN</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>136</td>
      <td>136</td>
      <td>136</td>
      <td>136</td>
      <td>136</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>633</td>
      <td>633</td>
      <td>633</td>
      <td>633</td>
      <td>633</td>
    </tr>
    <tr>
      <th>Other / Non-Disclosed</th>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_item_gender.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">Age</th>
      <th colspan="5" halign="left">Item ID</th>
      <th colspan="8" halign="left">Price</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>...</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>136.0</td>
      <td>22.558824</td>
      <td>7.419459</td>
      <td>7.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>25.0</td>
      <td>40.0</td>
      <td>136.0</td>
      <td>88.110294</td>
      <td>...</td>
      <td>123.25</td>
      <td>182.0</td>
      <td>136.0</td>
      <td>2.815515</td>
      <td>1.151027</td>
      <td>1.03</td>
      <td>1.8275</td>
      <td>2.615</td>
      <td>3.750</td>
      <td>4.95</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>633.0</td>
      <td>22.685624</td>
      <td>6.804740</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>25.0</td>
      <td>45.0</td>
      <td>633.0</td>
      <td>91.571880</td>
      <td>...</td>
      <td>137.00</td>
      <td>183.0</td>
      <td>633.0</td>
      <td>2.950521</td>
      <td>1.109967</td>
      <td>1.03</td>
      <td>2.0400</td>
      <td>2.910</td>
      <td>3.910</td>
      <td>4.95</td>
    </tr>
    <tr>
      <th>Other / Non-Disclosed</th>
      <td>11.0</td>
      <td>27.363636</td>
      <td>6.932139</td>
      <td>12.0</td>
      <td>24.0</td>
      <td>27.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>11.0</td>
      <td>114.636364</td>
      <td>...</td>
      <td>156.00</td>
      <td>183.0</td>
      <td>11.0</td>
      <td>3.249091</td>
      <td>0.957230</td>
      <td>1.88</td>
      <td>2.2850</td>
      <td>3.730</td>
      <td>3.985</td>
      <td>4.32</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>
</div>




```python
heroes_df.columns
```




    Index(['Age', 'Gender', 'Item ID', 'Item Name', 'Price', 'SN'], dtype='object')




```python
# Perform sorting of columns
# Reorganizing the columns using double brackets
organized_heroes_df = heroes_df[["SN","Gender","Age","Item ID", "Item Name", "Price"]]
organized_heroes_df.head()

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
      <th>SN</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aelalis34</td>
      <td>Male</td>
      <td>38</td>
      <td>165</td>
      <td>Bone Crushing Silver Skewer</td>
      <td>3.37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Eolo46</td>
      <td>Male</td>
      <td>21</td>
      <td>119</td>
      <td>Stormbringer, Dark Blade of Ending Misery</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Assastnya25</td>
      <td>Male</td>
      <td>34</td>
      <td>174</td>
      <td>Primitive Blade</td>
      <td>2.46</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pheusrical25</td>
      <td>Male</td>
      <td>21</td>
      <td>92</td>
      <td>Final Critic</td>
      <td>1.36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aela59</td>
      <td>Male</td>
      <td>23</td>
      <td>63</td>
      <td>Stormfury Mace</td>
      <td>1.27</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Using iloc[] will not find duplicates since a numeric index is always unique
cleaned_unique_players = organized_heroes_df.iloc[:,0:6]
print(cleaned_unique_players)
```

                      SN                 Gender  Age  Item ID  \
    0          Aelalis34                   Male   38      165   
    1             Eolo46                   Male   21      119   
    2        Assastnya25                   Male   34      174   
    3       Pheusrical25                   Male   21       92   
    4             Aela59                   Male   23       63   
    5         Tanimnya91                   Male   20       10   
    6        Undjaskla97                   Male   20      153   
    7       Iathenudil29                 Female   29      169   
    8       Sondenasta63                   Male   25      118   
    9         Hilaerin92                   Male   31       99   
    10        Chamosia29                   Male   24       57   
    11           Sally64                   Male   20       47   
    12         Iskossa88                   Male   30       81   
    13   Seorithstilis90                   Male   23       77   
    14         Sundast29                   Male   40       44   
    15        Haellysu29                   Male   21       96   
    16        Sundista85                 Female   22      123   
    17         Aenarap34                 Female   22       59   
    18         Iskista88                   Male   28       91   
    19         Assossa43                   Male   31      177   
    20           Irith83                   Male   24       78   
    21       Iaralrgue74                   Male   15        3   
    22          Deural48                 Female   11       11   
    23        Chanosia65                   Male   19      183   
    24          Qarwen67                   Male   11       65   
    25            Idai61                   Male   21       63   
    26     Aerithllora36                   Male   29      132   
    27       Assastnya25                   Male   34      106   
    28       Ilariarin45                   Male   15       49   
    29         Phaedai25                 Female   16       45   
    ..               ...                    ...  ...      ...   
    750         Eollym91                   Male   23       86   
    751     Lisjasksda68                 Female   26      179   
    752    Yalostiphos68                 Female   15      116   
    753      Thryallym62                   Male   20        4   
    754      Sondastan54                   Male   31      104   
    755      Ailaesuir66                 Female   22      179   
    756         Siasri67                   Male   22        6   
    757         Seosri62                   Male   35       11   
    758      Ryastycal90                   Male   20      122   
    759    Chanirrasta87                   Male   19       87   
    760    Aerithllora36                   Male   29       81   
    761      Raeduerin33                   Male   28      175   
    762      Lisosiast26                   Male   36       52   
    763       Eurisuru25  Other / Non-Disclosed   27       48   
    764     Assassasda84                   Male   25       70   
    765    Aerithnucal56                   Male   15       13   
    766      Nitherian58                 Female   22       84   
    767      Hailaphos89                   Male   20      122   
    768     Chamucosda93                   Male   21      158   
    769   Frichilsasya78                   Male   24       73   
    770         Aenasu69                   Male   22      141   
    771       Lassista97                   Male   24       25   
    772          Sidap51                   Male   15       31   
    773    Chamadarsda63                   Male   21       44   
    774     Lassassast73                   Male   24      123   
    775          Eural50                   Male   22       98   
    776       Lirtossa78                   Male   14      104   
    777       Tillyrin30                   Male   20      117   
    778       Quelaton80                   Male   20       75   
    779           Alim85                 Female   23      107   
    
                                         Item Name  Price  
    0                  Bone Crushing Silver Skewer   3.37  
    1    Stormbringer, Dark Blade of Ending Misery   2.32  
    2                              Primitive Blade   2.46  
    3                                 Final Critic   1.36  
    4                               Stormfury Mace   1.27  
    5                                  Sleepwalker   1.73  
    6                              Mercenary Sabre   4.57  
    7       Interrogator, Blood Blade of the Queen   3.32  
    8             Ghost Reaver, Longsword of Magic   2.77  
    9         Expiration, Warscythe Of Lost Worlds   4.53  
    10             Despair, Favor of Due Diligence   3.81  
    11                 Alpha, Reach of Ending Hope   1.55  
    12                                   Dreamkiss   4.06  
    13                  Piety, Guardian of Riddles   3.68  
    14                       Bonecarvin Battle Axe   2.46  
    15                 Blood-Forged Skeletal Spine   4.77  
    16                           Twilight's Carver   1.14  
    17               Lightning, Etcher of the King   1.65  
    18                                     Celeste   3.71  
    19    Winterthorn, Defender of Shifting Worlds   4.89  
    20                  Glimmer, Ender of the Moon   2.33  
    21                                Phantomlight   1.79  
    22                                   Brimstone   2.52  
    23                         Dragon's Greatsword   2.36  
    24                   Conqueror Adamantite Mace   1.96  
    25                              Stormfury Mace   1.27  
    26                                  Persuasion   3.90  
    27                         Crying Steel Sickle   2.29  
    28            The Oculus, Token of Lost Worlds   4.23  
    29                         Glinting Glass Edge   2.46  
    ..                                         ...    ...  
    750                          Stormfury Lantern   1.28  
    751            Wolf, Promise of the Moonwalker   1.88  
    752                    Renewed Skeletal Katana   2.37  
    753                         Bloodlord's Fetish   2.28  
    754                         Gladiator's Glaive   1.36  
    755            Wolf, Promise of the Moonwalker   1.88  
    756                                Rusty Skull   1.20  
    757                                  Brimstone   2.52  
    758                           Unending Tyranny   1.21  
    759                   Deluge, Edge of the West   2.20  
    760                                  Dreamkiss   4.06  
    761                 Woeful Adamantite Claymore   1.24  
    762                                     Hatred   4.39  
    763            Rage, Legacy of the Lone Victor   4.32  
    764                                 Hope's End   3.89  
    765                                   Serenity   1.49  
    766                                 Arcane Gem   2.23  
    767                           Unending Tyranny   1.21  
    768         Darkheart, Butcher of the Champion   3.56  
    769                                Ritual Mace   3.74  
    770                                 Persuasion   3.27  
    771                                  Hero Cane   1.03  
    772                                  Trickster   2.07  
    773                      Bonecarvin Battle Axe   2.46  
    774                          Twilight's Carver   1.14  
    775                Deadline, Voice Of Subtlety   3.62  
    776                         Gladiator's Glaive   1.36  
    777          Heartstriker, Legacy of the Light   4.15  
    778                    Brutality Ivory Warmace   1.72  
    779                  Splitter, Foe Of Subtlety   3.61  
    
    [780 rows x 6 columns]
    


```python
demo_heroes = cleaned_unique_players.sort_values("SN")
demo_heroes.head()
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
      <th>SN</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>144</th>
      <td>Adairialis76</td>
      <td>Male</td>
      <td>20</td>
      <td>44</td>
      <td>Bonecarvin Battle Axe</td>
      <td>2.46</td>
    </tr>
    <tr>
      <th>308</th>
      <td>Aduephos78</td>
      <td>Male</td>
      <td>37</td>
      <td>79</td>
      <td>Alpha, Oath of Zeal</td>
      <td>2.88</td>
    </tr>
    <tr>
      <th>377</th>
      <td>Aduephos78</td>
      <td>Male</td>
      <td>37</td>
      <td>174</td>
      <td>Primitive Blade</td>
      <td>2.46</td>
    </tr>
    <tr>
      <th>431</th>
      <td>Aduephos78</td>
      <td>Male</td>
      <td>37</td>
      <td>92</td>
      <td>Final Critic</td>
      <td>1.36</td>
    </tr>
    <tr>
      <th>224</th>
      <td>Aeduera68</td>
      <td>Male</td>
      <td>26</td>
      <td>106</td>
      <td>Crying Steel Sickle</td>
      <td>2.29</td>
    </tr>
  </tbody>
</table>
</div>




```python
demo_heroes["SN"].unique()
```




    array(['Adairialis76', 'Aduephos78', 'Aeduera68', 'Aela49', 'Aela59',
           'Aelalis34', 'Aelin32', 'Aeliriam77', 'Aeliriarin93', 'Aeliru63',
           'Aellyria80', 'Aellyrialis39', 'Aellysup38', 'Aelollo59',
           'Aenarap34', 'Aenasu69', 'Aeral43', 'Aeral85', 'Aeral97', 'Aeri84',
           'Aerillorin70', 'Aerithllora36', 'Aerithnucal56', 'Aerithnuphos61',
           'Aerithriaphos45', 'Aesty51', 'Aesur96', 'Aethe80', 'Aethedru70',
           'Aidain51', 'Aidaira26', 'Aidaira48', 'Aiduecal76', 'Aiduesu83',
           'Ailaesuir66', 'Aillycal84', 'Aillyriadru65', 'Aina42', 'Aina43',
           'Airal46', 'Airi27', 'Airidil41', 'Airithrin43', 'Aisur51',
           'Aisurphos78', 'Aithelis62', 'Alaephos75', 'Alaesu77', 'Alaesu91',
           'Alallo58', 'Alarap40', 'Alim85', 'Alo67', 'Anallorgue57',
           'Arithllorin55', 'Assassa38', 'Assassa43', 'Assassasda84',
           'Assassasta79', 'Assastnya25', 'Assesi91', 'Assilsan72',
           'Assistasda90', 'Assistast50', 'Assithasta65', 'Assosia38',
           'Assosiasta83', 'Assossa43', 'Assylla81', 'Astydil38', 'Asur53',
           'Baelollodeu94', 'Bartassaya73', 'Billysu76', 'Chadadarya31',
           'Chadanto83', 'Chadjask77', 'Chadossa56', 'Chadossa89',
           'Chamadar27', 'Chamadar61', 'Chamadar79', 'Chamadarnya73',
           'Chamadarsda63', 'Chamastya76', 'Chamilsala65', 'Chamilsan75',
           'Chamim85', 'Chamimla73', 'Chamirra53', 'Chamirrasya33',
           'Chamirraya83', 'Chamistast30', 'Chamjasknya65', 'Chamosia29',
           'Chamucosda93', 'Chanadar44', 'Chanassa48', 'Chanastnya43',
           'Chanastsda67', 'Chanastst38', 'Chanirra56', 'Chanirra79',
           'Chanirrala39', 'Chanirrasta87', 'Chanjask65', 'Chanjaskan37',
           'Chanjaskan89', 'Chanosia60', 'Chanosia65', 'Chanosiast43',
           'Chanosiaya39', 'Chanosseya79', 'Chrathybust28', 'Cosadar58',
           'Crausirra42', 'Deelilsasya30', 'Deural48', 'Dyally87',
           'Ennoncil86', 'Eoda93', 'Eodailis27', 'Eoduenurin62', 'Eollym91',
           'Eolo46', 'Eoral49', 'Eoralphos86', 'Eosrirgue62', 'Eosur70',
           'Eosurdru76', 'Eosursurap97', 'Eothe56', 'Eratiel90', 'Ermol76',
           'Erudrion71', 'Eryon48', 'Ethralan59', 'Ethralista69',
           'Ethruard50', 'Ethrusuard41', 'Eudasu82', 'Eula35', 'Eulaeria40',
           'Eulidru49', 'Euliria52', 'Eullydru35', 'Euna48', 'Eural50',
           'Eurallo89', 'Eurinu48', 'Eurisuru25', 'Eurith26', 'Eusri26',
           'Eusri44', 'Eusri70', 'Eustyria89', 'Eusur90', 'Eyircil84',
           'Faralcil63', 'Farenon57', 'Farusrian86', 'Filon68', 'Filrion44',
           'Filrion59', 'Fironon91', 'Frichadar89', 'Frichassala85',
           'Frichast72', 'Frichaststa61', 'Frichaya88', 'Frichilsasya78',
           'Frichim27', 'Frichim77', 'Frichistast39', 'Frichistasta59',
           'Frichjask31', 'Frichjaskan98', 'Frichosiala98', 'Frichossast75',
           'Frichossast86', 'Haedasu65', 'Haellysu29', 'Haerith37',
           'Haerithp41', 'Hailaphos89', 'Hainaria90', 'Hala31',
           'Hallysucal81', 'Heolo60', 'Heosrisuir72', 'Heosurnuru52',
           'Heuli25', 'Heunadil74', 'Heuralsti66', 'Hiadanurin36',
           'Hiarideu73', 'Hiasri33', 'Hiasur92', 'Hilaerin92', 'Hiral75',
           'Hirirap39', 'Iadueria43', 'Ialallo29', 'Ialidru40',
           'Ialistidru50', 'Iallyphos37', 'Ialo60', 'Iaralrgue74',
           'Iaralsuir44', 'Iarilis73', 'Iarithdil76', 'Iasur80',
           'Iathenudil29', 'Idai61', 'Idairin80', 'Idaria87', 'Iduedru67',
           'Ila44', 'Iladarla40', 'Ilaesudil92', 'Ilariarin45', 'Ilassa51',
           'Ilast79', 'Ilaststa70', 'Iliel92', 'Ililsa62', 'Ililsan66',
           'Ilimya66', 'Ilirrasda54', 'Ilogha82', 'Ilophos58', 'Ilosia37',
           'Ilosu82', 'Ilrian97', 'Ina92', 'Indcil77', 'Indirrian56',
           'Indonmol95', 'Ingatcil75', 'Ingonon91', 'Inguard95', 'Inguron55',
           'Iral74', 'Iri67', 'Irillo49', 'Irith83', 'Irithrap69',
           'Iskadarya95', 'Iskassa50', 'Isketo41', 'Iskichinya81',
           'Iskimnya76', 'Iskirra45', 'Iskista88', 'Iskista96',
           'Iskistasda86', 'Iskjaskan81', 'Iskjaskst81', 'Iskosia51',
           'Iskosian40', 'Iskossa88', 'Iskossan49', 'Iskossasda43',
           'Iskossaya95', 'Isri49', 'Isri59', 'Isrirgue68', 'Isurria36',
           'Isurriarap71', 'Isursti83', 'Ithergue48', 'Ithesuphos68',
           'Jeyciman68', 'Jiskassa76', 'Jiskilsa35', 'Jiskimsda56',
           'Jiskirran77', 'Jiskjask80', 'Jiskosiala43', 'Jiskossa51',
           'Koikirra25', 'Lamil70', 'Lamil79', 'Lamon28', 'Lamyon68',
           'Lassadarsda57', 'Lassassasda30', 'Lassassast73', 'Lassast89',
           'Lassilsa41', 'Lassilsa63', 'Lassilsala30', 'Lassimla92',
           'Lassista97', 'Lassjask63', 'Lassjaskan73', 'Layjask75',
           'Leulaesti78', 'Leyirra83', 'Liawista80', 'Liri91', 'Lirtassa47',
           'Lirtast83', 'Lirtilsan89', 'Lirtista72', 'Lirtistanya48',
           'Lirtistasta79', 'Lirtosia72', 'Lirtossa78', 'Lirtossa84',
           'Lirtossan50', 'Lirtossanya27', 'Lirtyrdesta65', 'Lisadar44',
           'Lisasi93', 'Lisassa26', 'Lisassa39', 'Lisassa49', 'Lisassasta50',
           'Lisico81', 'Lisimsda29', 'Lisiriya82', 'Lisirra55', 'Lisirrast82',
           'Lisirraya76', 'Lisista27', 'Lisistasya93', 'Lisistaya47',
           'Lisjaskan36', 'Lisjasksda68', 'Lisjaskya84', 'Lisosianya62',
           'Lisosiast26', 'Lisossa25', 'Lisossa63', 'Lisossan98',
           'Lisossanya98', 'Lisovynya38', 'Malista67', 'Malunil62',
           'Marassanya92', 'Marassaya49', 'Marilsa48', 'Marilsanya48',
           'Marilsasya33', 'Marim28', 'Marirrasta50', 'Marjasksda39',
           'Marundi65', 'Meosridil82', 'Mindadaran26', 'Mindassast27',
           'Mindetosya30', 'Mindilsa60', 'Mindimnya67', 'Mindirra92',
           'Mindjasksya61', 'Mindosiasya28', 'Mindossa76', 'Mindossasya74',
           'Minduli80', 'Narirra38', 'Nitherian58', 'Palatyon26',
           'Palurrian69', 'Phadai31', 'Phadue96', 'Phaedai25', 'Phaedan76',
           'Phaeduesurgue38', 'Phaestycal84', 'Phainasu47', 'Phairinum94',
           'Phalinun47', 'Pharithdil38', 'Phenastya51', 'Pheodai94',
           'Pheusrical25', 'Phially37', 'Philistirap41', 'Philodil43',
           'Phistym51', 'Phyali88', 'Qaronon57', 'Qarrian82', 'Qarwen67',
           'Qilanrion65', 'Qilatie51', 'Qiluard68', 'Qilunan34',
           'Quanenrian83', 'Quarunarn52', 'Quarusrion32', 'Quelatarn54',
           'Quelaton80', 'Queusurra38', 'Quinarap53', 'Raedalis34',
           'Raeduerin33', 'Raelly43', 'Raeri71', 'Raerithsti62', 'Raesty92',
           'Raesurdil91', 'Raesursurap33', 'Raillydeu47', 'Raithe71',
           'Ralaeriadeu65', 'Ralasti48', 'Ralonurin90', 'Rarith48',
           'Rasrirgue43', 'Rathellorin54', 'Raysistast71', 'Reolacal36',
           'Reula64', 'Reulae52', 'Reuthelis39', 'Rina82', 'Rinallorap73',
           'Riralsti91', 'Ririp86', 'Ristydru66', 'Rithe53', 'Rithe77',
           'Ryanara76', 'Ryastycal90', 'Saedaiphos46', 'Saedue76',
           'Saelaephos52', 'Saellyra72', 'Saelollop56', 'Saena74',
           'Saerallora71', 'Saida58', 'Saidairiaphos61', 'Saisrilis27',
           'Saistydru69', 'Saistyphos30', 'Salilis27', 'Sally64', 'Saralp86',
           'Sausosia74', 'Seolollo93', 'Seorithstilis90', 'Seosri62',
           'Seudaillorap38', 'Seudanu38', 'Seuthelis34', 'Shaidanu32',
           'Shidai42', 'Sialaera37', 'Siarinum43', 'Siarithria38', 'Siasri67',
           'Siathecal92', 'Sida61', 'Sidap51', 'Silaera56', 'Silideu44',
           'Silinu63', 'Sirira97', 'Smecherdi88', 'Sondadar26', 'Sondassa48',
           'Sondassa68', 'Sondassasya91', 'Sondastan54', 'Sondenasta63',
           'Sondilsa35', 'Sondilsa40', 'Sondim43', 'Sondim68', 'Sondim73',
           'Sondimla25', 'Sondossa55', 'Sondossa91', 'Strairisti57',
           'Streural92', 'Strithenu87', 'Stryanastip77', 'Styaduen40',
           'Sundadarla27', 'Sundassa93', 'Sundast29', 'Sundast87',
           'Sundastnya66', 'Sundaststa26', 'Sundim98', 'Sundista85',
           'Sundjask71', 'Sundosiasta28', 'Sundossast30', 'Sweecossa42',
           'Syadaillo88', 'Syally44', 'Syalollorap93', 'Syasriria69',
           'Syathe73', 'Taeduenu92', 'Tanimnya91', 'Tauldilsa43',
           'Thourdirra92', 'Thryallym62', 'Tillyrin30', 'Tridaira71',
           'Tyadaru49', 'Tyaelistidru84', 'Tyaelly53', 'Tyaelo67',
           'Tyaelorgue39', 'Tyaenasti87', 'Tyaeristi78', 'Tyaerith73',
           'Tyaili86', 'Tyalaesu89', 'Tyananurgue44', 'Tyarithn67',
           'Tyeosri53', 'Tyeosristi57', 'Tyeuduen32', 'Tyeuduephos81',
           'Tyeuladeu30', 'Tyeulisu40', 'Tyiaduenuru55', 'Tyida79',
           'Tyidainu31', 'Tyidue95', 'Tyirithnu40', 'Tyisriphos58',
           'Tyisur83', 'Tyithesura58', 'Umuard36', 'Undadar97', 'Undadarla37',
           'Undare39', 'Undast38', 'Undirra73', 'Undirrala66', 'Undirrasta74',
           'Undirrasta89', 'Undistasta86', 'Undiwinya88', 'Undjaskla97',
           'Undjasksya56', 'Undotesta33', 'Wailin72', 'Whaestysu86',
           'Yadacal26', 'Yadaisuir65', 'Yadanun74', 'Yalaeria91', 'Yaliru88',
           'Yalo71', 'Yalostiphos68', 'Yaralnura48', 'Yararmol43',
           'Yarirarn35', 'Yaristi64', 'Yarithllodeu72', 'Yarithphos28',
           'Yarithsurgue62', 'Yarmol79', 'Yarolwen77', 'Yasriphos60',
           'Yasrisu92', 'Yasur35', 'Yasur85', 'Yasurra52', 'Yathecal72',
           'Yathecal82', 'Zhisrisu83', 'Zontibe81'], dtype=object)




```python
no_dup_heroes = demo_heroes.drop_duplicates(subset=["SN"], keep ='first')
no_dup_heroes.head()
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
      <th>SN</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>144</th>
      <td>Adairialis76</td>
      <td>Male</td>
      <td>20</td>
      <td>44</td>
      <td>Bonecarvin Battle Axe</td>
      <td>2.46</td>
    </tr>
    <tr>
      <th>308</th>
      <td>Aduephos78</td>
      <td>Male</td>
      <td>37</td>
      <td>79</td>
      <td>Alpha, Oath of Zeal</td>
      <td>2.88</td>
    </tr>
    <tr>
      <th>224</th>
      <td>Aeduera68</td>
      <td>Male</td>
      <td>26</td>
      <td>106</td>
      <td>Crying Steel Sickle</td>
      <td>2.29</td>
    </tr>
    <tr>
      <th>394</th>
      <td>Aela49</td>
      <td>Male</td>
      <td>25</td>
      <td>44</td>
      <td>Bonecarvin Battle Axe</td>
      <td>2.46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aela59</td>
      <td>Male</td>
      <td>23</td>
      <td>63</td>
      <td>Stormfury Mace</td>
      <td>1.27</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Answer to Question #3
no_dup_heroes["Gender"].value_counts()
```




    Male                     465
    Female                   100
    Other / Non-Disclosed      8
    Name: Gender, dtype: int64




```python
group_heroes_gender = heroes_df.groupby("Gender")
```


```python
# Answer to Question #4 - Age (Need to use the non-duplicated clean table)
group_heroes_gender.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">Age</th>
      <th colspan="5" halign="left">Item ID</th>
      <th colspan="8" halign="left">Price</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>...</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>136.0</td>
      <td>22.558824</td>
      <td>7.419459</td>
      <td>7.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>25.0</td>
      <td>40.0</td>
      <td>136.0</td>
      <td>88.110294</td>
      <td>...</td>
      <td>123.25</td>
      <td>182.0</td>
      <td>136.0</td>
      <td>2.815515</td>
      <td>1.151027</td>
      <td>1.03</td>
      <td>1.8275</td>
      <td>2.615</td>
      <td>3.750</td>
      <td>4.95</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>633.0</td>
      <td>22.685624</td>
      <td>6.804740</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>25.0</td>
      <td>45.0</td>
      <td>633.0</td>
      <td>91.571880</td>
      <td>...</td>
      <td>137.00</td>
      <td>183.0</td>
      <td>633.0</td>
      <td>2.950521</td>
      <td>1.109967</td>
      <td>1.03</td>
      <td>2.0400</td>
      <td>2.910</td>
      <td>3.910</td>
      <td>4.95</td>
    </tr>
    <tr>
      <th>Other / Non-Disclosed</th>
      <td>11.0</td>
      <td>27.363636</td>
      <td>6.932139</td>
      <td>12.0</td>
      <td>24.0</td>
      <td>27.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>11.0</td>
      <td>114.636364</td>
      <td>...</td>
      <td>156.00</td>
      <td>183.0</td>
      <td>11.0</td>
      <td>3.249091</td>
      <td>0.957230</td>
      <td>1.88</td>
      <td>2.2850</td>
      <td>3.730</td>
      <td>3.985</td>
      <td>4.32</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>
</div>




```python
# Answer to Question #5
heroes_df.head()
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
      <th>Age</th>
      <th>Gender</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
      <th>SN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>Male</td>
      <td>165</td>
      <td>Bone Crushing Silver Skewer</td>
      <td>3.37</td>
      <td>Aelalis34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>Male</td>
      <td>119</td>
      <td>Stormbringer, Dark Blade of Ending Misery</td>
      <td>2.32</td>
      <td>Eolo46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>Male</td>
      <td>174</td>
      <td>Primitive Blade</td>
      <td>2.46</td>
      <td>Assastnya25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>Male</td>
      <td>92</td>
      <td>Final Critic</td>
      <td>1.36</td>
      <td>Pheusrical25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>Male</td>
      <td>63</td>
      <td>Stormfury Mace</td>
      <td>1.27</td>
      <td>Aela59</td>
    </tr>
  </tbody>
</table>
</div>




```python
age_bins = [10, 15, 20, 25, 30, 35, 40, 45]
age_labels = ["<10", "10-15", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44"]
```


```python
heroes_df.dtypes
```




    Age            int64
    Gender        object
    Item ID        int64
    Item Name     object
    Price        float64
    SN            object
    dtype: object




```python
heroes_df_bins = pd.cut(heroes_df["Age"], bins=age_bins, labels=age_labels)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-172-48eab0e3791c> in <module>()
    ----> 1 heroes_df_bins = pd.cut(heroes_df["Age"], bins=age_bins, labels=age_labels)
    

    ~\AppData\Local\Continuum\anaconda3\envs\PythonData\lib\site-packages\pandas\core\reshape\tile.py in cut(x, bins, right, labels, retbins, precision, include_lowest)
        134                               precision=precision,
        135                               include_lowest=include_lowest,
    --> 136                               dtype=dtype)
        137 
        138     return _postprocess_for_cut(fac, bins, retbins, x_is_series,
    

    ~\AppData\Local\Continuum\anaconda3\envs\PythonData\lib\site-packages\pandas\core\reshape\tile.py in _bins_to_cuts(x, bins, right, labels, precision, include_lowest, dtype, duplicates)
        252         else:
        253             if len(labels) != len(bins) - 1:
    --> 254                 raise ValueError('Bin labels must be one fewer than '
        255                                  'the number of bin edges')
        256         if not is_categorical_dtype(labels):
    

    ValueError: Bin labels must be one fewer than the number of bin edges

