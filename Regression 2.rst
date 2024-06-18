.. code:: ipython3

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error,r2_score

.. code:: ipython3

    data=pd.read_excel(r"C:\Users\Aryan Patro\Downloads\concrete+compressive+strength (2)\Concrete_Data.xls")

.. code:: ipython3

    data
    




.. raw:: html

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
          <th>Cement (component 1)(kg in a m^3 mixture)</th>
          <th>Blast Furnace Slag (component 2)(kg in a m^3 mixture)</th>
          <th>Fly Ash (component 3)(kg in a m^3 mixture)</th>
          <th>Water  (component 4)(kg in a m^3 mixture)</th>
          <th>Superplasticizer (component 5)(kg in a m^3 mixture)</th>
          <th>Coarse Aggregate  (component 6)(kg in a m^3 mixture)</th>
          <th>Fine Aggregate (component 7)(kg in a m^3 mixture)</th>
          <th>Age (day)</th>
          <th>Concrete compressive strength(MPa, megapascals)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>540.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>162.0</td>
          <td>2.5</td>
          <td>1040.0</td>
          <td>676.0</td>
          <td>28</td>
          <td>79.986111</td>
        </tr>
        <tr>
          <th>1</th>
          <td>540.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>162.0</td>
          <td>2.5</td>
          <td>1055.0</td>
          <td>676.0</td>
          <td>28</td>
          <td>61.887366</td>
        </tr>
        <tr>
          <th>2</th>
          <td>332.5</td>
          <td>142.5</td>
          <td>0.0</td>
          <td>228.0</td>
          <td>0.0</td>
          <td>932.0</td>
          <td>594.0</td>
          <td>270</td>
          <td>40.269535</td>
        </tr>
        <tr>
          <th>3</th>
          <td>332.5</td>
          <td>142.5</td>
          <td>0.0</td>
          <td>228.0</td>
          <td>0.0</td>
          <td>932.0</td>
          <td>594.0</td>
          <td>365</td>
          <td>41.052780</td>
        </tr>
        <tr>
          <th>4</th>
          <td>198.6</td>
          <td>132.4</td>
          <td>0.0</td>
          <td>192.0</td>
          <td>0.0</td>
          <td>978.4</td>
          <td>825.5</td>
          <td>360</td>
          <td>44.296075</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1025</th>
          <td>276.4</td>
          <td>116.0</td>
          <td>90.3</td>
          <td>179.6</td>
          <td>8.9</td>
          <td>870.1</td>
          <td>768.3</td>
          <td>28</td>
          <td>44.284354</td>
        </tr>
        <tr>
          <th>1026</th>
          <td>322.2</td>
          <td>0.0</td>
          <td>115.6</td>
          <td>196.0</td>
          <td>10.4</td>
          <td>817.9</td>
          <td>813.4</td>
          <td>28</td>
          <td>31.178794</td>
        </tr>
        <tr>
          <th>1027</th>
          <td>148.5</td>
          <td>139.4</td>
          <td>108.6</td>
          <td>192.7</td>
          <td>6.1</td>
          <td>892.4</td>
          <td>780.0</td>
          <td>28</td>
          <td>23.696601</td>
        </tr>
        <tr>
          <th>1028</th>
          <td>159.1</td>
          <td>186.7</td>
          <td>0.0</td>
          <td>175.6</td>
          <td>11.3</td>
          <td>989.6</td>
          <td>788.9</td>
          <td>28</td>
          <td>32.768036</td>
        </tr>
        <tr>
          <th>1029</th>
          <td>260.9</td>
          <td>100.5</td>
          <td>78.3</td>
          <td>200.6</td>
          <td>8.6</td>
          <td>864.5</td>
          <td>761.5</td>
          <td>28</td>
          <td>32.401235</td>
        </tr>
      </tbody>
    </table>
    <p>1030 rows × 9 columns</p>
    </div>



.. code:: ipython3

    data.info()


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1030 entries, 0 to 1029
    Data columns (total 9 columns):
     #   Column                                                 Non-Null Count  Dtype  
    ---  ------                                                 --------------  -----  
     0   Cement (component 1)(kg in a m^3 mixture)              1030 non-null   float64
     1   Blast Furnace Slag (component 2)(kg in a m^3 mixture)  1030 non-null   float64
     2   Fly Ash (component 3)(kg in a m^3 mixture)             1030 non-null   float64
     3   Water  (component 4)(kg in a m^3 mixture)              1030 non-null   float64
     4   Superplasticizer (component 5)(kg in a m^3 mixture)    1030 non-null   float64
     5   Coarse Aggregate  (component 6)(kg in a m^3 mixture)   1030 non-null   float64
     6   Fine Aggregate (component 7)(kg in a m^3 mixture)      1030 non-null   float64
     7   Age (day)                                              1030 non-null   int64  
     8   Concrete compressive strength(MPa, megapascals)        1030 non-null   float64
    dtypes: float64(8), int64(1)
    memory usage: 72.5 KB
    

.. code:: ipython3

    data.isnull().sum()




.. parsed-literal::

    Cement (component 1)(kg in a m^3 mixture)                0
    Blast Furnace Slag (component 2)(kg in a m^3 mixture)    0
    Fly Ash (component 3)(kg in a m^3 mixture)               0
    Water  (component 4)(kg in a m^3 mixture)                0
    Superplasticizer (component 5)(kg in a m^3 mixture)      0
    Coarse Aggregate  (component 6)(kg in a m^3 mixture)     0
    Fine Aggregate (component 7)(kg in a m^3 mixture)        0
    Age (day)                                                0
    Concrete compressive strength(MPa, megapascals)          0
    dtype: int64



.. code:: ipython3

    data.duplicated().sum()




.. parsed-literal::

    25



.. code:: ipython3

    data.dropna(inplace=True)#Syntax to drop null values

.. code:: ipython3

    data.drop_duplicates()




.. raw:: html

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
          <th>Cement (component 1)(kg in a m^3 mixture)</th>
          <th>Blast Furnace Slag (component 2)(kg in a m^3 mixture)</th>
          <th>Fly Ash (component 3)(kg in a m^3 mixture)</th>
          <th>Water  (component 4)(kg in a m^3 mixture)</th>
          <th>Superplasticizer (component 5)(kg in a m^3 mixture)</th>
          <th>Coarse Aggregate  (component 6)(kg in a m^3 mixture)</th>
          <th>Fine Aggregate (component 7)(kg in a m^3 mixture)</th>
          <th>Age (day)</th>
          <th>Concrete compressive strength(MPa, megapascals)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>540.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>162.0</td>
          <td>2.5</td>
          <td>1040.0</td>
          <td>676.0</td>
          <td>28</td>
          <td>79.986111</td>
        </tr>
        <tr>
          <th>1</th>
          <td>540.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>162.0</td>
          <td>2.5</td>
          <td>1055.0</td>
          <td>676.0</td>
          <td>28</td>
          <td>61.887366</td>
        </tr>
        <tr>
          <th>2</th>
          <td>332.5</td>
          <td>142.5</td>
          <td>0.0</td>
          <td>228.0</td>
          <td>0.0</td>
          <td>932.0</td>
          <td>594.0</td>
          <td>270</td>
          <td>40.269535</td>
        </tr>
        <tr>
          <th>3</th>
          <td>332.5</td>
          <td>142.5</td>
          <td>0.0</td>
          <td>228.0</td>
          <td>0.0</td>
          <td>932.0</td>
          <td>594.0</td>
          <td>365</td>
          <td>41.052780</td>
        </tr>
        <tr>
          <th>4</th>
          <td>198.6</td>
          <td>132.4</td>
          <td>0.0</td>
          <td>192.0</td>
          <td>0.0</td>
          <td>978.4</td>
          <td>825.5</td>
          <td>360</td>
          <td>44.296075</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1025</th>
          <td>276.4</td>
          <td>116.0</td>
          <td>90.3</td>
          <td>179.6</td>
          <td>8.9</td>
          <td>870.1</td>
          <td>768.3</td>
          <td>28</td>
          <td>44.284354</td>
        </tr>
        <tr>
          <th>1026</th>
          <td>322.2</td>
          <td>0.0</td>
          <td>115.6</td>
          <td>196.0</td>
          <td>10.4</td>
          <td>817.9</td>
          <td>813.4</td>
          <td>28</td>
          <td>31.178794</td>
        </tr>
        <tr>
          <th>1027</th>
          <td>148.5</td>
          <td>139.4</td>
          <td>108.6</td>
          <td>192.7</td>
          <td>6.1</td>
          <td>892.4</td>
          <td>780.0</td>
          <td>28</td>
          <td>23.696601</td>
        </tr>
        <tr>
          <th>1028</th>
          <td>159.1</td>
          <td>186.7</td>
          <td>0.0</td>
          <td>175.6</td>
          <td>11.3</td>
          <td>989.6</td>
          <td>788.9</td>
          <td>28</td>
          <td>32.768036</td>
        </tr>
        <tr>
          <th>1029</th>
          <td>260.9</td>
          <td>100.5</td>
          <td>78.3</td>
          <td>200.6</td>
          <td>8.6</td>
          <td>864.5</td>
          <td>761.5</td>
          <td>28</td>
          <td>32.401235</td>
        </tr>
      </tbody>
    </table>
    <p>1005 rows × 9 columns</p>
    </div>



.. code:: ipython3

    data.columns=['a','b','c','d','e','f','g','h','i']

.. code:: ipython3

    x=data.iloc[:,0:8]

.. code:: ipython3

    x




.. raw:: html

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
          <th>a</th>
          <th>b</th>
          <th>c</th>
          <th>d</th>
          <th>e</th>
          <th>f</th>
          <th>g</th>
          <th>h</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>540.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>162.0</td>
          <td>2.5</td>
          <td>1040.0</td>
          <td>676.0</td>
          <td>28</td>
        </tr>
        <tr>
          <th>1</th>
          <td>540.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>162.0</td>
          <td>2.5</td>
          <td>1055.0</td>
          <td>676.0</td>
          <td>28</td>
        </tr>
        <tr>
          <th>2</th>
          <td>332.5</td>
          <td>142.5</td>
          <td>0.0</td>
          <td>228.0</td>
          <td>0.0</td>
          <td>932.0</td>
          <td>594.0</td>
          <td>270</td>
        </tr>
        <tr>
          <th>3</th>
          <td>332.5</td>
          <td>142.5</td>
          <td>0.0</td>
          <td>228.0</td>
          <td>0.0</td>
          <td>932.0</td>
          <td>594.0</td>
          <td>365</td>
        </tr>
        <tr>
          <th>4</th>
          <td>198.6</td>
          <td>132.4</td>
          <td>0.0</td>
          <td>192.0</td>
          <td>0.0</td>
          <td>978.4</td>
          <td>825.5</td>
          <td>360</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1025</th>
          <td>276.4</td>
          <td>116.0</td>
          <td>90.3</td>
          <td>179.6</td>
          <td>8.9</td>
          <td>870.1</td>
          <td>768.3</td>
          <td>28</td>
        </tr>
        <tr>
          <th>1026</th>
          <td>322.2</td>
          <td>0.0</td>
          <td>115.6</td>
          <td>196.0</td>
          <td>10.4</td>
          <td>817.9</td>
          <td>813.4</td>
          <td>28</td>
        </tr>
        <tr>
          <th>1027</th>
          <td>148.5</td>
          <td>139.4</td>
          <td>108.6</td>
          <td>192.7</td>
          <td>6.1</td>
          <td>892.4</td>
          <td>780.0</td>
          <td>28</td>
        </tr>
        <tr>
          <th>1028</th>
          <td>159.1</td>
          <td>186.7</td>
          <td>0.0</td>
          <td>175.6</td>
          <td>11.3</td>
          <td>989.6</td>
          <td>788.9</td>
          <td>28</td>
        </tr>
        <tr>
          <th>1029</th>
          <td>260.9</td>
          <td>100.5</td>
          <td>78.3</td>
          <td>200.6</td>
          <td>8.6</td>
          <td>864.5</td>
          <td>761.5</td>
          <td>28</td>
        </tr>
      </tbody>
    </table>
    <p>1030 rows × 8 columns</p>
    </div>



.. code:: ipython3

    y=data.iloc[:,8]

.. code:: ipython3

    y




.. parsed-literal::

    0       79.986111
    1       61.887366
    2       40.269535
    3       41.052780
    4       44.296075
              ...    
    1025    44.284354
    1026    31.178794
    1027    23.696601
    1028    32.768036
    1029    32.401235
    Name: i, Length: 1030, dtype: float64



.. code:: ipython3

    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.10)

.. code:: ipython3

    model=LinearRegression()

.. code:: ipython3

    model.fit(xtrain,ytrain)




.. raw:: html

    <style>#sk-container-id-1 {
      /* Definition of color scheme common for light and dark mode */
      --sklearn-color-text: black;
      --sklearn-color-line: gray;
      /* Definition of color scheme for unfitted estimators */
      --sklearn-color-unfitted-level-0: #fff5e6;
      --sklearn-color-unfitted-level-1: #f6e4d2;
      --sklearn-color-unfitted-level-2: #ffe0b3;
      --sklearn-color-unfitted-level-3: chocolate;
      /* Definition of color scheme for fitted estimators */
      --sklearn-color-fitted-level-0: #f0f8ff;
      --sklearn-color-fitted-level-1: #d4ebff;
      --sklearn-color-fitted-level-2: #b3dbfd;
      --sklearn-color-fitted-level-3: cornflowerblue;
    
      /* Specific color for light theme */
      --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
      --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
      --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
      --sklearn-color-icon: #696969;
    
      @media (prefers-color-scheme: dark) {
        /* Redefinition of color scheme for dark theme */
        --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
        --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
        --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
        --sklearn-color-icon: #878787;
      }
    }
    
    #sk-container-id-1 {
      color: var(--sklearn-color-text);
    }
    
    #sk-container-id-1 pre {
      padding: 0;
    }
    
    #sk-container-id-1 input.sk-hidden--visually {
      border: 0;
      clip: rect(1px 1px 1px 1px);
      clip: rect(1px, 1px, 1px, 1px);
      height: 1px;
      margin: -1px;
      overflow: hidden;
      padding: 0;
      position: absolute;
      width: 1px;
    }
    
    #sk-container-id-1 div.sk-dashed-wrapped {
      border: 1px dashed var(--sklearn-color-line);
      margin: 0 0.4em 0.5em 0.4em;
      box-sizing: border-box;
      padding-bottom: 0.4em;
      background-color: var(--sklearn-color-background);
    }
    
    #sk-container-id-1 div.sk-container {
      /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
         but bootstrap.min.css set `[hidden] { display: none !important; }`
         so we also need the `!important` here to be able to override the
         default hidden behavior on the sphinx rendered scikit-learn.org.
         See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
      display: inline-block !important;
      position: relative;
    }
    
    #sk-container-id-1 div.sk-text-repr-fallback {
      display: none;
    }
    
    div.sk-parallel-item,
    div.sk-serial,
    div.sk-item {
      /* draw centered vertical line to link estimators */
      background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
      background-size: 2px 100%;
      background-repeat: no-repeat;
      background-position: center center;
    }
    
    /* Parallel-specific style estimator block */
    
    #sk-container-id-1 div.sk-parallel-item::after {
      content: "";
      width: 100%;
      border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
      flex-grow: 1;
    }
    
    #sk-container-id-1 div.sk-parallel {
      display: flex;
      align-items: stretch;
      justify-content: center;
      background-color: var(--sklearn-color-background);
      position: relative;
    }
    
    #sk-container-id-1 div.sk-parallel-item {
      display: flex;
      flex-direction: column;
    }
    
    #sk-container-id-1 div.sk-parallel-item:first-child::after {
      align-self: flex-end;
      width: 50%;
    }
    
    #sk-container-id-1 div.sk-parallel-item:last-child::after {
      align-self: flex-start;
      width: 50%;
    }
    
    #sk-container-id-1 div.sk-parallel-item:only-child::after {
      width: 0;
    }
    
    /* Serial-specific style estimator block */
    
    #sk-container-id-1 div.sk-serial {
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: var(--sklearn-color-background);
      padding-right: 1em;
      padding-left: 1em;
    }
    
    
    /* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
    clickable and can be expanded/collapsed.
    - Pipeline and ColumnTransformer use this feature and define the default style
    - Estimators will overwrite some part of the style using the `sk-estimator` class
    */
    
    /* Pipeline and ColumnTransformer style (default) */
    
    #sk-container-id-1 div.sk-toggleable {
      /* Default theme specific background. It is overwritten whether we have a
      specific estimator or a Pipeline/ColumnTransformer */
      background-color: var(--sklearn-color-background);
    }
    
    /* Toggleable label */
    #sk-container-id-1 label.sk-toggleable__label {
      cursor: pointer;
      display: block;
      width: 100%;
      margin-bottom: 0;
      padding: 0.5em;
      box-sizing: border-box;
      text-align: center;
    }
    
    #sk-container-id-1 label.sk-toggleable__label-arrow:before {
      /* Arrow on the left of the label */
      content: "▸";
      float: left;
      margin-right: 0.25em;
      color: var(--sklearn-color-icon);
    }
    
    #sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
      color: var(--sklearn-color-text);
    }
    
    /* Toggleable content - dropdown */
    
    #sk-container-id-1 div.sk-toggleable__content {
      max-height: 0;
      max-width: 0;
      overflow: hidden;
      text-align: left;
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-1 div.sk-toggleable__content.fitted {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    #sk-container-id-1 div.sk-toggleable__content pre {
      margin: 0.2em;
      border-radius: 0.25em;
      color: var(--sklearn-color-text);
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-1 div.sk-toggleable__content.fitted pre {
      /* unfitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    #sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
      /* Expand drop-down */
      max-height: 200px;
      max-width: 100%;
      overflow: auto;
    }
    
    #sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
      content: "▾";
    }
    
    /* Pipeline/ColumnTransformer-specific style */
    
    #sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Estimator-specific style */
    
    /* Colorize estimator box */
    #sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    #sk-container-id-1 div.sk-label label.sk-toggleable__label,
    #sk-container-id-1 div.sk-label label {
      /* The background is the default theme color */
      color: var(--sklearn-color-text-on-default-background);
    }
    
    /* On hover, darken the color of the background */
    #sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    /* Label box, darken color on hover, fitted */
    #sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Estimator label */
    
    #sk-container-id-1 div.sk-label label {
      font-family: monospace;
      font-weight: bold;
      display: inline-block;
      line-height: 1.2em;
    }
    
    #sk-container-id-1 div.sk-label-container {
      text-align: center;
    }
    
    /* Estimator-specific */
    #sk-container-id-1 div.sk-estimator {
      font-family: monospace;
      border: 1px dotted var(--sklearn-color-border-box);
      border-radius: 0.25em;
      box-sizing: border-box;
      margin-bottom: 0.5em;
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-1 div.sk-estimator.fitted {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    /* on hover */
    #sk-container-id-1 div.sk-estimator:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-1 div.sk-estimator.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Specification for estimator info (e.g. "i" and "?") */
    
    /* Common style for "i" and "?" */
    
    .sk-estimator-doc-link,
    a:link.sk-estimator-doc-link,
    a:visited.sk-estimator-doc-link {
      float: right;
      font-size: smaller;
      line-height: 1em;
      font-family: monospace;
      background-color: var(--sklearn-color-background);
      border-radius: 1em;
      height: 1em;
      width: 1em;
      text-decoration: none !important;
      margin-left: 1ex;
      /* unfitted */
      border: var(--sklearn-color-unfitted-level-1) 1pt solid;
      color: var(--sklearn-color-unfitted-level-1);
    }
    
    .sk-estimator-doc-link.fitted,
    a:link.sk-estimator-doc-link.fitted,
    a:visited.sk-estimator-doc-link.fitted {
      /* fitted */
      border: var(--sklearn-color-fitted-level-1) 1pt solid;
      color: var(--sklearn-color-fitted-level-1);
    }
    
    /* On hover */
    div.sk-estimator:hover .sk-estimator-doc-link:hover,
    .sk-estimator-doc-link:hover,
    div.sk-label-container:hover .sk-estimator-doc-link:hover,
    .sk-estimator-doc-link:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
    .sk-estimator-doc-link.fitted:hover,
    div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
    .sk-estimator-doc-link.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    /* Span, style for the box shown on hovering the info icon */
    .sk-estimator-doc-link span {
      display: none;
      z-index: 9999;
      position: relative;
      font-weight: normal;
      right: .2ex;
      padding: .5ex;
      margin: .5ex;
      width: min-content;
      min-width: 20ex;
      max-width: 50ex;
      color: var(--sklearn-color-text);
      box-shadow: 2pt 2pt 4pt #999;
      /* unfitted */
      background: var(--sklearn-color-unfitted-level-0);
      border: .5pt solid var(--sklearn-color-unfitted-level-3);
    }
    
    .sk-estimator-doc-link.fitted span {
      /* fitted */
      background: var(--sklearn-color-fitted-level-0);
      border: var(--sklearn-color-fitted-level-3);
    }
    
    .sk-estimator-doc-link:hover span {
      display: block;
    }
    
    /* "?"-specific style due to the `<a>` HTML tag */
    
    #sk-container-id-1 a.estimator_doc_link {
      float: right;
      font-size: 1rem;
      line-height: 1em;
      font-family: monospace;
      background-color: var(--sklearn-color-background);
      border-radius: 1rem;
      height: 1rem;
      width: 1rem;
      text-decoration: none;
      /* unfitted */
      color: var(--sklearn-color-unfitted-level-1);
      border: var(--sklearn-color-unfitted-level-1) 1pt solid;
    }
    
    #sk-container-id-1 a.estimator_doc_link.fitted {
      /* fitted */
      border: var(--sklearn-color-fitted-level-1) 1pt solid;
      color: var(--sklearn-color-fitted-level-1);
    }
    
    /* On hover */
    #sk-container-id-1 a.estimator_doc_link:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    #sk-container-id-1 a.estimator_doc_link.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-3);
    }
    </style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div>



.. code:: ipython3

    predict=model.predict(xtest)

.. code:: ipython3

    predict




.. parsed-literal::

    array([39.84870004, 30.80880339, 46.92288291, 36.92477354, 21.35663402,
           47.81276112, 25.43576678, 25.39875529, 60.91126869, 56.56256647,
           27.28755455, 33.85943357, 26.06720854, 24.60241565, 28.62258523,
           28.54190348, 16.33375465, 31.14793077, 37.77288165, 35.26201939,
           64.0521796 , 27.61105072, 56.23142624, 28.6216876 , 56.18405542,
           19.11402789, 54.34346282, 26.94780211, 52.02806423, 48.4975559 ,
           22.61505776, 58.22691036, 32.08948719, 27.60949469, 33.26388982,
           34.78334956, 32.53208363, 50.65991307, 30.9924235 , 63.59596442,
           43.89559238, 33.78230933, 53.8986191 , 13.93533168, 57.11738285,
           58.12540957, 31.83633372, 45.78218211, 55.76040359, 37.24753203,
           28.61081787, 56.67889234, 59.79279837, 33.66015242, 53.80908106,
           24.9554339 , 51.12470689, 22.51012079, 47.84343581, 20.2144777 ,
           49.36381689, 63.6851809 , 29.77348895, 26.14288293, 29.1782279 ,
           23.47524498, 25.82833406, 31.06708841, 26.05504423, 48.08488987,
           28.97558116, 60.91126869, 31.63766857, 24.831724  , 33.23677414,
           53.38698961, 29.80071904, 29.32042277, 39.46392747, 23.0453318 ,
           18.60163852, 30.05502722, 49.97215539, 19.91023039, 32.8919784 ,
           25.97272673, 28.3066149 , 27.10955534, 34.23030401, 24.21622715,
           36.79479263, 27.59555783, 27.66218485, 27.33522637, 19.39109942,
           51.5695506 , 31.73892717, 18.1865417 , 25.06258734, 41.53507815,
           53.50721918, 22.57623167, 23.44710698])





.. code:: ipython3

    mean_absolute_error(ytest,predict)




.. parsed-literal::

    8.216665697723537



.. code:: ipython3

    mean_squared_error(ytest,predict)




.. parsed-literal::

    98.71023476707187



.. code:: ipython3

    r2_score(ytest,predict)




.. parsed-literal::

    0.6713357073973201



