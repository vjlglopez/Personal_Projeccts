def get_cat_num_df(df):
    """
    Returns the names of the numerical and categorical
    features in the given dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe for which to retrieve 
        categorical and numerical features.
        
    Returns
    -------
    Tuple
        A tuple containing two lists, the first with the names
        of the numerical features and the second with the names
        of the categorical features.
    """
    num_df = []
    cat_df = []
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            num_df.append(col)
        if is_object_dtype(df[col]):
            cat_df.append(col)
    return num_df, cat_df


def get_details(df):
    """
    Prints details about the given dataframe, including the
    number of samples, features, categorical features,
    and numerical features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe for which to print details.
    """
    print(f'The Total Number of Samples of the Dataset:  {len(df)}' )
    print(f'The Total Number of Features of the Dataset: {len(df.columns)}')
    num_features, cat_features = get_cat_num_df(df)
    print(f'The Total Number of Categorical Features of the'
          f' Dataset: {len(cat_features)}')
    print(f'The Total Number of Numerical Features of the'
          f' Dataset:   {len(num_features)}')


def get_null_df(df):
    """
    Returns a dataframe containing information about the
    number and percentage of null values in each column of
    the input dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe for which to retrieve null value information.
        
    Returns
    -------
    pandas.DataFrame
        A dataframe containing the column names, column types,
        total number of null values, and percentage of null values
        for each column in the input dataframe.
    """
    null_df = pd.DataFrame(columns=['Column', 'Type', 'Total NaN', '%'])
    col_null = df.columns[df.isna().any()].to_list()
    L = len(df)
    for i, col in enumerate(col_null):
        T = 0
        if is_numeric_dtype(df[col]):
            T = "Numerical"
        else:
            T = "Categorical"
        nulls = len(df[df[col].isna() == True][col])
        null_df.loc[i] = {'Column': col,
                          'Type': T,
                          'Total NaN': nulls,
                          '%': (nulls / L)*100}
    return null_df


def get_outliers(df):
    """
    Get outlier information for numerical features in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing information about the outliers in the numerical
        features of the input DataFrame. The columns are:
        
        - Feature: The name of the numerical feature.
        - Total Outliers: The total number of outliers found in the feature.
        - Upper limit: The upper bound above which a value
                       is considered an outlier.
        - Lower limit: The lower bound below which a value
                       is considered an outlier.
    """
    num_feat, cat_feat = get_cat_num_df(df)
    outlier_df = pd.DataFrame(columns=['Feature', 'Total Outliers',
                                       'Upper limit', 'Lower limit'])
    for col in num_feat:
        first_quartile, third_quartile = (
            np.percentile(df[col], 25), np.percentile(df[col], 75)
        )
        iqr = third_quartile - first_quartile
        cutoff = iqr*1.5
        lower, upper = first_quartile - cutoff , third_quartile + cutoff
        upper_outliers = df[df[col] > upper]
        lower_outliers = df[df[col] < lower]
        total = lower_outliers.shape[0] + upper_outliers.shape[0]
        if total != 0 and (upper != 0 and lower != 0):
            outlier_df = (
                pd.concat([outlier_df,
                           pd.DataFrame({'Feature':col,
                                         'Total Outliers': total,
                                         'Upper limit': upper,
                                         'Lower limit':lower},
                                        index=[0])],
                          ignore_index=True)
            )
    return outlier_df


def plot_eda(df, target, plot_type):
    """
    Plots exploratory data analysis (EDA) for a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe with features and target columns.
    target : str
        The name of the target column in df.
    plot_type : str
        The type of plot to use. Can be 'scatter', 'box', or 'hist'.

    Returns
    -------
    None
        This function only displays the plots and does not return any values.

    Raises
    ------
    ValueError
        If plot_type is not one of 'scatter', 'box', or 'hist'.
    """
    df_x, df_y = df.drop(columns=target), df[target]
    
    num_feat, cat_feat = get_cat_num_df(df_x)

    # Determine number of rows and columns for subplots
    nrows = (len(num_feat) - 1) // 5 + 1
    ncols = min(len(num_feat), 5)

    # Create a figure and axis object
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 40))

    # Loop over data and create subplots
    for i, df in enumerate(num_feat):
        if i >= nrows * ncols:
            break
        row = i // ncols
        col = i % ncols
        ax = axs[row, col]
        if plot_type == 'scatter':
            sns.scatterplot(x=df_x[num_feat[i]], y=df_y, ax=ax)
            ax.set_title(f'Feature {i+1}')
        elif plot_type == 'box':
            sns.boxplot(y=df_x[num_feat[i]], ax=ax)
            ax.set_title(df_x[num_feat].columns[i])
        elif plot_type == 'hist':
            sns.histplot(x=df_x[num_feat[i]], kde=True, ax=ax)
            ax.set_title(f'Feature {i+1}')
        ax.set_ylabel('Target')
        ax.set_yticklabels([])

    # Hide the excess subplots without inputs
    for i in range(len(num_feat), nrows * ncols):
        row = i // ncols
        col = i % ncols
        axs[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()