import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import plotly.express as px

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.sidebar.checkbox("Add filters like market maker, lead manager, etc")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.sidebar.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 100:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


def load_data(file_name):
    """
    Load data from an Excel file into a Pandas DataFrame.

    Parameters
    ----------
    file_name : str
        The path to the Excel file.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame containing the data from the Excel file.
    """
    df = pd.read_excel(file_name, index_col=0)
    return df

def intersection(lst1, lst2):
    
    """
    Find the intersection of two lists.

    Parameters
    ----------
    lst1, lst2 : list
        The two lists to find the intersection of.

    Returns
    -------
    intersection_list : list
        The list of elements that are common to both lists.
    """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

pct_change_df = load_data('ipo_candle_data_pct_change.xlsx')
close_candle_df = load_data('ipo_candle_data_close_candles.xlsx')
df = load_data('IPO_Data_2024.xlsx')  

add_slider = st.sidebar.slider(
    'Select a range of Days',
    pct_change_df.index.min(), pct_change_df.index.max(), (pct_change_df.index.min(), 250)
)

df = filter_dataframe(df)
stocks_filtered = df['Symbol'].to_list()
intersection_stock_list = intersection(stocks_filtered, pct_change_df.columns)

specific_day = None
specific_percentage = None
if st.sidebar.checkbox('Add Day/Percentage Filter combo'):
    specific_day = st.sidebar.slider('Select a Day', pct_change_df.index.min(), pct_change_df.index.max(), (50))
    specific_percentage = st.sidebar.slider('Select stocks having more than this percentage on above Selected day', 0, 100, (20))

# print(type(specific_day))
# print(type(specific_percentage))

if (specific_day is not None) and (specific_percentage is not None):
    stocks_day_percent_filtered_series = pct_change_df.iloc[specific_day].gt(specific_percentage)
    stocks_day_percent_filtered = stocks_day_percent_filtered_series[stocks_day_percent_filtered_series == True].index
    intersection_stock_list = intersection(intersection_stock_list, stocks_day_percent_filtered)

fig = px.line(pct_change_df[intersection_stock_list].loc[add_slider[0]:add_slider[1]])
# fig.update_xaxes(showspikes=True)
# fig.update_yaxes(showspikes=True)

st.plotly_chart(fig)

@st.cache_data
def day_level_analysis(pct_change_df, stock_price_df, meta_data_df, min_stock_traded_days):
    N_Days = min_stock_traded_days
    stocks_with_N_days_data = pct_change_df.loc[N_Days].notna().index[pct_change_df.loc[N_Days].notna()].tolist()
    # print(len(stocks_with_N_days_data))

    start_day = 10
    max_day = 200 #200
    day_steps = 10

    buy_day_df_analysis_map = {}
    buy_day_pct_change_df_analysis_map ={}
    for buy_day_number in range(start_day, max_day, day_steps):
        #get list of stocks on Day X into 2 lists L1: dayX > day 0, L2: dayX <= day0
        # print(buy_day_number)
        df_copy = pct_change_df.copy()

        df_copy = df_copy[stocks_with_N_days_data]
        # df_copy

        # Get values for Day 0 and Day X
        day_0_values = df_copy.loc[0]
        day_X_values = df_copy.loc[buy_day_number]

        # Initialize lists
        stocks_on_day_x_gt_0 = []  # Values at Day X > Day 0
        stocks_on_day_x_lte_0 = []  # Values at Day X <= Day 0

        # Compare values and populate lists
        for stock_name in df_copy.columns:
            # Check if the value at Day X is NaN
            if pd.isna(day_X_values[stock_name]):
                continue  # Skip this stock_name if Day X value is NaN
            
            if day_X_values[stock_name] > day_0_values[stock_name]:
                stocks_on_day_x_gt_0.append(stock_name)
            else:
                stocks_on_day_x_lte_0.append(stock_name)

        # Display results
        stocks_on_day_x_gt_0.sort()
        stocks_on_day_x_lte_0.sort()
        # print("List 1 (Day {} > Day 0):".format(buy_day_number), len(stocks_on_day_x_gt_0), " -> ", stocks_on_day_x_gt_0)
        # print("List 2 (Day {} <= Day 0):".format(buy_day_number), len(stocks_on_day_x_lte_0), " -> ", stocks_on_day_x_lte_0)

        day_num_list = []
        profit_list1_count = []
        loss_list1_count = []
        profit_list2_count = []
        loss_list2_count = []
        profit_pct_change_list1 = []
        loss_pct_change_list1 = []
        profit_pct_change_list2 = []
        loss_pct_change_list2 = []

        for day_num in range(buy_day_number, max_day, day_steps):
            # Now consider only the stock_names in stocks_on_day_x_gt_0 for the new profit and loss lists
            day_num_list.append(day_num)
            profit_list = []
            loss_list = []

            profit_list_invested_amount = 0
            profit_list_current_amount = 0
            loss_list_invested_amount = 0
            loss_list_current_amount = 0

            # Check values at Day (buy_day_number + day_steps) against Day buy_day_number for columns in list1
            for stock_name in stocks_on_day_x_gt_0:

                if pd.isna(df_copy.loc[day_num, stock_name]):
                    continue  # Skip this stock_name if Day X value is NaN

                stock_lot_size = meta_data_df[meta_data_df['Symbol'] == stock_name]['Lot Size'].values[0]
                day_x_stock_price = stock_price_df.loc[day_num, stock_name]
                buy_day_stock_price = stock_price_df.loc[buy_day_number, stock_name]
                stock_invested_amt = buy_day_stock_price * stock_lot_size
                stock_current_amt = day_x_stock_price * stock_lot_size
                # print(stock_name, " --> ", stock_lot_size, " --> ", buy_day_stock_price, " --> ", day_x_stock_price)

                profit_list_invested_amount += stock_invested_amt
                profit_list_current_amount += stock_current_amt
                
                if(day_num == buy_day_number):
                    profit_list.append(stock_name)
                    # profit_list_invested_amount += stock_invested_amt
                    # profit_list_current_amount += stock_current_amt
                else:
                    if df_copy.loc[day_num, stock_name] > df_copy.loc[buy_day_number, stock_name]:
                        profit_list.append(stock_name)
                        # profit_list_invested_amount += stock_invested_amt
                        # profit_list_current_amount += stock_current_amt
                    else:
                        loss_list.append(stock_name)
                        # loss_list_invested_amount += stock_invested_amt
                        # loss_list_current_amount += stock_current_amt

            # print(profit_list_current_amount, profit_list_invested_amount, loss_list_current_amount, loss_list_invested_amount)
            
            if profit_list_invested_amount == 0:
                profit_pct_change = 0
            else:
                profit_pct_change = ((profit_list_current_amount - profit_list_invested_amount) / profit_list_invested_amount) * 100
            
            if loss_list_invested_amount == 0:
                loss_pct_change = 0
            else:
                loss_pct_change = ((loss_list_current_amount - loss_list_invested_amount) / loss_list_invested_amount) * 100
            # profit_pct_change = ((profit_list_current_amount - profit_list_invested_amount) / profit_list_invested_amount) * 100
            # loss_pct_change = ((loss_list_current_amount - loss_list_invested_amount) / loss_list_invested_amount) * 100
            profit_pct_change_list1.append(profit_pct_change)
            loss_pct_change_list1.append(loss_pct_change)

            profit_list.sort()
            loss_list.sort()
            profit_list1_count.append(len(profit_list))
            loss_list1_count.append(len(loss_list))
            # print("Profit List gt Day0 (Day {} > Day {}):".format(day_num, buy_day_number), len(profit_list), " -> ", profit_list)
            # print("Loss List gt Day0 (Day {} <= Day {}):".format(day_num, buy_day_number), len(loss_list), " -> ", loss_list)

            profit_list = []
            loss_list = []

            profit_list_invested_amount = 0
            profit_list_current_amount = 0
            loss_list_invested_amount = 0
            loss_list_current_amount = 0

            # Check values at Day (buy_day_number + day_steps) against Day buy_day_number for columns in list1
            for stock_name in stocks_on_day_x_lte_0:
                if pd.isna(df_copy.loc[day_num, stock_name]):
                    continue  # Skip this stock_name if Day X value is NaN

                stock_lot_size = meta_data_df[meta_data_df['Symbol'] == stock_name]['Lot Size'].values[0]
                day_x_stock_price = stock_price_df.loc[day_num, stock_name]
                buy_day_stock_price = stock_price_df.loc[buy_day_number, stock_name]
                stock_invested_amt = buy_day_stock_price * stock_lot_size
                stock_current_amt = day_x_stock_price * stock_lot_size

                if(day_num == buy_day_number):
                    profit_list.append(stock_name)
                    profit_list_invested_amount += stock_invested_amt
                    profit_list_current_amount += stock_current_amt
                else:
                    if df_copy.loc[day_num, stock_name] > df_copy.loc[buy_day_number, stock_name]:
                        profit_list.append(stock_name)
                        profit_list_invested_amount += stock_invested_amt
                        profit_list_current_amount += stock_current_amt
                    else:
                        loss_list.append(stock_name)
                        loss_list_invested_amount += stock_invested_amt
                        loss_list_current_amount += stock_current_amt

            if profit_list_invested_amount == 0:
                profit_pct_change = 0
            else:
                profit_pct_change = ((profit_list_current_amount - profit_list_invested_amount) / profit_list_invested_amount) * 100
            
            if loss_list_invested_amount == 0:
                loss_pct_change = 0
            else:
                loss_pct_change = ((loss_list_current_amount - loss_list_invested_amount) / loss_list_invested_amount) * 100
                
            # profit_pct_change = ((profit_list_current_amount - profit_list_invested_amount) / profit_list_invested_amount) * 100
            # loss_pct_change = ((loss_list_current_amount - loss_list_invested_amount) / loss_list_invested_amount) * 100
            profit_pct_change_list2.append(profit_pct_change)
            loss_pct_change_list2.append(loss_pct_change)
            
            profit_list.sort()
            loss_list.sort()
            profit_list2_count.append(len(profit_list))
            loss_list2_count.append(len(loss_list))
            # print("Profit List lte Day0 (Day {} > Day {}):".format(day_num, buy_day_number), len(profit_list), " -> ", profit_list)
            # print("Loss List lte Day0 (Day {} <= Day {}):".format(day_num, buy_day_number), len(loss_list), " -> ", loss_list)
        
        # print("Day List :", len(day_num_list), " -> ", day_num_list)
        # print("profit_list1_count :", len(profit_list1_count), " -> ", profit_list1_count)
        # print("loss_list1_count :", len(loss_list1_count), " -> ", loss_list1_count)
        # print("profit_list2_count :", len(profit_list2_count), " -> ", profit_list2_count)
        # print("loss_list2_count :", len(loss_list2_count), " -> ", loss_list2_count)
        
        day_level_data = {
            'Day': day_num_list,
            str(buy_day_number) + '_L1_P': profit_list1_count,
            str(buy_day_number) + '_L1_L': loss_list1_count,
            str(buy_day_number) + '_L2_P': profit_list2_count,
            str(buy_day_number) + '_L2_L': loss_list2_count
        }

        day_level_pct_change_data = {
            'Day': day_num_list,
            str(buy_day_number) + '_L1_P_pct_chg': profit_pct_change_list1,
            str(buy_day_number) + '_L1_L_pct_chg': loss_pct_change_list1,
            str(buy_day_number) + '_L2_P_pct_chg': profit_pct_change_list2,
            str(buy_day_number) + '_L2_L_pct_chg': loss_pct_change_list2
        }

        day_level_df = pd.DataFrame(day_level_data)
        day_level_df.set_index('Day', inplace=True)
        # print(day_level_df)

        day_level_pct_change_df = pd.DataFrame(day_level_pct_change_data)
        day_level_pct_change_df.set_index('Day', inplace=True)
        # print(day_level_pct_change_df)

        buy_day_df_analysis_map[buy_day_number] = day_level_df
        buy_day_pct_change_df_analysis_map[buy_day_number] = day_level_pct_change_df
        print("=" * 50)

    # print(buy_day_df_analysis_map)
    print(buy_day_pct_change_df_analysis_map)
    return [buy_day_df_analysis_map, buy_day_pct_change_df_analysis_map]



if st.sidebar.checkbox('Show analysis based on buy day'):
    min_stock_traded_days = st.sidebar.slider('consider stocks with atleast this many days', 0, pct_change_df.index.max(), (100), 10)
    buy_day_analysis = st.sidebar.slider('Select a Day', 0, pct_change_df.index.max(), (50), 10)

    [day_level_analysis_dict, day_level_pct_change_analysis_dict] = day_level_analysis(pct_change_df[intersection_stock_list], close_candle_df, df, min_stock_traded_days)
    # print(day_level_analysis_dict[10])
    
    stock_count_analysis_fig = px.line(day_level_analysis_dict[buy_day_analysis])
    # stock_count_analysis_fig.update_xaxes(showspikes=True)
    # stock_count_analysis_fig.update_yaxes(showspikes=True)

    st.plotly_chart(stock_count_analysis_fig)

    stock_pct_change_analysis_fig = px.line(day_level_pct_change_analysis_dict[buy_day_analysis])
    # stock_pct_change_analysis_fig.update_xaxes(showspikes=True)
    # stock_pct_change_analysis_fig.update_yaxes(showspikes=True)

    st.plotly_chart(stock_pct_change_analysis_fig)