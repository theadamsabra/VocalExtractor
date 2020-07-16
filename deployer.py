# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:11:21 2020

@author: 19512
"""

import streamlit as st
import pandas as pd
import numpy as np
st.title('Title!')

df = pd.DataFrame({
    'Col 1': [2,3,3,4],
    'Col 2': [3,3,32,5]
    })


if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)