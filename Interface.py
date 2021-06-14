# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:53:57 2021

@author: fredr
"""

import streamlit as st
import Lorenz_system as lz
import altair as alt
import pandas as pd
import numpy as np

#SYSTEM PARAMETERS 
x_0, y_0, z_0 = [0.001]*3 #Set Initial values
delta, TOL = 1e-3, 1e-4  #Set partition delta and quadrature error tolerance TOL
sigma, rho, beta = 10, 28, 8
t_limit = 40
t_final_shown = 40

def calculate():
    #CALCULATE AND FORMAT DATA      
    x,y,z,T = lz.run_lorenz(x_0, y_0, z_0, t_limit, delta, sigma, rho, beta, TOL)
    data = np.transpose(np.array([x,y,z,T]))
    return pd.DataFrame(data, columns = ["x","y","z","T"])  

#def make_charts(results):
#    #GENERATE AND UPLOAD CHARTS 
#    print("Rerunning charts")
#    chart = alt.Chart(results.head(t_final_shown*int(1/delta))).mark_line().encode(
#            x = horizontal_axis,
#            y = vertical_axis,
#            order = "T"        
#            ).configure_line(size = 0.5)    
#    return chart 

if __name__ == "__main__":
    results = calculate()
    st.altair_chart(make_charts(results))