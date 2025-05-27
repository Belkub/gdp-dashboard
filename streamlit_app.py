import numpy as np

import pandas as pd


from scipy.optimize import curve_fit 



import torch
import pickle
import streamlit as st



st.title('Калькулятор Альбрехта-MV')
col1, col2 = st.columns(2)

pac = col1.number_input('PAC', min_value = 2.0, max_value = 5.0, value = 3.0, step = 0.1)
flom = col1.number_input('Flomin', min_value = 0.2, value = 0.3, max_value = 0.45, step = 0.01)
#pac = float(input('Кнцентрация PAC: '))
#flom = float(input('Концентрация Flomin: '))
rates = {'2':2, '3':3}
conc = float(col1.selectbox('Концентрация', list(rates)))
#conc = float(input('Концентрация суспензии: '))
a = float(col1.number_input('Точность', min_value = 0.1, max_value = 1.0, value = 0.5, step = 0.1))
n = st.checkbox('Нейросеть')
f600 = col2.slider('f600', 20, 65, 45)
#f600 = float(input('f600: '))
f3 = col2.slider('f3', 2, 25, 23)
#f3 = float(input('f3: '))
gel_1 = col2.slider('GEL_1', 4, 30, 26)
#gel_1 = float(input('GEL_1: '))
gel = col2.slider('GEL_10', 6, 68, 62)
#gel = float(input('GEL_10: '))
tics = gel/gel_1


#def dff():
if st.button('Рассчет'):
    #global pac, flom, conc, f600, f3, gel, tics
   
    df_n = pd.DataFrame({'PAC':[pac],'Flomin':[flom],'Conc':[conc],'f600':[f600],'f3':[f3],'Gel10':[gel],'Ticsotrop':[tics]})

    with open('/mount/src/gdp-dashboard/scaler.pkl', 'rb') as u:
        scaler = pickle.load(u)
        
    with open('data_pr.pkl', 'rb') as f:
        data_preprocessor = pickle.load(f)
    if n:
        g = ['f600', 'f3', 'Gel10', 'Ticsotrop']
        df_n_scaled = scaler.transform(df_n[g])
        df_n_ = pd.DataFrame(df_n_scaled, columns=g)
        df_n_ = torch.FloatTensor(df_n_.values)
        with open('new_line.pkl', 'rb') as f:
            d = pickle.load(f)
        d = pickle.loads(d)
        Acc_train, roc_train, Acc_test, pred, test, n, roc_test, df_n_new = d(4, 78, 1, 0.3, 0.89, df_n_)
        y_new_pred_n = df_n_new.flatten().tolist()
    
##        if a != 0.5: 
        y_new_n = []
        for i in y_new_pred_n:
            if i >= a:
                y_new_n.append(1)
            else:
                y_new_n.append(0)
##        else:
##            y_new_n = []
##            for i in y_new_pred_n:
##                if i >= 0.5:
##                    y_new_n.append(1)
##            else:
##                y_new_n.append(0)
        df_n['Class'] = y_new_n
        df_n['prob'] = y_new_pred_n
    else:
        df_n_ = pd.DataFrame(
        data_preprocessor.transform(df_n),
        columns=data_preprocessor.get_feature_names_out()    
        )

        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)  
    
    
        y_new_pred = model.predict_proba(df_n_)[:,1]
        
##        if a != 0.5:
        y_new = []
        for i in y_new_pred:
            if i >= a:
                y_new.append(1)
            else:
                y_new.append(0)
##        else:
##            y_new = model.predict(df_n_)
    
        df_n['Class'] = y_new
        df_n['prob'] = y_new_pred
##    df_n.loc[(df_n['Conc'] == 3) & (df_n['Gel10'] > 58) & (df_n['f600'] > 58), 'Class'] = 0
##    df_n.loc[(df_n['Conc'] == 2) & (df_n['f600'] > 36), 'Class'] = 0
    df_n.loc[(df_n['Conc'] == 3) & ((df_n['Gel10'] >= 58) | (df_n['f600'] >= 58) | (df_n['f600'] <= 45) | (df_n['Gel10'] <= 34)), 'Class'] = 0
    df_n.loc[(df_n['Conc'] == 2) & ((df_n['f600'] >= 35) | (df_n['Gel10'] >= 23) | (df_n['f600'] <= 25) | (df_n['Gel10'] < 13)), 'Class'] = 0
    if list(df_n['Class'])[-1] == 0:
        
        st.error(f'Рецептура нуждается в коррекции, класс последнего измерения: {round(list(df_n['Class'])[-1])}')
        st.error(f'Вероятность класса: {round(list(df_n['prob'])[-1],3)}')
##        print('Рецептура нуждается в коррекции, класс последнего измерения: ', round(list(df_n['Class'])[-1]))
##        print('Вероятность класса: ', round(list(df_n['prob'])[-1],3))
        
    else:
        st.success(f'Рецептура соответствует требованиям, класс последнего измерения: {round(list(df_n['Class'])[-1])}')
        st.success(f'Вероятность класса: {round(list(df_n['prob'])[-1],3)}')
##        print('Рецептура соответствует нормм, класс последнего измерения: ', round(list(df_n['Class'])[-1]))
##        print('Вероятность класса: ', round(list(df_n['prob'])[-1],3))
        
    def parameter(Fmin, Fmax, Pmin, Pmax, A1, A2, A3, A4, pac, flom):
        a1 = A1*Fmin
        a2 = A2*Fmax
        amin = [a1, a2]
        bflom = [Fmin, Fmax]
        def mop(values_x,a,b):
            return a * values_x + b 
        values_x = bflom
        values_y = amin
        args_1, covar = curve_fit(mop, values_x, values_y)
    
        a3 = Fmin*A3
        a4 = Fmax*A4
        amax = [a3, a4]
        values_x = bflom
        values_y = amax
        args_2, covar = curve_fit(mop, values_x, values_y)
    
        c1 = [args_1[0], args_2[0]] 
        values_x = [Pmin, Pmax]
        values_y = c1
        args_3, covar = curve_fit(mop, values_x, values_y)
        c2 = [args_1[1], args_2[1]] 
        values_x = [Pmin, Pmax]
        values_y = c2
        args_4, covar = curve_fit(mop, values_x, values_y)
        a1 = A1*Pmin
        a2 = A3*Pmax
        amin = [a1, a2]
        bflom = [Pmin, Pmax]
        values_x = bflom
        values_y = amin
        args_111, covar = curve_fit(mop, values_x, values_y)
        a3 = Pmin*A2
        a4 = Pmax*A4
        amax = [a3, a4]
        values_x = bflom
        values_y = amax
        args_222, covar = curve_fit(mop, values_x, values_y)
        c1 = [args_111[0], args_222[0]] 
        values_x = [0.15, 0.4]
        values_y = c1
        args_333, covar = curve_fit(mop, values_x, values_y)
        c2 = [args_111[1], args_222[1]] 
        values_x = [0.15, 0.4]
        values_y = c2
        args_444, covar = curve_fit(mop, values_x, values_y)
        R1 = ((args_3[0]*pac + args_3[1])*flom + (args_4[0]*pac + args_4[1]))/flom
        R2 = ((args_333[0]*flom + args_333[1])*pac + (args_444[0]*flom + args_444[1]))/pac
        return (R1+R2)/2
    PAC = []
    Flomin = []
    
    #if df_n.iloc[-1, -2] == 0:
    if df_n.iloc[-1, 2] == 3:
        
        for i in range(21):
            pa = df_n.iloc[-1, 0] - 0.5 + 0.1*i
            PAC.append(pa)
        PAC = [float(i) for i in PAC]
    
        for i in range(21):
            fl = df_n.iloc[-1, 1] - 0.05 + 0.01*i
            Flomin.append(fl)
        Flomin = [float(i) for i in Flomin]
    
        
        f600_ = parameter(0.15, 0.4, 2.5, 4, 38, 50, 45, 61, df_n.iloc[-1, 0], df_n.iloc[-1, 1])

        delf = df_n.iloc[-1, 3] - f600_
        
        f600 = df_n.iloc[-1, 3]
        f3 = df_n.iloc[-1, 4]
        Gel_ = parameter(0.15, 0.4, 2.5, 4, 52, 31, 54, 32, df_n.iloc[-1, 0], df_n.iloc[-1, 1])
        
        delg = df_n.iloc[-1, 5] - Gel_
        
        Gel = df_n.iloc[-1, 5] 
        Tic = df_n.iloc[-1, 6] 
        Con = df_n.iloc[-1, 2]
    elif df_n.iloc[-1, 2] == 2:
        
        for i in range(21):
            PAC.append(df_n.iloc[-1, 0] - 1 + 0.1*i)
        
        for i in range(21):
            Flomin.append(df_n.iloc[-1, 1] - 0.1 + 0.01*i)  
        f600_ = parameter(0.15, 0.4, 2.5, 4, 20, 31, 26, 36, df_n.iloc[-1, 0], df_n.iloc[-1, 1])
        delf = df_n.iloc[-1, 3] - f600_
        f600 = df_n.iloc[-1, 3]
        f3 = df_n.iloc[-1, 4]
        Gel_ = parameter(0.15, 0.4, 2.5, 4, 25, 8, 26, 10, df_n.iloc[-1, 0], df_n.iloc[-1, 1])
        delg = df_n.iloc[-1, 5] - Gel_
        Gel = df_n.iloc[-1, 5] 
        Tic = df_n.iloc[-1, 6] 
        Con = df_n.iloc[-1, 2]     
    
    PAC_ = []
    Flomin_ = []
    Conc_ = []
    f600_1 = []
    f3_ = []
    Gel_1 = []
    Tic_ = []
    
    if df_n.iloc[-1, 2] == 3:
        for i in PAC:
            for j in Flomin:
                PAC_.append(i)
                Flomin_.append(j)
                Conc_.append(Con)
                f600_1.append(parameter(0.15, 0.4, 2.5, 4, 38, 50, 45, 61, i, j) + delf)
                f3_.append(f3)
                Gel_1.append(parameter(0.15, 0.4, 2.5, 4, 52, 31, 54, 32, i, j) + delg)
                Tic_.append(Tic)
        Conc_ = [float(i) for i in Conc_]
        f600_1 = [float(i) for i in f600_1]
        Gel_1 = [float(i) for i in Gel_1]
        f3_ = [float(i) for i in f3_]
        Tic_ = [float(i) for i in Tic_]
        
    elif df_n.iloc[-1, 2] == 2:
        for i in PAC:
            for j in Flomin:
                PAC_.append(i)
                Flomin_.append(j)
                Conc_.append(Con)
                f600_1.append(parameter(0.15, 0.4, 2.5, 4, 20, 31, 26, 36, i, j) + delf)
                f3_.append(f3)
                Gel_1.append(parameter(0.15, 0.4, 2.5, 4, 25, 8, 26, 10, i, j) + delg)
                Tic_.append(Tic)

    df_t = pd.DataFrame({'PAC':PAC_, 'Flomin':Flomin_, 'Conc':Conc_, 'f600':f600_1, 'f3':f3_, 'Gel10':Gel_1, 'Ticsotrop':Tic_})
    
    if n:
        df_t_scaled = scaler.transform(df_t[g])
        df_t_ = pd.DataFrame(df_t_scaled, columns=g)
        df_t_ = torch.FloatTensor(df_t_.values)
        Acc_train, roc_train, Acc_test, pred, test, n, roc_test, df_t_new = d(4, 78, 1, 0.3, 0.89, df_t_)
        y_t_pred = df_t_new.flatten().tolist()                              
        y_t = []
##        if a > 0:
        
        for i in y_t_pred:
            if i >= a:
                y_t.append(1)
            else:
                y_t.append(0)
##        else:
##            for i in y_t_pred:
##                if i >= 0.5:
##                    y_t.append(1)
##                else:
##                    y_t.append(0)
##    
        df_t['Class'] = y_t
        df_t['prob'] = y_t_pred

    else:    
        df_t_ = pd.DataFrame(
        data_preprocessor.transform(df_t),
        columns=data_preprocessor.get_feature_names_out()    
        )
    
    
        y_t_pred = model.predict_proba(df_t_)[:,1]
        if a != 0.5:
            y_t = []
            for i in y_t_pred:
                if i >= a:
                    y_t.append(1)
                else:
                    y_t.append(0)
        else:
            y_t = model.predict(df_t_) 
        df_t['Class'] = y_t
        df_t['prob'] = y_t_pred

     
    
    df_t.loc[(df_t['Conc'] == 3) & ((df_t['Gel10'] >= 58) | (df_t['f600'] >= 58) | (df_t['f600'] <= 45) | (df_t['Gel10'] <= 34)), 'Class'] = 0
    df_t.loc[(df_t['Conc'] == 2) & ((df_t['f600'] >= 35) | (df_t['Gel10'] >= 23) | (df_t['f600'] <= 25) | (df_t['Gel10'] < 13)), 'Class'] = 0
    
    df_t_1 = df_t.query('Class == 1')
    print(df_t_1[:5])
    
    df_t_1.reset_index(drop=True, inplace=True)
   
    df_t_1 = df_t_1.sort_values(by = 'prob', ascending = False)
    ind_med = len(df_t_1['prob'])//2

    

    PAC_R = df_t_1['PAC'].mean()
    Flomin_R = df_t_1['Flomin'].mean()
    Prob_R = df_t_1['prob'].mean()
    f600_R = df_t_1['f600'].mean()
    Gel_R = df_t_1['Gel10'].mean()

    PAC_M = df_t_1.iloc[0, 0]
    Flomin_M = df_t_1.iloc[0, 1]
    Prob_M = df_t_1.iloc[0, -1]
    f600_M = df_t_1.iloc[0, 3]
    Gel_M = df_t_1.iloc[0, 5]

   # st.write(round(PAC_R,1))
   # st.write(round(Flomin_R,2))
    if df_n.iloc[-1, -2] == 0:
        if n:
            st.warning(f'Рекомендуемая NN концентрация PAC: {round(PAC_R,1)}')
            st.warning(f'Рекомендуемая NN концентрация Flomin: {round(Flomin_R,2)}')
        else:
            st.warning(f'Рекомендуемая ML концентрация PAC: {round(PAC_R,1)}')
            st.warning(f'Рекомендуемая ML концентрация Flomin: {round(Flomin_R,2)}')
    else:
        if n:
            st.warning(f'Оптимальная по NN концентрация PAC: {round(PAC_R,1)}')
            st.warning(f'Оптимальная по NN концентрация Flomin: {round(Flomin_R,2)}')
        else:
            st.warning(f'Оптимальная по ML концентрация PAC: {round(PAC_R,1)}')
            st.warning(f'Оптимальная по ML концентрация Flomin: {round(Flomin_R,2)}')

                     
    
    st.warning(f'Прогноз вероятности класса: {round(Prob_R,2)}')
    st.warning(f'Прогноз величины f600: {round(f600_R,2)}')
    st.warning(f'Прогноз величины GEL10: {round(Gel_R,2)}')
##
###dff()
##
##
##
##
