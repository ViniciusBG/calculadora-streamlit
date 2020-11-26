import numpy as np
import streamlit as st
import pickle

model = pickle.load(open('regressor.pkl', 'rb'))

zona_to_onehot = {'norte': np.array([1, 0, 0]),
                  'oeste': np.array([0, 1, 0]),
                  'sul'  : np.array([0, 0, 1]),
                  'leste': np.array([0, 0, 0])}

def preparing(zona,quartos,area):
    zona_prep = zona_to_onehot[zona.lower()]
    quartos_prep = np.log1p(int(quartos))
    area_prep = np.log1p(int(area))
    features = np.r_[zona_prep, quartos_prep, area_prep].reshape(1,-1)
    return features


def main():
    st.title('Calculadora de Imoveis SP')
    zona = st.selectbox('Zona', ('norte', 'sul', 'leste', 'oeste'))
    quartos = st.selectbox('Número de quartos', ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    area = st.text_input('Área total')
    pred = st.button('Predict')
    if pred:
        features = preparing(zona=zona,quartos=quartos,area=area)
        prediction = np.expm1(model.predict(features))
        output = round(prediction[0], 2)
        st.success(f'O valor do aluguel é R${output}')

if __name__=='__main__':
    main()
