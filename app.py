import numpy as np
import streamlit as st
import pickle

# LOADS THE MODEL
model = pickle.load(open("regressor.pkl", "rb"))

# one hot encodes the zones from São Paulo city
zona_to_onehot = {
    "north": np.array([1, 0, 0]),
    "west": np.array([0, 1, 0]),
    "south": np.array([0, 0, 1]),
    "east": np.array([0, 0, 0]),
}


def preparing(zona, quartos, area):
    """
    Transforms the inputs collected by the interaction of the users into a
    numpy array to feed the model

    Args:
        zona (str): Zone on the city of São Paulo
        quartos (str): Number of bedrooms
        area (str): Total area (square meters)

    Returns:
        np.array : array with the moedel's input variables in right type and
        order
    """
    zona_prep = zona_to_onehot[zona.lower()]
    quartos_prep = np.log1p(int(quartos))
    area_prep = np.log1p(int(area))

    features = np.r_[zona_prep, quartos_prep, area_prep].reshape(1, -1)

    return features


def main():
    """Creates the main function that basically displays the buttons and boxes
    and calls the predictions after getting the inputs from the users.
    """

    st.title("São Paulo Rent Calculator")
    # collecting the data
    zona = st.selectbox("Zone", ("north", "south", "east", "west"))
    quartos = st.selectbox(
        "Number of bedrooms", ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
    )
    area = st.text_input("Total Area")
    pred = st.button("Predict")

    if pred:  # if someone cliks predict
        # prepares the features and returns the predictions
        features = preparing(zona=zona, quartos=quartos, area=area)
        prediction = np.expm1(model.predict(features))
        output = round(prediction[0], 2)

        st.success(f"The rent value is R${output}")


if __name__ == "__main__":
    main()
