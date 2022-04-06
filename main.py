import numpy as np 
import pickle
import pandas as pd
import streamlit as st 



pickle_dtree=open("crop_pred_dtree.pkl","rb")
#pickle_ran=open("crop_pred_rand.pkl","rb")
pickle_knn=open("crop_pred_knn.pkl","rb")
pickle_svm=open("crop_pred_svc.pkl","rb")

classifier=pickle.load(pickle_dtree)
#classifier1=pickle.load(pickle_ran)
classifier2=pickle.load(pickle_knn)
classifier3=pickle.load(pickle_svm)




def main():
    st.title("Crop Prediction") 
    html_temp = """
    <div style="background-color:teal; padding:10px;">
    <h2 style="color:white; text-align:center;">Classification Report ML App</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    activities=['Decision Tree Classification','Random Forest Classification','K-Nearest Neighbor(KNN)','Support Vector Machine(SVM)']
    option=st.sidebar.selectbox('which model would you like to use ?',activities)
    st.subheader(option)
    temperature=st.text_input('Input your Temparature here:','Type Here')
    humidity=st.text_input('Input your Humidity here:','Type Here')
    ph=st.text_input('Input your Ph here:','Type Here')
    inputs=[[temperature,humidity,ph]]

    result=""


    if st.button('Predict'):
        if option == 'Random Forest Classification':
            result=classifier1.predict(inputs)[0]
            st.success('Crop Prediction Result : {}'.format(result))
        elif option == 'K-Nearest Neighbor(KNN)':
            result=classifier2.predict(inputs)[0]
            st.success('Crop Prediction Result : {}'.format(result))
        elif option == 'Support Vector Machine(SVM)':
            result=classifier3.predict(inputs)[0]
            st.success('Crop Prediction Result : {}'.format(result))
        else:
            result=classifier.predict(inputs)[0] 
            st.success('Crop Prediction Result : {}'.format(result))



        # result=predict_note_authentication(temperature,humidity,ph)
        # st.success('the output is {}'.format(result))

    if st.button('about'):
        st.text("This work is to construct a model for testing the soil fertility \nIt also suggests the crop which has to be planted depending \nupon the value obtained from the sensor. It also provides the regional \nwise information about the crop in the form of graph. \nWe have farmer chat where the farmers can share and get idea \nfrom the expert by registering in this application. \nIt also suggests the fertilizer which has to be added to the soil in \norder to increase the crop productivity. It helps the farmer to analyze the fertility of their yard and plant the better \ncrop to increase their productivity and profit. It also provides the information about the fertilizer to be \nadded in the soil and also provide the information about the nearby fertilizer shop")
        st.text("Built by @Chaitanya Kumar@")
        st.text("Email: chaitu.emandi@gmail.com")

@st.cache
def load_data(nrows):
    data=pd.read_csv('cpdata.csv',nrows=nrows)
    return data


data_list=load_data(1000)

st.subheader('CROP DATA')
st.write(data_list)

st.bar_chart(data_list['labels'])

df=pd.DataFrame(data_list[:200],columns=['temperature','humidity','ph'])
# st.hist(df)

st.line_chart(df)

chart_data=pd.DataFrame(data_list[:40],columns=['ph','rainfall'])
st.area_chart(chart_data)



if __name__ == '__main__':
    main()
