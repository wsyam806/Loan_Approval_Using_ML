import streamlit as st
import pandas as pd 
import joblib
import numpy as np
from sklearn.cluster import KMeans
import time

page_bg_img = """
<style>[data-testid="stAppViewContainer"]{
background-color: #d1d1e6;
opacity: 0.2;
background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #d1d1e6 10px ), repeating-linear-gradient( #444cf755, #444cf7 );
}

[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
</style>


"""

Home, Eda, App, Conclusion = st.tabs(['Home', 'Eda', 'App', 'Conclusion'])

with Conclusion:
  st.title("Conclusion")
  st.write("Yes, the result of implementing the loan prediction model can indeed solve the mentioned problems effectively. The loan prediction model offers various advantages to the lending institution and its customers, addressing key challenges in the loan approval process. Here's how the model can solve the problems:")
  st.write("1. Enhanced Loan Approval Process: The loan prediction model automates and optimizes the loan approval process by accurately predicting loan approval or rejection. This reduces the need for manual reviews, streamlines the process, and saves time and resources.")
  st.write("2. Risk Mitigation and Improved Decision Making: The model's segmentation of approved loans into low, moderate, and high-risk categories enables better assessment of creditworthiness. By identifying high-risk borrowers, the institution can take appropriate risk mitigation measures to reduce defaults and potential losses.")
  st.write("3. Customer-Centric Approach: The model provides faster and more reliable loan decisions to customers, creating a better customer experience. Faster loan approvals and access to funds benefit customers, while low-risk borrowers may enjoy more favorable loan terms, incentivizing timely repayments.")
  st.write("4. Improved Portfolio Management: Accurate risk segmentation and loan approval predictions support better portfolio management. The institution can maintain a balanced distribution of risk levels, optimizing the overall portfolio risk and potentially increasing profitability.")
  st.write("5. Compliance and Regulatory Alignment: The model ensures that lending decisions align with regulatory guidelines and compliance requirements. By incorporating fairness and transparency, the institution reduces bias and ensures adherence to industry standards.")
  st.write("6. Competitive Advantage: The advanced loan prediction model gives the lending institution a competitive edge. Quicker loan approvals, reduced risk exposure, and improved customer satisfaction can attract more borrowers and enhance the institution's reputation in the market.")
  st.write("In conclusion, implementing the loan prediction model brings significant benefits to the lending institution and its customers. It streamlines operations, enhances risk assessment, improves decision-making, and ensures compliance, ultimately providing a more efficient and customer-friendly loan approval process. The model's positive impact can position the lending institution as a competitive player in the market, attracting more borrowers and strengthening its position in the industry.")
  st.subheader("**Possible Reasons for Unmatched Predictions:**")
  st.write("- Data Quality Issues: Unmatched predictions could be attributed to data quality problems, such as missing or incorrect values, outliers, or inconsistencies in the dataset. Ensuring data cleanliness and accuracy is crucial for model performance.")
  st.write("- Complex Cases: Some loan applications might involve complex or unusual situations that the model has not encountered during training. The model might struggle to generalize to such cases, leading to unmatched predictions.")
  st.write("- Imbalanced Data: While the overall dataset might be balanced, certain subgroups or specific classes within the data might be imbalanced. The model may face challenges in accurately predicting these imbalanced classes.")
  st.write("- Model Complexity: The RFC model, while powerful, could potentially be overfitting the training data, leading to difficulties in generalizing to unseen instances.")
  st.subheader("**Steps for Improvement:**")
  st.write("- Data Analysis and Preprocessing: Investigate the unmatched predictions to identify patterns and potential data quality issues. Address missing values, outliers, and inconsistencies in the data. Conduct thorough exploratory data analysis to understand the distribution of features in the unmatched instances.")
  st.write("- Feature Engineering: Evaluate the existing features and consider creating new features that might better represent complex relationships in the data. Feature engineering could provide the model with additional information to improve predictions.")
  st.write("- Model Evaluation and Tuning: Reevaluate the model's hyperparameters and consider alternative approaches for hyperparameter tuning. Techniques like GridSearch or RandomizedSearch can help optimize the model's performance.")
  st.write("- Ensemble Methods: Consider using ensemble methods, such as Stacking or Boosting, to combine predictions from multiple models and improve overall accuracy and robustness.")
  st.write("- Handling Imbalanced Data: If certain classes are imbalanced, implement strategies like oversampling, undersampling, or using class weights to balance the dataset during training.")
  st.write("- Regularization: Apply regularization techniques to prevent overfitting and improve the model's ability to generalize to unseen data.")
  st.subheader("**Model Conclusion**")
  st.write("The presence of 55 unmatched predictions in the RFC model indicates the need for further analysis and improvement. By addressing data quality issues, optimizing hyperparameters, considering feature engineering, and exploring ensemble methods, the model's performance can be enhanced. Regular evaluation and monitoring will ensure that the model remains effective in predicting loan approvals and supports the lending institution's decision-making process.")  

with Home:
  st.title("Milestone 2 - Create Model Loan Prediction")
  st.subheader("Problem Statement")
  st.write('Develop a robust machine learning model for loan prediction that accurately classifies loan applications as either approved or rejected, while also segmenting the approved loans into three risk categories: low risk, moderate risk, and high risk. The model should leverage historical loan data, applicant information, and credit analysis to make informed decisions, enabling the lending institution to streamline the loan approval process and mitigate potential credit risks effectively.')
  st.image('loan.jpg')
  bisnis = st.container()
  with bisnis:
    st.subheader("Business Implcation")
    st.write("**1. Enhanced Loan Approval Process:**")
    st.write("Enhancing the loan approval process involves using advanced predictive models to automate and optimize the decision-making process for loan applications. By implementing a loan prediction model, lending institutions can accurately assess the creditworthiness of applicants, predict loan approval outcomes, and expedite the approval process. This results in reduced manual reviews and faster loan decisions, improving operational efficiency and customer experience. The importance of an enhanced loan approval process lies in its ability to streamline operations, save time and resources, and increase the institution's capacity to handle a higher volume of loan applications effectively.")
    st.write("**2. Risk Mitigation and Improved Decision Making:**")
    st.write("Risk mitigation is a critical aspect of lending institutions' operations. The use of predictive models, such as credit risk assessment models, enables lenders to identify high-risk borrowers accurately. By assessing credit risk more effectively, lending institutions can make informed decisions regarding loan approvals, interest rates, and loan terms. Improved decision-making leads to a reduced risk of default and potential losses, enhancing the overall financial stability of the institution. The importance of risk mitigation lies in safeguarding the institution's financial health and ensuring responsible lending practices.")
    st.write("**3. Customer-Centric Approach:**")
    st.write("A customer-centric approach emphasizes meeting the needs and preferences of borrowers. Implementing a loan prediction model can lead to faster loan approvals and more personalized loan terms based on borrowers' credit profiles. Low-risk borrowers may receive more favorable interest rates, encouraging responsible borrowing behavior. The importance of a customer-centric approach is twofold: it enhances customer satisfaction and loyalty while also reducing the institution's exposure to high-risk borrowers, resulting in improved loan portfolio quality.")
    st.write("**4. Improved Portfolio Management:**")
    st.write("Effective portfolio management is crucial for lending institutions to maintain a balanced and diversified loan portfolio. Loan prediction models aid in categorizing loans into different risk segments (low, moderate, high), allowing institutions to optimize their portfolios by strategically allocating resources to various risk categories. A well-managed portfolio reduces the risk of concentration in high-risk loans and increases the potential for profitable returns. The importance of improved portfolio management lies in ensuring long-term financial stability and maximizing returns on investment.")
    st.write("**5. Compliance and Regulatory Alignment:**")
    st.write("Lending institutions operate within a complex regulatory landscape. Predictive models can be designed to incorporate fairness and transparency, aligning lending decisions with regulatory guidelines. This helps prevent discriminatory practices, ensures compliance with fair lending laws, and enhances the institution's reputation. The importance of compliance and regulatory alignment is to protect the institution from legal risks, maintain trust with customers and regulators, and uphold ethical lending standards.")
    st.write("**6. Competitive Advantage:**")
    st.write("Implementing advanced predictive models for loan approvals can provide lending institutions with a competitive advantage in the market. Faster loan processing, accurate risk assessment, and personalized loan offerings attract more borrowers and improve the institution's market positioning. A competitive advantage enables the institution to capture a larger market share, achieve higher customer retention rates, and ultimately drive business growth and profitability.attract more borrowers and enhance the institution's reputation.")

with Eda:
  st.title("Exploratory Data Analysis")
  dataset = st.container()
  analysis = st.container()
  conclusion = st.container()
  
  with dataset:
    st.subheader("Dataset")
    st.text("https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset")
    df = pd.read_csv('loan_approval_dataset.csv')
    st.dataframe(df)
    st.write('Default Curreny : INR')
  with analysis:  
    st.subheader("Data Overview")
    st.image('data overview.PNG')
    st.subheader("Varibles")
    st.image('variables 1.PNG')
    st.image('variables 2.PNG')
    st.image('variables 3.PNG')
    st.image('variables 4.PNG')
    st.image('variables 5.PNG')
    st.image('variables 6.PNG')
    st.image('variables 7.PNG')
    st.image('variables 8.PNG')
    st.image('variables 9.PNG')
    st.image('variables 10.PNG')
    st.image('variables 11.PNG')
    st.image('variables 12.PNG')
    st.image('variables 13.PNG')
    st.subheader("Interacations")
    st.image('intercation.PNG')
    st.subheader("Correlations")
    st.image('correlation.PNG')
    st.subheader("Missing Value")
    st.image('correlation.PNG')
  with conclusion:
    st.subheader("Conclusion")
    st.image('conclusion.PNG')

with App:
  st.subheader("New user model prediction")
  cluster = joblib.load('cluster.pkl')
  model = joblib.load('all_process.pkl')  
  
  df = pd.read_csv('loan_approval_dataset.csv')
  num_col = [' income_annum', ' loan_amount', ' loan_term', ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value']
  cluster_data = df.drop(columns=[col for col in df.columns if col not in num_col])
  cluster_df = cluster.fit_transform(cluster_data)
  k_3 = KMeans(n_clusters=3)
  label3 = k_3.fit_transform(cluster_df)
  label = pd.DataFrame(label3)
  df = pd.concat([label.reset_index(drop=True), df], axis=1)
  df.rename(columns={0: "cluster"}, inplace=True)
    
  income_annum = st.slider('income_annum', 0,9999999)
  loan_amount = st.slider('loan_amount', 0,9999999)
  loan_term = st.slider('loan_term', 2, 24)
  cibil_score = st.slider('cibil_score', 0,900)
  residential_assets_value = st.slider('residential_assets_value', 0, 9999999)
  commercial_assets_value = st.slider('commercial_assets_value',0,9999999)
  luxury_assets_value = st.slider('luxury_assets_value',0,9999999)
  bank_asset_value = st.slider('bank_asset_value',0,9999999)
  education = st.selectbox('education',[' Not Graduate', ' Graduate'])
  data = {
    ' income_annum': income_annum,
    ' loan_amount': loan_amount,
    ' loan_term': loan_term,
    ' cibil_score': cibil_score,
    ' residential_assets_value': residential_assets_value,
    ' commercial_assets_value': commercial_assets_value,
    ' luxury_assets_value': luxury_assets_value,
    ' bank_asset_value': bank_asset_value,
    ' education': education,
  }
  input = pd.DataFrame(data, index=[0])
  st.subheader('User Input')
  st.write(input)
  if st.button('Predict'):
    progress_bar = st.progress(0)
    for perc_completed in range(100):
        time.sleep(0.05)
        progress_bar.progress(perc_completed+1)
    num_col = [' income_annum', ' loan_amount', ' loan_term', ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value']
    def_cluster = input.drop(columns=[col for col in input.columns if col not in num_col])
    def_pca = cluster.transform(def_cluster)
    labelz = k_3.predict(def_pca)
    cluster_names = {
    0: 'Moderate-Risk Customers',
    1: 'Low-Risk Customers',
    2: 'High-Risk Customers'
    }
    cluster_labels = [cluster_names[label] for label in labelz]
    input['cluster'] = cluster_labels
    
    if np.all(labelz == 0):
        st.write('Based on user input, the placement cluster: Moderate-Risk Customers')
    elif np.all(labelz == 1):
        st.write('Based on user input, the placement cluster: Low-Risk Customers')
    elif np.all(labelz == 2):
        st.write('Based on user input, the placement cluster: High-Risk Customers')
    
    prediction = model.predict(input)
    
    if prediction == 0:
        prediction = ' Rejected'
    else:
        prediction = ' Approved'

    st.write('The model approval status: ',prediction)
   
  