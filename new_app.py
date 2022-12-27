import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

pickled_model=pickle.load(open('C:\Users\RAM JAS\Downloads\new_sort_bbc\news sorting\model_nlp.pk', 'rb'))  

class Switch:
	def new(self, news):

		default = "Incorrect day"

		return getattr(self, 'case_' + str(news), lambda: default)()

	def case_0(self):
		return "business"

	def case_1(self):
		return "tech"

	def case_2(self):
		return "politics"

	def case_3(self):
		return "sport"

	def case_4(self):
		return "entertainment"

# {0: 'business', 1: 'tech', 2: 'politics', 3: 'sport', 4: 'entertainment'}

my_switch = Switch()


def prediction_result(input):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,ngram_range=(1, 2),stop_words='english')
    result =pickled_model.predict(tfidf.fit_transform(input))
    result_=my_switch.new(result)
    return result_

# my_switch.new(1)

def main():
    st.title('News Category Prediction App')
    st.image("""https://civildigital.com/wp-content/uploads/2016/07/Hydraulic-Compression-Testing-Machine.jpg""")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">News Category Prediction</h2>
    </div>
    """
    st.header('Enter the Text here to Verify the Category') 

    text_input = st.text_input('Text Input Here', 'Text only')
    
    st.write('The current movie title is', text_input)

    # text_input = st.text_input(
    #     "Enter some text 👇",
    # )

    # if text_input:
    #     st.write("You entered: ", text_input)

    # result=""

    if st.button("Prediction"):
        result_final=prediction_result(text_input)
    st.success('The output is {}'.format(result_final))

    if st.button("About"):
        st.text("Made with Love & Streamlit")
        st.text("by RAM_JAS MAURYA")
        st.text("for ineoron.ai project")

if __name__ == "__main__":
    main()