

# ### Import the appropriate libraries you need to solve the questions.

# In[ ]:


import pandas as pd
from collections import Counter
import json


# ### Please assign the variables `first_name`, `last_name`, `student_id`, and `email` with your first name, last name, student ID, and email address.

# In[ ]:


first_name = str("Khushboo")
last_name = str("Dodani")
student_id = int("71907257")
email = str("khushdodani@gmail.com")


# ## Download and preprocess the data
# 
# - Download `corpus_10k_2015-2019.csv` file into the same directory, where `hw2_starter.ipynb` is located. (If not, there will be an extra deduction on your grade)
# 
# - First, create user-defined `isYear` function with two parameters (`target_year`, `text`) which check the `year`column value is the same as `target_year` in the `text`.
# - Second, open `corpus_10k_2015-2019.csv` file with `open` function and filter the data which the `year` is `2019` from <b>the first 10,000 companies</b> using `isYear` function you defined.
# - Save the filtered data as a `txt` file called `corpus.subset.txt`.
# 
# [Hint]
# - `open` function : https://www.w3schools.com/python/ref_func_open.asp
# 

# In[ ]:


def isYear(target_year, text):
    columns = text.split(",")
    year = columns[4]
    if year == "year":
        return True
    else:
        return int(year) == target_year

limit = 10001 # no need to setup
filtered = []
with open("corpus_10k_2015-2019.csv","r") as infile:
    for line in infile:
        if len(filtered)<limit:
            if isYear(2019,line):
                filtered.append(line)
        else:
            break

with open("corpus.subset.txt","w") as outfile:
    for line in filtered:
        outfile.write("{}\n".format(line))


# - Create dataframe `df` from `corpus.subset.txt` that you made right before, and view the first 5 rows using `head` method.

# In[ ]:


df = pd.read_csv("corpus.subset.txt",sep=",")
df.head()


# - Drop <b>the columns</b> where <b>all elements are NaN</b> (in other words, this column contains no useful information)
#  using `dropna` method from `pandas`.
#  
# 
# [Hint]
# - `dropna` method : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html

# In[ ]:


df = df.dropna(axis="columns", how="all")
df.shape


# - Fill <b>the missing values</b> with <b>empty string ("")</b> using `fillna` method from `pandas`.
# - Then, view the first 5 rows to confirm that missing values have been replaced using `head` method. 
# 
# [Hint]
# - `fillna` method : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html

# In[ ]:


df = df.fillna("")
df.head()


# ##  Scrape SIC code and names on the web using BeautifulSoup 
# - Collect the industry names for sic codes from the <b>"List"</b> section of the Wikipedia page ("https://en.wikipedia.org/wiki/Standard_Industrial_Classification").
# - If the above link is not directly linked with the Wikipedia page, please copy and paste the URL on the new tab.
# - Create `code_to_industry_name` dictionary where the `key` is the sic code and the `value` is the industry name.
# - Then, replace the SIC code "0100 (01111...)" from the table with 0100.
# 

# In[ ]:


from bs4 import BeautifulSoup
from requests import get
url = "https://en.wikipedia.org/wiki/Standard_Industrial_Classification"
response = get(url)

html_soup = BeautifulSoup(response.text, "html.parser")

# the SIC code table with industry names is the second table of the page
table = html_soup.findChildren("table")[1]
rows = table.findChildren("tr")

code_to_industry_name = dict()

def getStringFromCell(row, index):
    return row[index].get_text().strip()

# skip header row
for row in rows[1:]:
    row_columns = row.findChildren("td")
    sic_code = getStringFromCell(row_columns,0)
    industry_name = getStringFromCell(row_columns,1)
    sic_code = int(sic_code) if sic_code != "0100 (01111...)" else 100
    code_to_industry_name[sic_code] = industry_name

code_to_industry_name


# - Add a new column `industry_name` to `df` using `lambda` function.
# - Values in `industry_name` must correspond to the `sic` in the `df`.
# - For example, if a row has a SIC code of `1000`, then value of its industry name will be `Forestry`.
# 
# [Hint]
# - `lamda` : https://www.w3schools.com/python/python_lambda.asp

# In[ ]:


df["industry_name"] = df["sic"].map(lambda x: code_to_industry_name[x])
df.head()


# ## Now, you get the preprocess dataframe `df` to analyze.
# 
# ## Industry analysis (Q1-Q4)
# ### Question1. What are the 5 most common industries? Get them from `industry_name`, not from `sic` code
# - Store a list of 5 most common industry names in `ans1`.

# In[ ]:


ans1 = Counter(df["industry_name"]).most_common(5)
ans1 = [ele[0] for ele in ans1]
print(ans1)


# ### Question2. Out of all the industries with the prefix `Services`, what are the 4 most common?
# - Store a list of 4 most common industry names in `ans2`.

# In[ ]:


ans2 = Counter(df[df["industry_name"].str.contains("Services")]["industry_name"]).most_common(4)
ans2 = [ele[0] for ele in ans2]
print(ans2)


# ### Question3. What is the `name` of the company `id` with `1353611-2019`?
# - Store the company name as a string in `ans3`.

# In[ ]:


ans3 = df[df["id"] == "1353611-2019"]["name"].values
ans3 = str(ans3[0])
print(ans3)


# ### Question4. What is the `industry_name` of the company with name `Solar Quartz Technologies Corp`?
# - Store the industry name as a string in `ans4`.

# In[ ]:


ans4 = df[df["name"] == "Solar Quartz Technologies Corp"]["industry_name"].values
ans4 = str(ans4[0])
print(ans4)


# ## Keyword analysis (Q5 and Q6)
# 
# ### For Q5 and Q6 you will filter out stopwords and non-alphanumeric English characters. 
# - You can use `nltk.corpus.stopwords` for our definition of stopwords. 
# - Alphanumeric English characters are letters in the alphabet (a-z) and numbers (0-9).
# - For example, <b>"Python is awesome :scream_cat:"</b> would be filtered to <b>"Python awesome"</b> after removing stopwords (in this case "is") and the emoji (non-alphanumeric).
# 
# [Hint]
# - `nltk.corpus` for stopwords : https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# 

# In[ ]:


# if you are using nltk.word_tokenize, you can use the code below.
# if you are using split, you can use the commented code.

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def tokenize_counter(values):
    result = Counter()
    for text in values:
        result = result + Counter([w for w in nltk.word_tokenize(text.lower()) if w.isalnum()])
        #result = result + Counter([w for w in text.lower().split() if w.isalnum()])
    return result

def tokenize_filter_stopwords_counter(values):
    result = Counter()
    for text in values:
        tokens = [w for w in nltk.word_tokenize(text.lower()) if w.isalnum()]
        #tokens = [w for w in text.lower().split() if w.isalnum()]
        filtered_tokens = [t for t in tokens if t not in stopwords.words("english")]
        result = result + Counter(filtered_tokens)
    return result


# ### Question5. What are the 5 most common words from the `Item_5` column?
# - Store a list of the 5 most common words in `ans5`.

# In[ ]:


df["item_5"] = df["item_5"].astype(str)
ans5 = tokenize_counter(df["item_5"]).most_common(5)
ans5 = [ele[0] for ele in ans5]
print(ans5)


# ### Question6. What are the 5 most common words from the `Item_5` column without stopwords?
# - Store a list of the 5 most common words in `ans6`.

# In[ ]:


ans6 = tokenize_filter_stopwords_counter(df["item_5"]).most_common(5)
ans6 = [ele[0] for ele in ans6]
print(ans6)


# ## Named Entity Recognition (Q7-Q9)
# 
# - If any of the entities are spaces, exclude them when considering the most common.
# - For example, `(" ")` is not a valid entity.

# In[ ]:



import spacy

nlp = spacy.load("en")

def ner_counter_label(values, label):
    result = Counter()
    for text in values:
        #tokens = [t.strip() for t in text.split()]
        tokens = [t.strip() for t in nltk.word_tokenize(text)]
        doc = nlp(" ".join(tokens))
        entities = list(doc.ents)
        result = result + Counter([ent.text for ent in entities if ent.label_ == label])
    return result

def ner_counter(values):
    result = Counter()
    for text in values:
        #tokens = [t.strip() for t in text.split()]
        tokens = [t.strip() for t in nltk.word_tokenize(text)]
        doc = nlp(" ".join(tokens))
        entities = list(doc.ents)
        result = result + Counter([ent.text for ent in entities])
    return result


# ### Question7. What are the 5 most common `PERSON` named entities overall from the `item_1` column?
# - Store a list of the 5 most common `PERSON` named entities in `ans7`.

# In[ ]:


df["item_1"] = df["item_1"].astype(str)
ans7 = ner_counter_label(df["item_1"].loc[:99,], "PERSON").most_common(5)
ans7 = [ele[0] for ele in ans7]
print(ans7)


# ### Question8. What are the 5 most common `ORG` named entities overall from the `item_2` column? 
# - Store a list of the 5 most common `ORG` named entities in `ans8`.
# - ORG: Companies, agencies, institutions, etc.

# In[ ]:


df["item_2"] = df["item_2"].astype(str)
ans8 = ner_counter_label(df["item_2"].loc[:99,], "ORG").most_common(5)
ans8 = [ele[0] for ele in ans8]
print(ans8)


# ### Question9. What are the 4 most common named entities overall from the `item_9` column?
# - Store a list of the 4 most common named entities in `ans9`.

# In[ ]:


df["item_9"] = df["item_9"].astype(str)
ans9 = ner_counter(df["item_9"].loc[:99,]).most_common(4)
ans9 = [ele[0] for ele in ans9]
print(ans9)


# ## NER for specific firm (Q10-Q12)
# - You want to find the information on the company with id `1653710-2019`.
# - Given list comprehension, you want to find out common entities in the dataframe `df`.

# In[ ]:


columns = [col for col in df.columns if 'item' in col]


# ### Question10. what are the 4 most common `PERSON` named entities mentioned by the company with id `1653710-2019` across all `item_* rows`?
# - Store a list of the 4 most common `PERSON` named entities in `ans10`.

# In[ ]:


ans10 = ner_counter_label(df[df["id"]=="1653710-2019"][columns].values[0], "PERSON").most_common(4)
ans10 = [ele[0] for ele in ans10]
print(ans10)


# ### Question11. What are the 2 most common `GPE` named entities mentioned by the company with id `1653710-2019` across all `item_* rows`?
# - Store a list of the 2 most common `GPE` named entities in `ans11`.

# In[ ]:


ans11 = ner_counter_label(df[df["id"]=="1653710-2019"][columns].values[0], "GPE").most_common(2)
ans11 = [ele[0] for ele in ans11]
print(ans11)


# ### Question12. What are the 5 most common named entities mentioned by the company with id `1653710-2019` across all `item_* rows`?
# - Store a list of the 5 most common named entities in `ans12`.

# In[ ]:


ans12 = ner_counter(df[df["id"]=="1653710-2019"][columns].values[0]).most_common(5)
ans12 = [ele[0] for ele in ans12]
print(ans12)


# ## Twitter analysis (Q13-Q15)
# ### `tweets.json` collected  50,000 tweets containing below keywords: 
# - Keyword : `analytics`, `technology`, `big data`, `machine learning`, `artificial intelligence`
# - The way used to collect the Twitter streaming data is using `tweepy` and `twython` module.
# - `tweepy` for Twitter streaming : http://docs.tweepy.org/en/latest/streaming_how_to.html
# 
# ### Save and read the `tweets.json` file as `tweets` 
# - Download `tweets.json` file into the same directory, where `hw2_starter.ipynb` is located. (If not, there will be an extra deduction on your grade)
# - Open `tweets.json` file as `tweets` with `open` function.
# 
# [Hint]
# - `open` function : https://www.w3schools.com/python/ref_func_open.asp
# 

# In[ ]:


with open("tweets.json") as infile:
    tweets = json.load(infile)


# ### Question13. In the collected 50,000 tweets, what are the 100 most common words after removing stop words?
# - Store a list of the 100 most common words in `ans13`.

# In[ ]:


tweet_text = [t["text"] for t in tweets if "text" in t]
tweet_text


# In[ ]:


tweet_text = [t["text"] for t in tweets if "text" in t]

ans13 = tokenize_filter_stopwords_counter(tweet_text).most_common(100)
ans13 = [ele[0] for ele in ans13]

print(ans13)


# ### Question14. Find the firm that has the most common words between `item_1` and 50,000 tweets. 
# - First, find the 100 most common words of each firm's `item_1` column.
# - Then, use the top the top 100 most common words of the 50,000 tweets after removing stop words (`ans13`) to find the most common words between `item_1` and 50,000 tweets. 
# - Disregard the word count, we are only interested in the number of unqiue words that appear in intersection of both common words.
# - Store the answer as a string in `ans14`.

# In[ ]:


company_names = df["name"].loc[:99,].values
company_item_1 = df["item_1"].loc[:99,].values

name_common = dict()
for i in range(len(company_names)):
    text = company_item_1[i]
    result = tokenize_filter_stopwords_counter([text]).most_common(100)
    result = [w[0] for w in result]
    num_common = len(set(result) & set(ans13))
    name_common[company_names[i]] = num_common

ans14 = max(company_names, key=name_common.get)
print(ans14)


# ### Question15. In the collected 50,000 tweets, what are the 5 most common named entities mentioned?
# - You need to use the NER for this question.
# - Store a list of the 5 most common named entities in `ans15`.

# In[ ]:


ans15 = ner_counter(tweet_text).most_common(5)
ans15 = [ele[0] for ele in ans15]
print(ans15)


# ## For the following analyses, find the top two most common industries names
# - Assign the most common industry name as `top_1` and the second most common industry name as `top_2`.

# In[ ]:


industry_count = Counter(df["industry_name"].values)
most_common_industry = industry_count.most_common(2)
top_1 = most_common_industry[0][0]
top_2 = most_common_industry[1][0]
print(top_1)
print(top_2)


# ## Word cloud and sentiment analysis (Q16-Q19)
# 
# - Use `wordcloud` library and `WordCloud` function in it.
# - Define user-defined `generate_wordcloud` function with one parameter `values` to generate word cloud for one input value.
# - You don't need `axis` in the wordcloud and use `bilinear` interpolation. 
# 
# [Hint]
# - `bilinear` for `imshow()` : https://matplotlib.org/3.3.1/gallery/images_contours_and_fields/interpolation_methods.html

# In[ ]:


# word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def generate_wordcloud(values):
    wordcloud_text = " ".join(values)
    wordcloud = WordCloud().generate(wordcloud_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# ### Question16. Make two separate wordclouds for `item_1` column.
# - One for the most common industry and another one for the second most common industry.
# - Save the graph named "`hw2_ans16a_(student_id).png`" and "`hw2_ans16b_(student_id).png`".<br/>
#   (e.g.) <b>hw2_ans16a_37510930.png</b>, <b>hw2_ans16b_37510930.png</b>, respectively.

# In[ ]:


generate_wordcloud(df[df["industry_name"] == top_1]["item_1"].values)
plt.savefig('hw2_ans16a_{}.png'.format(student_id))
generate_wordcloud(df[df["industry_name"] == top_2]["item_1"].values)
plt.savefig('hw2_ans16b_{}.png'.format(student_id))


# ### Question17. Make two separate wordclouds for `item_1a`column.
# - One for the most common industry and another one for the second most common industry.
# - Save the graph named "`hw2_ans17a_(student_id).png`" and "`hw2_ans17b_(student_id).png`".<br/>
#   (e.g.) <b>hw2_ans17a_37510930.png</b>, <b>hw2_ans17b_37510930.png</b>, respectively.

# In[ ]:


generate_wordcloud(df[df["industry_name"] == top_1]["item_1a"].values)
plt.savefig('hw2_ans17a_{}.png'.format(student_id))
generate_wordcloud(df[df["industry_name"] == top_2]["item_1a"].values)
plt.savefig('hw2_ans17b_{}.png'.format(student_id))


# ### Question18. Make two separate wordclouds for `item_7` column
# - One for the most common industry and another one for the second most common industry.
# - Save the graph named "`hw2_ans18a_(student_id).png`" and "`hw2_ans18b_(student_id).png`".<br/>
#   (e.g.) <b>hw2_ans18a_37510930.png</b>, <b>hw2_ans18b_37510930.png</b>, respectively.

# In[ ]:


generate_wordcloud(df[df["industry_name"] == top_1]["item_7a"].values)
plt.savefig('hw2_ans18a_{}.png'.format(student_id))
generate_wordcloud(df[df["industry_name"] == top_2]["item_7a"].values)
plt.savefig('hw2_ans18b_{}.png'.format(student_id))


# ### Question19. Make two histograms of the polarity for `item_1a` column. 
# - One for the most common industry and another one for the second most common industry.
# - Save the graph named "`hw2_ans19a_(student_id).png`" and "`hw2_ans19b_(student_id).png`".<br/>
#   (e.g.) <b>hw2_ans19a_37510930.png</b>, <b>hw2_ans19b_37510930.png</b>, respectively.

# In[ ]:


from textblob import TextBlob

polarity = [TextBlob(txt).sentiment for txt in df[df["industry_name"] == top_1]["item_1a"].values]
plt.hist(polarity)
plt.savefig('hw2_ans19a_{}.png'.format(student_id))

polarity = [TextBlob(txt).sentiment for txt in df[df["industry_name"] == top_2]["item_1a"].values]
plt.hist(polarity)
plt.savefig('hw2_ans19b_{}.png'.format(student_id))


# ### Question 20: Make outfile name format as `hw2_answers_(student_id).txt` and save it to `txt` file                
# - When you write the answer, please keep format(please refer to word doc example).
# - File name should be like this : <b>hw2_answers_37510930.txt</b>

# In[ ]:


outfile = open('hw2_answers_{}.txt'.format(student_id), 'w')
outfile.write('{}, {}, {}\n'.format(last_name, first_name, email))
outfile.write("answer1={}\n".format(ans1))
outfile.write("answer2={}\n".format(ans2))
outfile.write("answer3={}\n".format(ans3))
outfile.write("answer4={}\n".format(ans4))
outfile.write("answer5={}\n".format(ans5))
outfile.write("answer6={}\n".format(ans6))
outfile.write("answer7={}\n".format(ans7))
outfile.write("answer8={}\n".format(ans8))
outfile.write("answer9={}\n".format(ans9))
outfile.write("answer10={}\n".format(ans10))
outfile.write("answer11={}\n".format(ans11))
outfile.write("answer12={}\n".format(ans12))
outfile.write("answer13={}\n".format(ans13))
outfile.write("answer14={}\n".format(ans14))
outfile.write("answer15={}\n".format(ans15))
outfile.write("HW 2 is done!!!\n")
outfile.close()


# #### After finishing `hw2`, please submit this python code file on Canvas!!
# #### But, you don't need to submit the `.png` files. 
# 
# #### Again, the code file name should be as follows: `hw2_(student_id).py` 
# (e.g.) hw2_37510930.py
