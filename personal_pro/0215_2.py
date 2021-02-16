# https://www.kaggle.com/spuchalski/predicting-price-of-airbnb-listings-in-nyc
# 이건 뉴욕안 지역별로 가격/ 이런식으로 특정값 가진 것만 추출하는 거 해보기
# 가장 결측 값이 적고 데이터가 많은 뉴욕을 선택(2019년 자료 선택 - 코로나 이전)

#import modules:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('C:/data/personal/final/AB_NYC_2019.csv')
#randomize the data
data.sample(frac=1)
df=data.copy()
# print(df.shape) # (48895, 16) 
# print(df.head().iloc[:,0:8])
# print(df.head().iloc[:,8:])

# last_review 및 reviews_per_month에 대한 누락된 값은 0
df[["last_review", "reviews_per_month"]] = df[["last_review", "reviews_per_month"]].fillna(0) #수치

# 호스트 이름 없는 것은 nan으로 지정
df[["name", "host_name"]] = df[["name", "host_name"]].fillna("None") #문자

#Drop rows were price of the listing is 0. We are not intersted in "free" 
#listings as they are most likely an error.
#또한 공짜인 price 0 제거(에어비엔비 상업용 돈을 지불하는 관광객을 대상으로 할 것) 
free = len(df[df.price == 0])
df = df[df.price != 0].copy() # x != y	x와 y가 같지 않다(즉, 0인것 외만 선택)

# 위에 결측값 내용 총정리
print("첫 데이터에서는 " + str(free)+ " 개의 가격이 0달러인 리스트를 포함하고 있어 삭제했다")
print("이 데이터는 " + str(len(df["id"].unique()))+" 의 행이 있고") #unique 고유값
print("이 데이터는 "+str(len(df.host_id.unique()))
      +" 개의 고유한 이름이 있는 "+ "호스트가 있다.")
print("이 데이터는 "+str(len(df[df["host_name"]=="None"]))
      +" 개의 이름이 없는 "+ "호스트가 있다.")
print("Dataframe shape: "+str(df.shape))

'''
(len(df[df["host_id"]==30985759]) == df[df["id"]==36485609]["calculated_host_listings_count"]).tolist()
df[(df["calculated_host_listings_count"]>1)][["host_id","calculated_host_listings_count"]].sort_values(by=['host_id']).head(10)
'''

# 히스토그램으로 호스트들의 최소숙박일 minimum_nights
df_old=df.copy()
df = df[df["minimum_nights"] <=90].copy() # 관광 목적이기 때문에 90일 최대 거주일인 제한(총 197행 제거)
removed_listings = len(df_old)-len(df)
'''
fig = plt.figure(figsize=(14,3))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.hist(df_old.minimum_nights, bins=100, log=True) # 31일 제한 전, bins : x축 단위
ax1.set_ylabel("Frequency")
ax1.set_title("No limit on minimum nights")

ax2.hist(df.minimum_nights, bins=90, log=True) # 제한 후
ax2.set_ylabel("Frequency")
ax2.set_title("Maximum 90 minimum nights")

plt.show()
'''
print("As a result of imposing minimum nights limit, " + str(removed_listings)+" listings were removed.")

# print(df.isnull().sum()) # 결측값 있나 확인
# print(df.describe().iloc[:,0:8])
# print(df.describe().iloc[:,8:])
# print(df.dtypes) # 변수 형태를 알아보고 변환해야 하는지 파악

# ============================= 컬럼별 log 대입 전 그래프 ===================================
#separate out numerical variables
a=pd.DataFrame(df.dtypes.copy())
b= a[a[0] != 'object'].reset_index() # reset_index :인덱스를 다시 처음부터 재배열
#drop id and host id:
numeric_vars=b["index"].tolist()[2:] # tolist : array를 list로 변경
# 숫자화 된것중 3번째꺼 부터 출력(id, host_name 버리기)
'''
fig = plt.figure(figsize=(14,14))
ax1 = fig.add_subplot(3, 3, 1) #9개인데 그중 1번째
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)

ax1.hist(df[numeric_vars[0]], bins=90) # latitude
ax1.set_ylabel("Frequency")
ax1.set_title(numeric_vars[0])

ax2.hist(df[numeric_vars[1]], bins=90) # longitude
ax2.set_ylabel("Frequency")
ax2.set_title(numeric_vars[1])

ax3.hist((df[numeric_vars[2]]), bins=90) #price 
ax3.set_ylabel("Frequency")
ax3.set_title('price')

ax4.hist(df[numeric_vars[3]], bins=90) #minimum_nights
ax4.set_ylabel("Frequency")
ax4.set_title(numeric_vars[3])

ax5.hist(df[numeric_vars[4]], bins=90) # number_of_reviews 
ax5.set_ylabel("Frequency")
ax5.set_title("number of reviews")

ax6.hist(df[numeric_vars[5]], bins=90) # last_review
ax6.set_ylabel("Frequency")
ax6.set_title("last review")

ax7.hist(df[numeric_vars[6]], bins=90) #calculated_host_listings_count
ax7.set_ylabel("Frequency")
ax7.set_title(numeric_vars[6])

ax8.hist(df[numeric_vars[7]]) #availability_365
ax8.set_ylabel("Frequency")
ax8.set_title(numeric_vars[7])
plt.show()

print(numeric_vars) # 숫자인것만 출력

# ================ 칼럼별 log 대입 후 그래프 ===========================
fig = plt.figure(figsize=(14,14))
ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)

ax1.hist(df[numeric_vars[0]], bins=30)
ax1.set_ylabel("Frequency")
ax1.set_title(numeric_vars[0])

ax2.hist(df[numeric_vars[1]], bins=30)
ax2.set_ylabel("Frequency")
ax2.set_title(numeric_vars[1])

ax3.hist(np.log((df[numeric_vars[2]])), bins=30)
ax3.set_ylabel("Frequency")
ax3.set_title('log(price)')

ax4.hist(np.log((df[numeric_vars[3]])), bins=31)
ax4.set_ylabel("Frequency")
ax4.set_title("log(minimum nights + 1)")

ax5.hist(np.log(df[numeric_vars[4]]+1), bins=30)
ax5.set_ylabel("Frequency")
ax5.set_title("log(number of reviews + 1)")

ax6.hist(np.log(df[numeric_vars[5]]+1), bins=30)
ax6.set_ylabel("Frequency")
ax6.set_title("log(last review + 1)")

ax7.hist(np.log(df[numeric_vars[6]]+1), bins=30)
ax7.set_ylabel("Frequency")
ax7.set_title("log(calculated host listing count) + 1)")

ax8.hist(np.log(df[numeric_vars[7]]+1), bins=30)
ax8.set_ylabel("Frequency")
ax8.set_title("log(availability 365 + 1)")

plt.show()
'''
for num in numeric_vars[3:]:#경도, 위도 빼고
    df["log_("+num+" +1)"] = np.log(df[num]+1)
df["log_price"] = np.log(df.price)
df=df.drop(columns = numeric_vars[2:]).copy() #id, host_id빼고 카피

print(df.columns.tolist()) # log넣은 것도 숫자화로 바꾸기


numeric_vars = df.columns.tolist()[6:8]+df.columns.tolist()[10:]
print(numeric_vars) # 경도, 위도랑 로그씌운 6개 변수
# ['latitude', 'longitude', 'log_(minimum_nights +1)', 'log_(number_of_reviews +1)', 
#'log_(reviews_per_month +1)', 'log_(calculated_host_listings_count +1)', 'log_(availability_365 +1)', 'log_price']

# =============== 상관관계 ================================
import seaborn as sns
x=df[numeric_vars].apply(lambda x: np.log(np.abs(x+1))).corr(method='pearson')
# sns.heatmap(x, annot=True)
# plt.show()

#separate out numerical variables
a=pd.DataFrame(df.dtypes.copy())
print('a:', a)
b= a[a[0] == 'object'].reset_index() # reset_index :인덱스를 다시 처음부터 재배열
#drop id and host id:
non_num=b["index"].tolist() # tolist : array를 list로 변경
print('non_num:', non_num) #object 인것만 추출
# non_num: ['name', 'host_name', 'neighbourhood_group', 'neighbourhood', 'room_type', 'last_review']

'''
# ============= 지도화해 숙소 나타내기 =======================
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"
y = df.latitude
x = df.longitude
p = df.log_price
plt.figure(figsize=(16,9))

#scaling the image based on the latitude and longitude max and mins for proper output
# plt.scatter(x,y,c=p,cmap='viridis') # viridis: 컬러 맵에 매핑되는 산점도를 생성, gray:흑백
# plt.colorbar()
# plt.xlabel("경도") # 경도
# plt.ylabel("위도") # 위도
# plt.title("숙소 가격 분포도")
i = plt.imread('C:/data/personal/AB_NYC_2019/map.png')
plt.imshow(i,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
ax=plt.gca()
plt.scatter(df['longitude'],df['latitude'],label="숙소 가격 분포도",alpha=0.5, c=p, cmap='viridis')
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
'''

# ===================== 지역별 price ========================
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"
grouped = df.groupby("neighbourhood")
price_grouped = grouped["log_price"]
price = price_grouped.agg([np.mean, np.median, np.std]).sort_values("mean")
# agg(group간 연산할 때쓰임)가격을 평균, 중간값, 최대, 표준편차로 분류
'''
plt.figure(figsize=(14,4))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)

ax1.barh(price.index,price["mean"])
ax1.set_yticklabels([]) # y축에는 모든 눈금표시
ax1.set_ylabel("Neighborhood")
ax1.set_xlabel("Mean Price")
ax1.set_title("지역별 정렬된 평균 가격")
ax1.set_xlim(3,7)

ax2.barh(price.index,price["median"])
ax2.set_yticklabels([])
ax2.set_ylabel("Neighborhood")
ax2.set_xlabel("Median Price")
ax2.set_title("지역당 중간값")
ax2.set_xlim(3,7)

ax3.barh(price.index,price["std"])
ax3.set_yticklabels([])
ax3.set_ylabel("Neighborhood")
ax3.set_xlabel("Standard Deviation of Price")
ax3.set_title("지역별 가격 표준편차")
plt.show()
'''
# ============== One hot encoding ===========================
df = pd.concat([df, pd.get_dummies(df["neighbourhood"], drop_first=False)], axis=1)
#save neighborhoods into a list for further analysis:
neighborhoods = df.neighbourhood.values.tolist()
boroughs = df.neighbourhood_group.unique().tolist()
#drop the neighbourhood column from the database
df.drop(['neighbourhood'],axis=1, inplace=True)

print(df.shape) #(48687, 236)

grouped = df.groupby("room_type")
room_type_price_grouped = grouped["log_price"]
room_type_price = room_type_price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")
print(room_type_price)

# sns.boxplot(x="room_type",y="log_price", data=df)
# plt.show()

# ====================  이상치 제거 =================================
# ======== 지역, 룸타입, 가격을 0.25부터 0.75까지만 사용하는 함수 만들기
def removal_of_outliers(df,room_t, nhood, distance):
    '''Function removes outliers that are above 3rd quartile and below 1st quartile'''
    '''The exact cutoff distance above and below can be adjusted'''

    new_feature = df[(df["room_type"]==room_t)&(df["neighbourhood_group"]==nhood)]["log_price"]
    #defining quartiles and interquartile range
    q1 = new_feature.quantile(0.25)
    q3 = new_feature.quantile(0.75)
    iqr=q3-q1

    trimmed = df[(df.room_type==room_t)&(df["neighbourhood_group"]==nhood) &(df.log_price>(q1-distance*iqr))&(df.log_price<(q3+distance*iqr))]
    return trimmed

#각각 방 별로 나누기
df_private = pd.DataFrame()
for neighborhood in boroughs:
    a = removal_of_outliers(df, "Private room",neighborhood,3)
    df_private = df_private.append(a)

df_shared = pd.DataFrame()
for neighborhood in boroughs:
    a = removal_of_outliers(df, "Shared room",neighborhood,3)
    df_shared = df_shared.append(a)
    
df_apt = pd.DataFrame()
for neighborhood in boroughs:
    a = removal_of_outliers(df, "Entire home/apt",neighborhood,3)
    df_apt = df_apt.append(a)

# Create new dataframe to absorb newly produced data    
df_old=df.copy()    
df = pd.DataFrame()
df = df.append([df_private,df_shared,df_apt])
'''
#plot the results
fig = plt.figure(figsize=(14,4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.hist(df_old.log_price)
ax1.set_xlim(2,7)
ax1.set_ylabel("Frequency")
ax1.set_xlabel("Log Price")
ax1.set_title("이상치 제거 전 방 가격")

ax2.hist(df.log_price)
ax2.set_xlim(2,7)
ax2.set_ylabel("Frequency")
ax2.set_xlabel("Log Price")
ax2.set_title("이상치 제거 후 방 가격")
plt.show()
'''
print("As a result of oulier removal " + str(df_old.shape[0]-df.shape[0]) + " rows of data were removed.")
# 325 개 의 행 제거
print(df.shape) # (48362, 236)

# ================== 룸 타입별 요금 ======================================
grouped = df.groupby("room_type")
room_type_price_grouped = grouped["log_price"]
room_type_price = room_type_price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")
print(room_type_price)

#convert room types to dummies
df = pd.concat([df, pd.get_dummies(df["room_type"], drop_first=False)], axis=1)
df.drop(['room_type'],axis=1, inplace=True)
print(df.shape)

'''
# 이거 빼기
y = df[(df["SoHo"]==1) & (df["Private room"]==1)].latitude
x = df[(df["SoHo"]==1) & (df["Private room"]==1)].longitude
p = df[(df["SoHo"]==1) & (df["Private room"]==1)].log_price
plt.scatter(x,y,c=p,cmap='viridis')
plt.xlim(-74.01,-73.995)
plt.ylim(40.718,40.73)
plt.colorbar()
plt.show()
'''
'''
# 이것도 빼자
# ====================== 리뷰개수 =============================
print(df.shape)

# 이것도 안써
plt.hist(df["last_review"], bins=100)
plt.ylabel("Frequency")
plt.xlabel("Days since last review")
plt.ylabel("Frequency")
plt.title("Histogram of days since last review")
plt.show()

sns.boxplot(x="last_review", y=df.log_price, data=df)
plt.show()
'''
import datetime as dt
#convert object to datetime:
df["last_review"] = pd.to_datetime(df["last_review"])
#Check the latest review date in the datebase:
print(df["last_review"].max())

df["last_review"]=df["last_review"].apply(lambda x: dt.datetime(2019,7,8)-x)
df["last_review"]=df["last_review"].dt.days.astype("int").replace(18085, 1900)

def date_replacement(date):
    if date <=3:
        return "Last_review_last_three_day"
    elif date <= 7:
        return "Last_review_last_week"
    elif date <= 30:
        return "Last_review_last_month"
    elif date <= 183:
        return "Last_review_last_half_year"
    elif date <= 365:
        return "Last_review_last year"
    elif date <= 1825:
        return "Last_review_last_5_years"
    else:
        return "Last_review_never"
    
df["last_review"]=df["last_review"].apply(lambda x: date_replacement(x))
grouped = df.groupby("last_review")
last_review_price_grouped = grouped["log_price"]
last_review_price = last_review_price_grouped.agg([np.mean,np.median,np.max, np.std]).sort_values("mean")
print(last_review_price)

#convert last review to dummies
df = pd.concat([df, pd.get_dummies(df["last_review"], drop_first=False)], axis=1)
df.drop(["last_review"],axis=1, inplace=True)

#import necessary libraries
import nltk
import os
import nltk.corpus
from nltk import ne_chunk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

#initiate stopwords
a = set(stopwords.words('english'))
#obtain text
text = df["name"].iloc[10]
#tokenize text
text1 = word_tokenize(text.lower())
#create a list free of stopwords
no_stopwords = [x for x in text1 if x not in a]
#lemmatize the words
lemmatizer = WordNetLemmatizer() 
lemmatized = [lemmatizer.lemmatize(x) for x in no_stopwords]

def unique_words1(dwelling):
    
    apt = df[df[dwelling]==1]["name"]
    a = set(stopwords.words('english'))
    words = []
    # append each to a list
    for lis in range(0, len(apt)):
        listing = apt.reset_index().iloc[lis,1]
        #tokenize text
        text1 = word_tokenize(listing.lower())
        #create a list free of stopwords
        no_stopwords = [x for x in text1 if x not in a]
        #lemmatize the words
        lemmatized = [lemmatizer.lemmatize(x) for x in no_stopwords]
        no_punctuation = [x.translate(str.maketrans('','',string.punctuation)) for x in lemmatized]
        no_digits = [x.translate(str.maketrans('','',"0123456789")) for x in no_punctuation ]
        for item in no_digits:
            words.append(item)


    #create a dictionary
    unique={}
    for word in words:
        if word in unique:
            unique[word] +=1
        else:
            unique[word] = 1

    #sort the dictionary
    a=[]
    b=[]

    for key, value in unique.items():
        a.append(key)
        b.append(value)

    aa=pd.Series(a)
    bb=pd.Series(b)    

    comb=pd.concat([aa,bb],axis=1).sort_values(by=1, ascending=False).copy()

    return comb

#apply the function
private = unique_words1("Private room")
home = unique_words1("Entire home/apt")
shared = unique_words1("Shared room")

words_private = private.iloc[1:,1]
words_home = home.iloc[1:,1] 
words_shared = shared.iloc[1:,1] 

#plot the results
plt.plot(words_shared.reset_index()[1], label="shared")
plt.plot(words_private.reset_index()[1], label ="private")
plt.plot(words_home.reset_index()[1], label="Entire home/apt")
plt.xlim(0,200)
plt.ylabel("WordFrequency")
plt.xlabel("Word position on the list")
plt.legend()
plt.show()






