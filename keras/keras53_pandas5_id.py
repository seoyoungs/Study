import pandas as pd
###판다스만 해당되는 이야기

df = pd.DataFrame([[1,2,3,4], [4,5,6,7], [7,8,9,10]],
                  columns=list('abcd'), index=('가', '나', '다'))
# print(df)
#판다스는 열이 우선

df2 = df
#df 건드리는거 방지를 위해
df2['a'] =100 #a라는 column을 100으로 부여

# print(df2)
# print(df)
# print(id(df), id(df2))
'''
     a  b  c   d
가  100  2  3   4
나  100  5  6   7
다  100  8  9  10
     a  b  c   d
가  100  2  3   4
나  100  5  6   7
다  100  8  9  10

이렇게 df를 안건드리기 위해 df2로 했는데 df도 바뀌었다
왜? pandas는 그냥 이름만 바뀐것
pandas에 한해서만 그렇다

print(id(df), id(df2))
2347858167216 2347858167216
이렇게 id가 같다.

그럼 아예 전 데이터 영향없이 새로운것 만들려면???
'''

df3= df.copy()
df2['b']=333

# print(df)
# print(df2)
# print(df3)
'''
     a    b  c   d
가  100  333  3   4
나  100  333  6   7
다  100  333  9  10
     a    b  c   d
가  100  333  3   4
나  100  333  6   7
다  100  333  9  10
     a  b  c   d
가  100  2  3   4
나  100  5  6   7
다  100  8  9  10

df3= df.copy()
df3영향 안받음
'''
df = df + 99
print(df)
print(df2)

'''
     a    b    c    d 
가  199  432  102  103
나  199  432  105  106
다  199  432  108  109
     a    b  c   d
가  100  333  3   4
나  100  333  6   7
다  100  333  9  10

근데 이렇게 덧셈은 영향 안받는다. 
'''

