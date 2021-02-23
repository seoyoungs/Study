# -*- coding:utf-8 -*-
## 머신러닝으로 하기


import gzip

path = 'C:/data/personal/listings.csv.gz'

# 그냥 읽으면 바이너리값 그대로 출력
with open(path,'rb') as f:
    print(f.readlines())  # [b'\x1f\x8b\x08\x08\xdb\x85\xc5[\x00\x03test1.csv\x00\r\xc6\xb9\x11\x00 \x0c\x03\xc1\xdc3\xea\xe4\x02dl\x9e\xfe\x1b\x83\x8d\xd6$SQ4K\xb19\\\x85\x076\xce\xbf\x89\x0b\xb7\xe2\x01N,\x9a\x1c)\x00\x00\x00']


# gzip으로 읽으면 파일 원본 내용 출력
with gzip.open(path,'rb') as f:
    print(f.readlines())  # [b'1,2,3\r\n', b'4,5,6\r\n', b'7,8,9\r\n', b'10,11,12\r\n', b'13,14,15\r\n']

import pandas as pd

dc_listings = pd.read_csv(path)
print(dc_listings.shape)#(4107, 74)

print(dc_listings.head())
'''
     id                        listing_url  ...  calculated_host_listings_count_shared_rooms reviews_per_month
0  9419  https://www.airbnb.com/rooms/9419  ...                                            0              1.17
1  9531  https://www.airbnb.com/rooms/9531  ...                                            0              0.38
2  9534  https://www.airbnb.com/rooms/9534  ...                                            0              0.50
3  9596  https://www.airbnb.com/rooms/9596  ...                                            0              0.84
4  9909  https://www.airbnb.com/rooms/9909  ...                                            0              0.55
'''