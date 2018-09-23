import os
import csv
import numpy as np
import pandas

os.chdir("C:\\Users\\cd\\Desktop")

"""
books = np.loadtxt(open("books.csv", "rb"), delimiter=",", skiprows=1,
                   dtype={'names': ('book_id','goodreads_book_id','best_book_id','work_id','books_count','isbn','isbn13','authors','original_publication_year','original_title','title','language_code','average_rating','ratings_count','work_text_reviews_count','ratings_1','ratings_2','ratings_3','ratings_4','ratings_5','image_url','small_image_url'),
                          'formats': (
                              'i8',
                              'i4'
                              ,'i4','i4',
                                      'i4','i8','S','S','i4',
                                      'S','S','S','f4','i4','i4','i4','i4','i4','i4','i4','i4','S','S')})
"""

[tag_ids, tag_names] = np.loadtxt(open("tags.csv", "rb"), delimiter=",", skiprows=1, usecols=(0,1),
                  dtype='int,str', unpack=True)
print(tag_ids)
print(tag_names)
#tag_id_counts = np.vstack((tag_ids, np.zeros(34252, 'int')))
#print(tag_id_counts)
#print(tag_id_counts[0])

book_tags = np.loadtxt(open("book_tags.csv", "rb"), delimiter=",", skiprows=1, usecols=(0,1,2),
                  dtype="int")
print(book_tags)

#print(tag_id_counts[0].size)

tag_id_counts = []
for i in range(tag_ids.size):
    tag_id_counts.append(0)

for line in book_tags:
        tag_id_counts[line[1]-1] += line[2]
print(tag_id_counts)
np.savetxt('tag_counts.csv', tag_id_counts, delimiter='\n')


