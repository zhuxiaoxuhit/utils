import os,re,time,shutil
import pandas as pd
from numpy import dtype
import numpy as np
from multiprocessing import Pool
import multiprocessing
import argparse
import wget
import codecs

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

text_suff = set(["doc", "docx", "pdf"])
audio_suff = set(["mp3", "wav"])


def prepare_data_sub(link):
    pass 

def prepare_data():
    # 读取文本音频 .xlsx 文件
    df = pd.read_excel(io=args.input_file, dtype=dtype)
    # book_dict : {book1 : [spk1, spk2] , book2 : [spk1, spk2, spk3]}
    # utter_dict : {book1 : {spk1 : 12345, spk2 : 12236} , book2 : {spk1 : 27634, spk2 : 12, spk3 : 88376}}
    book_dict = {} 
    utter_dict = {} 
    book_docs_list = []
    #books_links_prev = '0'
    #pattern = re.compile(r'【.*?】', re.S)
    prev_no_end = False
    now_no_end = False
    for index, row in df.iterrows():
        # 当前句子没有终结的情况：
        # 1 上一句没终结，这一句句尾不是”
        # 2 不管上一句是否终结，这一句以“开头，但是不以”结尾
        begin = str(row["文本"]).strip()[0]
        end = str(row["文本"]).strip()[-1]
        #if prev_no_end == True and end != "”" or begin == "“" and end != "”":
        #    now_no_end = True
        #else:
        #    now_no_end = False
        #
        #if end == "”":
        #    now_no_end = False
        #
        #
        #if prev_no_end == False and begin != "“" and end != "”":
        #    prev_no_end = now_no_end
        #    continue
        #elif prev_no_end == False and end != "”":
        #    now_no_end = False
        #prev_no_end = now_no_end

        #if begin != "“" or end != "”":
        #    continue
        if len(str(row["文本"]).strip()) <= 6:
            continue

        book_id = str(row["书id"])
        if str(row["音频"]) != 'nan' and str(row["书链接"]) != 'nan' and str(row["书id"]) != 'nan':
            books_links = str(row["书链接"])
            #if books_links_prev != books_links:
            if "," in str(row["书链接"]).strip().strip(","):
                book_docs_list = [os.path.basename(i) for i in str(row["书链接"]).strip().strip(",").split(",")]
            else:
                book_docs_list = [str(row["书链接"]).strip().strip(",").strip()]
            
            # exit_flag用于跳出外层循环: 如果找到句子在文本中对应的位置，则令exit_flag=True
            exit_flag = False
            for book_doc_file in book_docs_list:
                book_txt_file = os.path.join(args.input_book_dir, book_doc_file.split('.')[0] + '.txt') 
                if os.path.exists(book_txt_file): 
                    try:
                        lines = open(book_txt_file).readlines()
                        #lines = codecs.open(book_txt_file, encoding='gbk').readlines()
                    except:
                        print("ERROR IN OPEN TEXT FILE: ", book_txt_file)
                        continue
                    # 文本应大于30行(存在文本转的时候大批量保存为一行, 应舍弃) 
                    if len(lines) <= 30:
                        continue
                    for col in range(len(lines)):
                        # 匹配到对应的行
                        if str(row["文本"]).strip() in lines[col]:
                            exit_flag = True
                            location = lines[col].strip().find(str(row["文本"]).strip())
                            left_loc = lines[col].strip().rfind("“", 0, location)
                            right_loc = lines[col].strip().rfind("”", 0, location)
                            right_mid = lines[col].strip().rfind("】", 0, location)
                            if right_mid == -1:
                                break
                            if left_loc != -1 and left_loc > right_loc and right_mid == left_loc - 1:
                                print(str(row["文本"]).strip(), "    对应的文本是: ", book_txt_file, "  : 1111")
                            elif str(row["文本"]).strip()[0] == "“" and location != 0 and right_mid == location - 1:
                                print(str(row["文本"]).strip(), "    对应的文本是: ", book_txt_file, "  : 2222")
                            elif str(row["文本"]).strip()[-1] == "”" and left_loc > right_loc and right_mid == left_loc - 1:
                                print(str(row["文本"]).strip(), "    对应的文本是: ", book_txt_file, "  : 3333")

                                

                    if exit_flag is True:
                        break


def set_dirs():
    not os.path.exists(args.output_dir) and os.makedirs(args.output_dir)

if __name__ == "__main__":
    """
    usage
    python process_pangbai.py -i input_file -ia input_audio_dir -ib input_book_dir 
    """
    start_time = time.time()  # 统计时间
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str)
    parser.add_argument('-ia', '--input_audio_dir', type=str)
    parser.add_argument('-ib', '--input_book_dir', type=str)
    args = parser.parse_args()
    #set_dirs()
    prepare_data()
    #rm_dirs()
    print('time elapsed: ', time.time() - start_time, 's')



######  文本音频整体说明 .xlsx 文件格式   ######
# "任务id"(only one : 1581677729313505282)
# "位置"(所选的匹配文本里出现的位置, 总文本的第几个字符. only one :)
# "文本"(文本, only one : “怎么回事？”)
# "音频"(音频链接, only one)
# "书id"(书籍的ID，only one :1470713516915736577)
# "书链接"(书本或者章节内容的文本链接, more than one .docx/.doc/.pdf file)
######  文本音频整体说明 .xlsx 文件格式   ######



