import os,re,time
import pandas as pd
import numpy as np
from multiprocessing import Pool
import multiprocessing
import argparse
import wget
import urllib.request
from urllib.parse import urlparse
import requests


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

text_suff = set(["doc", "docx", "pdf", "txt"])
audio_suff = set(["mp3", "wav"])


def download_audio_and_text_sub(link):
    #filename = wget.filename_from_url(link)
    #print(filename)
    #if link.split('.')[-1] in text_suff:
    #    try:
    #        wget.download(link, args.output_book_dir)
    #    except:
    #        print("ERROR in text link : ", str(link))
            
    if link.split('.')[-1] in audio_suff:
        try:
            #wget.download(link, args.output_audio_dir)

            parsed_url = urlparse(link)
            filename = os.path.basename(parsed_url.path)
            # 文件在本地保存的名字和路径
            local_filename = os.path.join(args.output_audio_dir, filename)

            response = requests.get(link, timeout=20)
                
            # 将获取的音频内容写入文件
            with open(local_filename, 'wb') as audio_file:
                audio_file.write(response.content)


        except:
            print("ERROR in audio link: ", str(link))
    else:
        print("ERROR : ", str(link))
    
def download_audio_and_text():
    ### 读取文本音频 .xlsx 文件
    df = pd.read_excel(io=args.input_file)
    mp3_dict = set([])
    for index, row in df.iterrows():
        if str(row["音频"]) != 'nan':
            mp3_dict.add(str(row["音频"]).strip())

    print("nb of mp3 :  ", mp3_dict)
    with Pool(processes=multiprocessing.cpu_count()-3) as p:
        p.map(download_audio_and_text_sub, list(mp3_dict))
    
    #c = 0
    #for i in download_list:
    #    print(i)
    #    if c<= 100:
    #        download_audio_and_text_sub(i)
    #        c += 1
    #    else:
    #        break

def set_dirs():
    not os.path.exists(args.output_audio_dir) and os.makedirs(args.output_audio_dir)
    #not os.path.exists(args.output_book_dir) and os.makedirs(args.output_book_dir)

if __name__ == "__main__":
    """
    usage
    python wav_preprocess.py -i input_dir -o output_dir
    """
    start_time = time.time()  # 统计时间
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str)
    parser.add_argument('-oa', '--output_audio_dir', type=str)
    #parser.add_argument('-ob', '--output_book_dir', type=str)
    args = parser.parse_args()
    set_dirs()
    #download_audio_and_text_sub("000001.wav")
    download_audio_and_text()
    #rm_dirs()
    print('time elapsed: ', time.time() - start_time, 's')



##############  文本音频整体说明 .xlsx 文件格式   ##############
### "任务id"(only one : 1581677729313505282)
### "位置"(所选的匹配文本里出现的位置, 总文本的第几个字符. only one :)
### "文本"(文本, only one : “怎么回事？”)
### "音频"(音频链接, only one)
### "书id"(书籍的ID，only one :1470713516915736577)
### "书链接"(书本或者章节内容的文本链接, more than one .docx/.doc/.pdf file)
##############  文本音频整体说明 .xlsx 文件格式   ##############
